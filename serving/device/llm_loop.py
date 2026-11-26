"""llm_loop.py — the step-driven LLM engine loop, shared by both isolation modes.

We drive vLLM's LLMEngine ourselves (add_request / step), as
experiments/lora_fair_share does, instead of letting AsyncLLMEngine run it.
That puts admission and charging in our hands at every step:

    admit   while the engine holds fewer than max_running + queue_k requests,
            the policy (scheduler.py: FifoPolicy, TokenSTFQPolicy, ...) picks
            the next request from the per-task queues and it is added to the
            engine. vLLM's own waiting queue never holds more than queue_k, so
            its FCFS order cannot override the policy, and the engine's batch
            is kept full (work conserving).
    step    one engine iteration; decoded tokens are charged to their tasks
            (policy.on_decode, on decode steps); finished requests are handed
            to on_done.

The loop does not own a thread. Shared mode runs it in a thread of its own
(runtime.VLLMRuntime); process mode runs it as the backbone process's main
loop (backbone_proc.py), fed from the lanes' slots. Requests are
RequestEnvelopes in a TenantQueues; how a request's prompt ids are read
(prompt_of) and where results go (on_done / on_error) is the caller's.
"""
from __future__ import annotations

import time
from collections import defaultdict

# A request the engine stops reporting for this many consecutive steps while
# it has nothing unfinished is lost (should not happen): fail it, don't hang.
_LOST_AFTER_STEPS = 200


class LLMStepLoop:

    def __init__(self, runtime, policy, queues, max_running: int, queue_k: int,
                 prompt_of, on_done, on_error, name: str = "LLMStepLoop"):
        """
        runtime    VLLMRuntime loaded with engine="step" (llm_add, llm_step, ...)
        policy     a scheduler.py policy; on_decode(task, n) is called if it has one
        queues     TenantQueues of RequestEnvelopes (task, n_prompt attribute)
        prompt_of  env -> prompt token ids (list of int)
        on_done    (env, token_ids, admit_ns, end_ns) for a finished request
        on_error   (envs, why) for requests that failed
        """
        self.runtime = runtime
        self.policy = policy
        self.queues = queues
        self.max_running = max(1, int(max_running))
        self.queue_k = max(0, int(queue_k))
        self.prompt_of = prompt_of
        self.on_done = on_done
        self.on_error = on_error
        self.name = name
        self._running: dict = {}      # engine request id -> [env, admit_ns, n_tokens_seen]
        self._next_id = 0
        self._idle_steps = 0
        print(f"[{name}] policy={type(policy).__name__} max_running={self.max_running} "
              f"queue_k={self.queue_k}")

    # -- state ---------------------------------------------------------------

    def has_work(self) -> bool:
        return bool(self._running) or self.queues.pending_count() > 0

    def n_running(self) -> int:
        return len(self._running)

    # -- one iteration ---------------------------------------------------------

    def admit(self) -> None:
        """Top the engine up to max_running + queue_k, in policy order."""
        room = self.max_running + self.queue_k - self.runtime.llm_num_unfinished()
        while room > 0 and self.queues.pending_count() > 0:
            picked = self.queues.select_batch(self.policy, room)
            if not picked:
                return
            for env in picked:
                rid = f"r{self._next_id}"
                self._next_id += 1
                try:
                    ids = self.prompt_of(env)
                    self.runtime.llm_add(rid, ids, env.task)
                except Exception as exc:
                    self.on_error([env], f"admit_error: {str(exc)[:200]}")
                    continue
                self._running[rid] = [env, time.time_ns(), 0]
            room -= len(picked)

    def step(self) -> None:
        """One engine step, if anything is in flight."""
        if not self._running:
            return
        try:
            outs = self.runtime.llm_step()
        except Exception as exc:
            print(f"[{self.name}] Engine step error: {exc}")
            self.fail_all(f"engine_error: {str(exc)[:200]}")
            return
        end_ns = time.time_ns()
        new = defaultdict(int)
        is_prefill = False
        for o in outs:
            entry = self._running.get(o.request_id)
            if entry is None:
                continue
            env, admit_ns, seen = entry
            ids = o.outputs[0].token_ids
            n = len(ids)
            if seen == 0:
                is_prefill = True
            new[env.task] += n - seen
            entry[2] = n
            if o.finished:
                del self._running[o.request_id]
                self.on_done(env, list(ids), admit_ns, end_ns)
        # Decode tokens are charged per step, prompt tokens at admission
        # (TokenSTFQPolicy.select); a step that ran a prefill is not charged
        # again, as in lora_fair_share.
        on_decode = getattr(self.policy, "on_decode", None)
        if on_decode is not None and not is_prefill:
            for task, k in new.items():
                on_decode(task, k)
        # Safety net: requests we track but the engine no longer has.
        if self._running and not outs and self.runtime.llm_num_unfinished() == 0:
            self._idle_steps += 1
            if self._idle_steps >= _LOST_AFTER_STEPS:
                envs = [e[0] for e in self._running.values()]
                self._running.clear()
                self._idle_steps = 0
                self.on_error(envs, "lost_in_engine")
        else:
            self._idle_steps = 0

    def run_once(self) -> bool:
        """admit + step. Returns False if there was nothing to do."""
        if not self.has_work():
            return False
        self.admit()
        self.step()
        return True

    # -- failure ---------------------------------------------------------------

    def abort_where(self, pred) -> list:
        """Abort in-flight requests whose envelope matches pred (their KV
        blocks are freed); return their envelopes. No callback is made."""
        rids = [rid for rid, e in self._running.items() if pred(e[0])]
        if rids:
            try:
                self.runtime.llm_abort(rids)
            except Exception as exc:
                print(f"[{self.name}] abort error: {exc}")
        return [self._running.pop(rid)[0] for rid in rids]

    def fail_all(self, why: str) -> None:
        envs = self.abort_where(lambda env: True)
        if envs:
            self.on_error(envs, why)
