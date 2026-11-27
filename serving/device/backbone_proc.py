"""backbone_proc.py — Backbone process for process-isolation mode (design 3).

Ownership rule: every piece of GPU memory that crosses a process boundary is
OWNED by this (trusted) process and LENT to exactly one task process through
CUDA IPC. This process never maps memory a task owns, so a task that dies
cannot leave the backbone holding a dangling pointer.

Owns:
  - FM backbone weights (GPU), run on bb_stream: the TPC-masked stream from
    EdgeRuntimeApplication._setup_tpc when --tpc-mode is set, as shared mode
  - TenantQueues + the scheduling policy from EdgeRuntimeApplication._make_policy,
    driven as batch_former.py / batcher.py drive it
  - One POOL per lane (= per vFM), allocated when the task asks for it:
        x[N, *in_shape]      inputs, written by the task
        mask[N, *m_shape]    masks (time-series backbones), written by the task
        feat[N, *feat_shape] features, written here, read by the task
    N = config.slots_per_task. Slot j of a lane holds one request from input
    to features; it belongs to the task until the task sends it again.

Messages on sched_q[i] (task i -> here):
    ("alloc", x_shape, x_dtype, m_shape, m_dtype)  first request of the task:
        allocate lane i's pool (one dry-run forward gives the feature shape),
        answer ("pool", x, mask, feat) or ("pool_error", why) on done_q[i].
    ("slot", j, enqueued_at)  x[j] (and mask[j]) hold a request, already
        copied and synchronized by the task: queue it.
Messages on done_q[i] (here -> task i):
    ("pool", x, mask, feat) / ("pool_error", why)
    ("feat", feat)                after swap_backbone: new feature pool
    ("done", [j, ...], start_ns, proc_ns)   features of slots j are in feat[j]
    ("error", [j, ...], why)                the batch with slots j failed

Loop:
  1. poll(): read every attached lane's sched_q. A malformed message, or a
     slot number the lane does not own, drops that lane, never the backbone.
  2. Wait for the batch window; policy.select picks a cross-task batch.
  3. Gather the batch's inputs from the lanes' pools into backbone-owned
     memory (index_select), forward, scatter each row into its lane's feat
     pool (index_copy_), bb_stream.synchronize(), tell each lane.

Control messages arrive on ctrl_q: load / swap_backbone (replied on
ctrl_reply_q), attach (a lane now carries a task; replied) and detach (the
lane's task process is gone; not replied).
"""

import json
import math
import queue
import time
from contextlib import nullcontext
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class ControlMsg:
    type: str = "control"
    command: str = ""          # "load" | "swap_backbone" | "attach" | "detach"
    payload_json: str = ""
    seq: int = 0               # matched to reply on ctrl_reply_q


@dataclass
class SentinelMsg:
    type: str = "stop"


# ---------------------------------------------------------------------------
# Backbone process entry point
# ---------------------------------------------------------------------------

def backbone_proc_main(
    config,                    # RuntimeServerConfig
    device: str,
    sched_qs: list,            # lane i: task proc → backbone
    done_qs: list,             # lane i: backbone → task proc
    dyings: list,              # lane i: set when its task proc fails
    ctrl_q,                    # ControlMsg / SentinelMsg from isolated_app
    ctrl_reply_q,              # backbone_proc → isolated_app (control replies)
    run_stop,                  # set when the server stops
    stats_arr=None,            # shared array for Control("stats") (device.memstats)
):
    import os

    import torch
    import torch.multiprocessing  # noqa: F401  (CUDA tensors through queues)
    os.environ["CUDA_DEVICE"] = device

    from device.executors import make_executor
    from device.scheduler import RequestEnvelope, SABAPolicy, STFQPolicy, TenantQueues
    from device.memstats import task_mem_mb
    from device.server import EdgeRuntimeApplication
    from fmtk.logger import Logger

    dev = torch.device(device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    if stats_arr is not None:
        from device.memstats import BB_FIELDS, cuda_mem, start_writer
        start_writer(stats_arr, BB_FIELDS, lambda: cuda_mem(dev), run_stop)

    max_b = config.max_batch_size
    window_s = config.max_batch_wait_ms / 1000.0
    n_slots = config.slots_per_task or 2 * max_b
    n = len(sched_qs)

    # Same stream as shared mode: TPC-masked if configured, else a dedicated one.
    bb_stream = EdgeRuntimeApplication._setup_tpc(config) if cuda else None
    if cuda and bb_stream is None:
        bb_stream = torch.cuda.Stream(device=dev)
    policy = EdgeRuntimeApplication._make_policy(config)
    tenant_queues = TenantQueues()

    print(f"[BackboneProc] Started device={device} policy={type(policy).__name__} "
          f"max_batch_size={max_b} max_batch_wait_ms={config.max_batch_wait_ms} "
          f"lanes={n} slots_per_task={n_slots} task_mem_mb={config.task_mem_mb}")

    logger = Logger(device, "backbone_proc")

    task_of = [None] * n        # lane -> task name (None: not attached)
    pools = [None] * n          # lane -> {"key", "x", "mask", "feat", "n"}
    # Pools of dropped lanes, kept for reuse by a later lane with the same
    # input shape. A dead task may never release its CUDA IPC mapping, so the
    # memory is not freed; reusing it keeps a restarting task from leaking.
    spare_pools: dict = {}      # key -> [pool, ...]

    def stream_ctx():
        return torch.cuda.stream(bb_stream) if cuda else nullcontext()

    def sync_stream():
        if cuda:
            bb_stream.synchronize()

    # The model, behind one interface (executors.py): PyTorchExecutor runs a
    # batch per forward, VLLMExecutor is stepped by llm_loop.py.
    executor = make_executor(config, device, logger, stream_ctx, sync_stream, bb_stream)
    # Called with a lane being dropped (the LLM loop aborts its requests).
    drop_hooks: list = []

    # -----------------------------------------------------------------------
    # Lanes and pools
    # -----------------------------------------------------------------------

    def drop_lane(i: int, why: str) -> None:
        """Lane i's task is gone or misbehaved: never batch it again."""
        task = task_of[i]
        if task is None:
            return
        task_of[i] = None
        if task in tenant_queues._queues:
            q = tenant_queues._queues[task]
            # Only this lane's requests: a re-deployed task may be on another lane.
            for env in [e for e in q if e.req_id[0] == i]:
                q.remove(env)
        if pools[i] is not None:
            spare_pools.setdefault(pools[i]["key"], []).append(pools[i])
            pools[i] = None
        for hook in drop_hooks:
            hook(i)
        print(f"[BackboneProc] Dropped lane {i} (task={task}): {why}")

    def feature_spec(x_shape, x_dtype, m_shape, m_dtype):
        """(shape, dtype) of one request's output slot (executors.py)."""
        return executor.output_spec(x_shape, x_dtype, m_shape, m_dtype)

    def pool_slots(task, key, f_shape, f_dtype) -> int:
        """Slots in task's pool: --slots-per-task, or as many as fit in its
        --task-mem-mb budget (one slot = one request's input + mask + features)."""
        budget = task_mem_mb(config.task_mem_mb, task)
        if budget is None:
            return n_slots

        def nbytes(shape, dtype):
            return math.prod(shape) * torch.empty((), dtype=dtype).element_size()

        slot_b = nbytes(key[0], key[1]) + nbytes(f_shape, f_dtype)
        if key[2] is not None:
            slot_b += nbytes(key[2], key[3])
        n_i = int(budget * 2**20 // slot_b)
        if n_i < 1:
            raise RuntimeError(f"task_mem_mb {budget:g} MB is below one slot "
                               f"({slot_b / 2**20:.2f} MB)")
        return n_i

    def alloc_pool(i: int, x_shape, x_dtype, m_shape, m_dtype) -> None:
        """Allocate (or reuse) lane i's pool and lend it to the task."""
        if pools[i] is not None:
            raise RuntimeError("pool already allocated")
        if not executor.loaded:
            raise RuntimeError("backbone_not_loaded")
        key = (tuple(x_shape), x_dtype, tuple(m_shape) if m_shape is not None else None, m_dtype)
        f_shape, f_dtype = feature_spec(*key)
        n_i = pool_slots(task_of[i], key, f_shape, f_dtype)
        spares = spare_pools.get(key)
        pool = None
        if spares:
            for k, sp in enumerate(spares):
                if sp["n"] == n_i:
                    pool = spares.pop(k)
                    break
        if pool is not None:
            if tuple(pool["feat"].shape[1:]) != f_shape or pool["feat"].dtype != f_dtype:
                pool["feat"] = torch.empty((pool["n"], *f_shape), dtype=f_dtype, device=dev)
        else:
            pool = {
                "key": key,
                "n": n_i,
                "x": torch.empty((n_i, *key[0]), dtype=key[1], device=dev),
                "mask": (torch.empty((n_i, *key[2]), dtype=key[3], device=dev)
                         if key[2] is not None else None),
                "feat": torch.empty((n_i, *f_shape), dtype=f_dtype, device=dev),
            }
        pools[i] = pool
        done_qs[i].put(("pool", pool["x"], pool["mask"], pool["feat"]))
        mb = sum(t.numel() * t.element_size() for t in (pool["x"], pool["mask"], pool["feat"])
                 if t is not None) / 2**20
        print(f"[BackboneProc] Lane {i} (task={task_of[i]}) pool: {pool['n']} slots, "
              f"x{tuple(pool['x'].shape[1:])} feat{f_shape}, {mb:.1f} MB")

    def handle_lane_msg(i: int, msg) -> None:
        kind = msg[0]
        if kind == "alloc":
            _, x_shape, x_dtype, m_shape, m_dtype = msg
            try:
                alloc_pool(i, x_shape, x_dtype, m_shape, m_dtype)
            except Exception as exc:        # OOM included: this lane only
                done_qs[i].put(("pool_error", f"{type(exc).__name__}: {str(exc)[:200]}"))
            return
        if kind == "slot":
            # ("slot", j, enqueued_at[, n_prompt]): n_prompt only from LLM tasks
            j, enqueued_at = msg[1], msg[2]
            pool = pools[i]
            if pool is None or not isinstance(j, int) or not 0 <= j < pool["n"]:
                raise ValueError(f"slot {j!r} not owned by lane")
            env = RequestEnvelope(req_id=(i, j), task=task_of[i], x=pool["x"][j],
                                  mask=pool["mask"][j] if pool["mask"] is not None else None,
                                  question=None, enqueued_at=enqueued_at, future=None)
            if len(msg) > 3:
                n_prompt = msg[3]
                if not isinstance(n_prompt, int) or not 0 < n_prompt <= pool["x"].shape[1]:
                    raise ValueError(f"bad prompt length {n_prompt!r}")
                env.n_prompt = n_prompt
            tenant_queues.push(env)
            if isinstance(policy, STFQPolicy):
                policy.assign_start_time(env)
            return
        raise ValueError(f"unknown message {kind!r}")

    def poll() -> None:
        for i in range(n):
            if task_of[i] is None:
                continue
            if dyings[i].is_set():
                drop_lane(i, "task process failed")
                continue
            while task_of[i] is not None:
                try:
                    msg = sched_qs[i].get_nowait()
                except queue.Empty:
                    break
                except Exception as exc:
                    drop_lane(i, f"unreadable message: {str(exc)[:80]}")
                    break
                try:
                    handle_lane_msg(i, msg)
                except Exception as exc:
                    # A task sending garbage loses its lane; the backbone goes on.
                    drop_lane(i, f"bad message: {str(exc)[:80]}")

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    def poll_ctrl_q() -> bool:
        """Non-blocking check for control/stop messages. Returns True if stop."""
        while True:
            try:
                msg = ctrl_q.get_nowait()
            except queue.Empty:
                return False
            t = getattr(msg, "type", None)
            if t == "stop":
                return True
            if t == "control":
                handle_control(msg)

    def refresh_feature_pools() -> None:
        """After a backbone swap the feature shape may change: give every lane
        a new feature pool. Inputs are unchanged, so queued requests stay."""
        for i in range(n):
            pool = pools[i]
            if task_of[i] is None or pool is None:
                continue
            f_shape, f_dtype = feature_spec(*pool["key"])
            if tuple(pool["feat"].shape[1:]) != f_shape or pool["feat"].dtype != f_dtype:
                pool["feat"] = torch.empty((pool["n"], *f_shape), dtype=f_dtype, device=dev)
                done_qs[i].put(("feat", pool["feat"]))
        spare_pools.clear()

    def handle_control(msg: ControlMsg) -> None:
        payload = json.loads(msg.payload_json) if msg.payload_json else {}

        if msg.command == "attach":
            i, task = payload["lane"], payload["task"]
            task_of[i] = task
            pools[i] = None
            print(f"[BackboneProc] Attached lane {i} to task={task}")
            ctrl_reply_q.put({"seq": msg.seq, "status": f"attached_{task}"})
            return
        if msg.command == "detach":
            drop_lane(payload["lane"], "task process exited")
            return

        status = "ok"
        gpu_backbone_mem_mb = 0.0
        model_category = "tsfm"
        try:
            if msg.command == "load":
                backbone = payload["backbone"]
                print(f"[BackboneProc] Loading backbone={backbone}")
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                executor.load(backbone, payload)
                if cuda:
                    torch.cuda.synchronize(dev)
                    gpu_backbone_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
                print(f"[BackboneProc] Backbone loaded  gpu_alloc_peak={gpu_backbone_mem_mb:.1f} MB")
                status = f"loaded_{backbone}"
            elif msg.command == "swap_backbone":
                backbone = payload["backbone"]
                print(f"[BackboneProc] Swapping backbone → {backbone}")
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                executor.swap(backbone, payload)
                if cuda:
                    torch.cuda.synchronize(dev)
                    gpu_backbone_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
                refresh_feature_pools()
                status = f"swapped_{backbone}"
            else:
                status = f"unknown_command_{msg.command}"
            if executor.loaded:
                model_category = executor.model_category
        except Exception as e:
            status = f"error_{e}"
            print(f"[BackboneProc] Control error: {e}")
        gpu_backbone_reserved_mb = torch.cuda.memory_reserved(dev) / (1024 ** 2) if cuda else 0.0
        summary = {"backbone_process": {"gpu peak": gpu_backbone_mem_mb,
                                        "gpu reserved": gpu_backbone_reserved_mb}}
        reply = {"seq": msg.seq, "status": status, "logger_summary": str(summary),
                 "model_category": model_category}
        if executor.kind == "continuous" and executor.loaded:
            # what task processes need to tokenize the same way the engine would
            reply["tokenizer"] = executor.tokenizer_name()
            reply["max_prompt_len"] = config.llm_max_prompt_len
        ctrl_reply_q.put(reply)

    # -----------------------------------------------------------------------
    # Batches
    # -----------------------------------------------------------------------

    def by_lane(batch: list) -> dict:
        """lane -> slot numbers, in batch order."""
        lanes: dict = {}
        for env in batch:
            i, j = env.req_id
            lanes.setdefault(i, []).append(j)
        return lanes

    def fail_batch(batch: list, why: str) -> None:
        for i, js in by_lane(batch).items():
            if task_of[i] is not None:
                done_qs[i].put(("error", js, why))

    def run_batch(batch: list) -> None:
        start_ns = time.time_ns()
        t0 = time.monotonic()
        lanes = by_lane(batch)
        with stream_ctx(), torch.no_grad():
            # Gather inputs from the lanes' pools into backbone-owned memory,
            # lane by lane; rows of the batch follow the order of `lanes`.
            idx = {i: torch.tensor(js, dtype=torch.long, device=dev) for i, js in lanes.items()}
            bx = torch.cat([pools[i]["x"].index_select(0, idx[i]) for i in lanes])
            if all(pools[i]["mask"] is not None for i in lanes):
                bmask = torch.cat([pools[i]["mask"].index_select(0, idx[i]) for i in lanes])
            else:
                bmask = None
            feats = executor.forward(bx, bmask)
            # Scatter each lane's rows into its own feature pool.
            row = 0
            for i, js in lanes.items():
                k = len(js)
                pools[i]["feat"].index_copy_(0, idx[i], feats[row:row + k].to(pools[i]["feat"].dtype))
                row += k
            # Features must be written before any task reads them (host-side
            # queues do not order kernels across CUDA contexts).
            if cuda:
                bb_stream.synchronize()
            del bx, bmask, feats
        proc_ns = time.time_ns() - start_ns

        # As device/batcher.py: the batch's duration corrects the policy.
        duration_s = time.monotonic() - t0
        if isinstance(policy, SABAPolicy):
            policy.update_batch_duration(duration_s)
        if isinstance(policy, STFQPolicy):
            policy.update_after_execution([env.task for env in batch], duration_s,
                                          tenant_queues._queues)

        for i, js in lanes.items():
            if task_of[i] is not None:
                done_qs[i].put(("done", js, start_ns, proc_ns))

        if config.verbose_batch_logs:
            print(f"[BackboneProc] Batch done size={len(batch)} "
                  f"tasks={[env.task for env in batch]} proc_time_ns={proc_ns}")

    # -----------------------------------------------------------------------
    # LLM loop (executor.kind == "continuous"): llm_loop.py, fed from slots
    # -----------------------------------------------------------------------

    def run_llm_loop() -> None:
        from device.llm_loop import LLMStepLoop
        from device.server import EdgeRuntimeApplication

        def prompt_of(env):
            i, j = env.req_id
            return pools[i]["x"][j, :env.n_prompt].tolist()

        def on_done(env, ids, admit_ns, end_ns):
            i, j = env.req_id
            if task_of[i] is None or pools[i] is None:
                return
            feat = pools[i]["feat"]
            k = min(len(ids), feat.shape[1] - 1)
            row = torch.tensor([k] + ids[:k], dtype=feat.dtype)
            feat[j, :k + 1].copy_(row)
            # Tokens must be in the slot before the task reads them.
            if cuda:
                torch.cuda.current_stream(dev).synchronize()
            done_qs[i].put(("done", [j], admit_ns, end_ns - admit_ns))

        def on_error(envs, why):
            fail_batch(envs, why)

        llm = LLMStepLoop(executor.runtime, policy, tenant_queues,
                          max_running=EdgeRuntimeApplication._llm_max_running(config),
                          queue_k=config.llm_queue_k, prompt_of=prompt_of,
                          on_done=on_done, on_error=on_error, name="BackboneProc:LLM")
        # A dropped lane's in-flight requests leave the engine (KV freed).
        drop_hooks.append(lambda i: llm.abort_where(lambda env: env.req_id[0] == i))

        with (torch.cuda.stream(bb_stream) if executor.runtime._cuda_stream is not None
              else nullcontext()):
            while not run_stop.is_set():
                if poll_ctrl_q():
                    print("[BackboneProc] Received stop, exiting")
                    break
                poll()
                for i in range(n):
                    if task_of[i] is not None and dyings[i].is_set():
                        drop_lane(i, "task process failed")
                if not llm.has_work():
                    time.sleep(0.0002)
                    continue
                if not executor.loaded:
                    fail_batch(tenant_queues.select_batch(policy, tenant_queues.pending_count()),
                               "backbone_not_loaded")
                    continue
                llm.run_once()
        llm.fail_all("backbone_stopping")

    if executor.kind == "continuous":
        run_llm_loop()
        print("[BackboneProc] Exited")
        return

    # -----------------------------------------------------------------------
    # Scheduler loop
    # -----------------------------------------------------------------------
    while not run_stop.is_set():
        if poll_ctrl_q():
            print("[BackboneProc] Received stop, exiting")
            break
        poll()
        if tenant_queues.pending_count() == 0:
            time.sleep(0.0002)
            continue

        # Batch collection window, as BatchFormer.next_batch().
        deadline = time.monotonic() + window_s
        while tenant_queues.pending_count() < max_b and time.monotonic() < deadline:
            time.sleep(0.0002)
            poll()

        # Last look before reading the tasks' slots.
        for i in range(n):
            if task_of[i] is not None and dyings[i].is_set():
                drop_lane(i, "task process failed")

        batch = tenant_queues.select_batch(policy, max_b)
        if not batch:
            continue

        if not executor.loaded:
            fail_batch(batch, "backbone_not_loaded")
            continue
        try:
            run_batch(batch)
        except Exception as e:
            print(f"[BackboneProc] Forward error: {e}")
            if "out of memory" in str(e):   # a real OOM: the backbone exits, every task with it
                print("[BackboneProc] CUDA OOM: exiting", flush=True)
                os._exit(1)
            fail_batch(batch, str(e))

    print("[BackboneProc] Exited")


# ---------------------------------------------------------------------------
# Helper to spawn
# ---------------------------------------------------------------------------

def start_backbone_process(ctx, config, device: str, sched_qs: list, done_qs: list,
                           dyings: list, ctrl_q, ctrl_reply_q, run_stop, stats_arr=None):
    p = ctx.Process(
        target=backbone_proc_main,
        args=(config, device, sched_qs, done_qs, dyings, ctrl_q, ctrl_reply_q, run_stop,
              stats_arr),
        daemon=True,
        name="backbone_proc",
    )
    p.start()
    return p
