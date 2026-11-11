# RTVSntask/llm — response time vs. number of tasks (vLLM multi-LoRA)

LLM analog of [`RTVSntask/vision`](../vision). Each "task" is a LoRA adapter
over one shared Qwen2.5 backbone; the sweep asks how response time grows with
the number of co-located tasks, and where each placement strategy runs out of
GPU.

## Conditions

Run by default — the same three arms, in the same order, as
`RTVSntask/vision/run.sh`:

| Condition | Servers | GPU memory | Load |
|---|---|---|---|
| `no_sharing_tpc` | N vLLM, each pinned to a TPC set | `GPU_MEM_UTIL / N` | `rps` per task |
| `sharing` | 1 vLLM, all N LoRAs (Punica) | full `GPU_MEM_UTIL` | `rps` per task |
| `no_sharing` | N vLLM, 1 backbone copy each + 1 LoRA | `GPU_MEM_UTIL / N` (clamped ≥ 0.10) | `rps` per task |

Set `INCLUDE_TPC=0` to skip `no_sharing_tpc` (it needs libsmctrl).

Off by default:

| Condition | Servers | GPU memory | Load |
|---|---|---|---|
| `single_<task>` | 1 vLLM, 1 LoRA | full `GPU_MEM_UTIL` | **aggregate** `rps * ntasks` |

The single baseline drives one task at `rps * ntasks` so the GPU sees the same
total arrival rate as the multi-task arms. It is implemented but not run by
default, because vision never ran it either — no `single_*` directory exists in
any `RTVSntask` results set — and leaving it out keeps the figures comparable.
Opt in with `CONDITIONS=single,sharing,no_sharing`; note that path has never
been exercised against real results.

## The `no_sharing` wall is a result

Each `no_sharing` server loads its own full copy of the 1.5B backbone (~3 GB in
fp16) inside `GPU_MEM_UTIL/N` of the GPU. Past a handful of tasks there is no
room left for a KV cache and the server cannot start. That is the point of the
experiment, so failures are recorded rather than swallowed: when a condition
cannot be placed, `run.sh` writes

```
ntasks_<N>/rps_<R>/<condition>/failure.json
```

with a classified `reason` (`kv_cache_insufficient`, `cuda_oom`,
`insufficient_gpu_memory`, `port_conflict`, `adapter_missing`, `tpc_partition`),
the stage it failed at (e.g. `device_start_4_of_8` — it placed 4 of 8 servers),
the per-server memory split, and a 25-line log tail. The sweep then continues to
the next condition. `plot.py` draws those points as an X with the reason
annotated, and produces a per-condition placement ceiling chart.

## Task pool and `__appN` cycling

`NUM_TASKS_SWEEP` has **no upper bound**. When it exceeds the adapter pool
(`TASKS`, default `qwenA,qwenB`), adapters are cycled with a `__appN` suffix:

```
n=2 -> qwenA,qwenB
n=5 -> qwenA__app1,qwenB__app1,qwenA__app2,qwenB__app2,qwenA__app3
```

Each slot is a distinct task ID and gets its own LoRA slot in the engine —
`Pipeline.add_adapter` mints a fresh `adapter_<id>` per call and never dedupes
by path, so the same adapter directory registered twice is two independent
adapters. `base_task()` in `run.py` strips the suffix when resolving the
on-disk adapter directory.

## Prerequisites

```bash
python experiments/fair_share/llm/make_loras.py \
    --backbone qwen2.5-1.5b --names qwenA,qwenB
```

## Deploy timeout

`EdgeRuntimeClient.control()` defaults to a 120s gRPC deadline, which the vLLM
engine build blows past when weights are read over a networked filesystem —
the symptom is `DEADLINE_EXCEEDED` on the `load` call while the device log is
still printing `Loading safetensors checkpoint shards: 0%`. The server is fine;
only the RPC gave up. `DEPLOY_TIMEOUT` (default 1800s) sizes that deadline.

## Running

```bash
# smoke test
NUM_TASKS_SWEEP=2 RPS_SWEEP=2 PHASE_DURATION=90 \
    bash experiments/RTVSntask/llm/run.sh

# full sweep
NUM_TASKS_SWEEP=1,2,4,6,8,10 RPS_SWEEP=2 \
    bash experiments/RTVSntask/llm/run.sh

# skip the TPC arm (no libsmctrl)
INCLUDE_TPC=0 NUM_TASKS_SWEEP=1,2,4,6 RPS_SWEEP=2 \
    bash experiments/RTVSntask/llm/run.sh

# only the arm that scales, out to large N
CONDITIONS=sharing NUM_TASKS_SWEEP=1,2,4,8,16,32 \
    bash experiments/RTVSntask/llm/run.sh

python experiments/RTVSntask/llm/plot.py --results experiments/RTVSntask/llm/results
```

Results are resumable: a condition whose `latencies.csv` already exists is
skipped, so an interrupted sweep can be rerun. Delete a `failure.json` dir to
retry that point.

## Layout

```
results/
  ntasks_<N>/rps_<R>/<condition>/latencies.csv       per-request records
                                 task_results.csv    per-task summary
                                 run_config.json     what actually ran
                                 failure.json        (only if it could not be placed)
  ntasks_<N>/rps_<R>/trace.json                      shared arrivals, multi-task arms
  singles/rps_<aggregate>/trace.json                 shared arrivals, singles
  logs/                                              device server stdout
  plots/                                             plot.py output
```

Note the directory says `rps_<R>` (per-task) even for singles, which actually
ran at `R * N`; `run_config.json` records the true rate in `rps_applied` and
`rps_single_aggregate`.
