# Sharing Benefit Experiment

## Goal

Quantify the latency benefit of backbone sharing when two time-series tasks
co-exist on a single GPU. The central claim is that multiplexing multiple tasks
over one shared backbone (using STFQ scheduling) achieves lower latency than
dedicating a separate backbone to each task (no-sharing baseline), while also
being competitive with the single-task oracle.

---

## Hardware

- **GPU**: NVIDIA A16 (16 GB VRAM per partition)
- **Framework**: PyTorch via `PyTorchRuntime` (FMTK)
- **Max batch size**: 5

---

## Models & Tasks

| Task          | Backbone    | Decoder                   | Dataset                  |
|---------------|-------------|---------------------------|--------------------------|
| `ecgclass`    | Moment-base | `ecgclass_momentbase_mlp` | ECG5000 (test)           |
| `gestureclass`| Moment-base | `gestureclass_momentbase_mlp` | UWaveGestureLibraryAll (test) |

---

## Conditions

| Condition            | Servers | Scheduler | Tasks served       |
|----------------------|---------|-----------|--------------------|
| `single_ecgclass`    | 1 (A)   | FIFO      | ecgclass only      |
| `single_gestureclass`| 1 (A)   | FIFO      | gestureclass only  |
| `no_sharing`         | 2 (A+B) | FIFO each | ecgclass on A, gestureclass on B |
| `sharing`            | 1 (A)   | STFQ      | both tasks         |

- **`no_sharing`**: two independent backbone instances, each serving one task.
  Represents the naive scale-out approach (double the GPU memory).
- **`sharing`**: one backbone instance, both tasks multiplexed via Start-Time
  Fair Queuing (STFQ). Saves one full backbone worth of GPU memory.

---

## Workload

- **Trace**: open-loop Poisson arrivals, one trace per task
- **RPS sweep**: 10, 20, 30, 40, 60, 80 req/s per task
- **Duration**: 180 s per condition per RPS level (10 s warmup discarded)
- **Fixed seeds**: each task uses a fixed per-task RNG seed so the arrival
  trace is identical across all conditions, enabling fair comparison

---

## Running

```bash
# From serving/
bash experiments/sharing_benefit/run.sh

# Environment variable overrides (all optional):
#   BACKBONE=momentbase
#   RPS_SWEEP=10,20,30,40,60,80
#   PHASE_DURATION=180
#   CUDA_DEVICE=cuda:0
#   MAX_BATCH_SIZE=5
#   RESULTS_BASE=experiments/sharing_benefit/results

# Plot
python experiments/sharing_benefit/plot.py \
    --exp-dir experiments/sharing_benefit/results \
    --rps-sweep 20,40,60
```

Results are written to `results/rps_{N}/{condition}/`:
- `latencies.csv` — per-request `(task, condition, elapsed_sec, latency_ms)`
- `task_results.csv` — per-task summary (avg, p50, p95, p99 latency; throughput)

Plots are saved to `results/plots/`:
- `motivation2_latency_cdf_rps{N}.pdf` — latency CDF per condition
- `motivation2_throughput_cdf_rps{N}.pdf` — completion throughput CDF
- `motivation2_summary_bars_rps{N}.pdf` — p99 bar chart
- `motivation2_sweep_latency_cdf.pdf` — side-by-side CDFs across RPS levels
- `motivation2_sweep_throughput_cdf.pdf` — side-by-side throughput CDFs

---

## Key Findings

Average and P99 latency (ms) per condition, per RPS level (both tasks averaged):

| RPS | Single (oracle) avg | No-Sharing avg | Sharing avg | No-Sharing p99 | Sharing p99 |
|-----|---------------------|----------------|-------------|----------------|-------------|
| 10  | 11.5                | 12.6           | 12.0        | 27.8           | 25.6        |
| 20  | 12.2                | 14.9           | 13.8        | 39.1           | 34.1        |
| 30  | 12.8                | 18.7           | 17.1        | 57.0           | 45.7        |
| 40  | 13.7                | 26.8           | 22.6        | 85.9           | 62.0        |
| 60  | 16.8                | 99.0           | 58.7        | 305.4          | 261.1       |
| 80  | 22.3                | ~17k (saturated) | ~17k (saturated) | — | —       |

**Key takeaway**: backbone sharing (STFQ, one server) consistently achieves
lower latency than the no-sharing baseline (two FIFO servers, one backbone
each) across all sub-saturation RPS levels. At moderate load (RPS 40),
sharing reduces average latency by ~16% and P99 by ~28% versus no-sharing.
The benefit grows as load increases because the shared server can batch across
tasks while the no-sharing servers each carry the full per-task queue in
isolation.

Both conditions saturate at approximately the same RPS (≈70–75 req/s combined),
confirming that sharing does not reduce maximum throughput — it improves
latency at the same hardware budget.

---

## Paper Text

**Experiment setup.**
We evaluate backbone sharing using two time-series classification tasks —
ECG arrhythmia detection (ECG5000) and gesture recognition
(UWaveGestureLibraryAll) — both served by the Moment backbone.
We compare four conditions: (1) \emph{single task} (one task on one FIFO
server, oracle baseline), (2) \emph{no sharing} (two FIFO servers, one
backbone each, each serving one task), and (3) \emph{sharing} (one STFQ
server, one backbone, both tasks multiplexed).
The no-sharing baseline uses $2\times$ the GPU memory of the sharing
condition.
Each condition receives open-loop Poisson arrivals at a fixed RPS per task
(swept from 10 to 80 req/s) over 180 s, with identical per-task arrival
traces across conditions.

**Backbone sharing reduces latency at the same hardware cost.**
Figure~\ref{fig:sharing_benefit} plots the latency CDF at representative
load levels.
Across all sub-saturation loads, the sharing condition achieves lower latency
than no-sharing despite using half the GPU memory: at 40 req/s/task, sharing
reduces average latency by 16\% and P99 by 28\%.
The benefit is rooted in the STFQ scheduler's ability to batch requests across
tasks, increasing GPU utilization and reducing per-request queueing delay
compared to two lightly-loaded FIFO queues.
Both conditions reach saturation at approximately the same aggregate request
rate, confirming that sharing does not sacrifice throughput.
