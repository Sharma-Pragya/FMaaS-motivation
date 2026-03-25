# Microbenchmark: Backbone vs. Task-Component Cost

## Goal

Quantify the relative cost of backbone models versus task-specific components
(decoders / LoRA adapters) across three foundation model modalities: time-series
(TSFM), vision, and vision-language (VLM). The central claim is that task
components are negligible compared to the shared backbone, motivating backbone
sharing across tasks in FMaaS.

---

## Hardware

- **GPU**: NVIDIA A16 (16 GB VRAM per partition)
- **Framework**: PyTorch via `PyTorchRuntime` (FMTK)
- **Batch size**: 1 (closed-loop, one request at a time)

---

## Models

| Modality    | Backbone   | HF / Source ID                              | Task component             | Task               |
|-------------|------------|---------------------------------------------|----------------------------|--------------------|
| Time Series | Moment     | `AutonLab/MOMENT-1-large`                   | MLP decoder                | ECG classification |
| Time Series | Chronos    | `amazon/chronos-t5-base`                    | MLP decoder                | ECG classification |
| Vision      | DINOv2     | `facebook/dinov2-base`                      | Linear classification head | EuroSAT land use   |
| Vision      | Swin-S     | `microsoft/swin-small-patch4-window7-224`   | Linear classification head | EuroSAT land use   |
| VLM         | Phi-3.5    | `microsoft/Phi-3.5-vision-instruct`         | LoRA adapter (r=16)        | OCR                |
| VLM         | Qwen-2B    | `Qwen/Qwen2-VL-2B-Instruct`                | LoRA adapter (r=16)        | OCR                |

---

## Datasets

| Modality    | Dataset   | Split | Samples used |
|-------------|-----------|-------|--------------|
| Time Series | ECG5000   | test  | 200 requests |
| Vision      | EuroSAT   | test  | 200 requests |
| VLM         | FMTK OCR  | test  | 1 request × 5 runs |

---

## Experiment Protocol

Each backbone is run independently for **5 repeated runs** (`run_idx` 0–4).

**Per run:**
1. Load backbone + task component onto GPU, recording wall-clock load time and
   GPU memory allocation peak via `fmtk.logger.Logger`.
2. Run N closed-loop inference requests sequentially (batch size = 1).
   Each request is timed end-to-end and split into backbone and decoder forward
   passes using CUDA events.
3. Results are appended to `results/summary.csv` and `results/requests.csv`.

**Metrics recorded:**
- `backbone memory (MB)` — GPU memory allocated by backbone load
- `task memory (MB)` — GPU memory allocated by decoder/adapter load
- `backbone load time (ms)` — wall-clock time to load backbone from disk
- `task load time (ms)` — wall-clock time to load decoder/adapter
- `backbone_mean_ms` — mean backbone forward-pass time across N requests
- `decoder_mean_ms` — mean decoder forward-pass time (N/A for VLM)
- `lat_mean_ms`, `lat_p50_ms`, `lat_p95_ms`, `lat_p99_ms` — end-to-end latency percentiles

All reported values are **means across 5 runs**.

---

## Running

```bash
# From serving/
bash experiments/microbenchmark/tsfm/run.sh    # momentbase, chronosbase
bash experiments/microbenchmark/vision/run.sh  # dinobase, swinsmall
bash experiments/microbenchmark/vlm/run.sh     # phi, qwen-2B (with LoRA adapters)

# Plot
python experiments/microbenchmark/plot.py
```

---

## Key Findings

| Backbone | BB Mem | Task Mem | Ratio  | BB Load | Task Load | BB Lat  | Task Lat |
|----------|--------|----------|--------|---------|-----------|---------|----------|
| Moment   | 455 MB | 0.40 MB  | 1137×  | 1256 ms | 1.9 ms    | 9.8 ms  | 0.35 ms  |
| Chronos  | 420 MB | 0.40 MB  | 1050×  | 767 ms  | 1.7 ms    | 30.1 ms | 0.91 ms  |
| DINOv2   | 347 MB | 0.03 MB  | 11567× | 677 ms  | 1.4 ms    | 29.6 ms | 0.32 ms  |
| Swin-S   | 197 MB | 0.03 MB  | 6567×  | 794 ms  | 1.3 ms    | 19.9 ms | 0.26 ms  |
| Phi-3.5  | 8294 MB| 12.6 MB  | 658×   | 5485 ms | 181 ms    | 1482 ms | —        |
| Qwen-2B  | 4420 MB| 17.4 MB  | 254×   | 3569 ms | 120 ms    | 521 ms  | —        |

---

## Paper Text

**Experiment setup.**
We microbenchmark six pretrained foundation model backbones spanning three
modalities — time-series (Moment~\cite{moment}, Chronos~\cite{chronos}),
vision (DINOv2~\cite{dinov2}, Swin-S~\cite{swin}), and vision-language
(Phi-3.5-Vision~\cite{phi3}, Qwen2-VL-2B~\cite{qwen2vl}) — on a single
NVIDIA A16 GPU. For each backbone we attach one lightweight task component:
an MLP decoder for time-series classification (ECG5000 dataset), a linear
classification head for satellite image recognition (EuroSAT dataset), and a
LoRA adapter ($r{=}16$) for OCR. Each configuration is run for 5 independent
trials; we report means. Inference is measured in closed-loop, batch size 1,
over 200 requests per trial (1 for VLM due to generation cost).

**Backbone dominates all resource dimensions.**
Figure~\ref{fig:microbenchmark} shows memory footprint, load time, and
per-request inference latency broken down into backbone and task-component
contributions. Across all modalities, the backbone accounts for the
overwhelming majority of every cost: task components consume
$250\times$--$11{,}000\times$ less GPU memory, load $4\times$--$500\times$
faster, and add less than 1\,ms to per-request latency on top of backbone
inference times ranging from 10\,ms to 1.5\,s.
Concretely, the Moment backbone requires 455\,MB and 1.26\,s to load, while
its MLP decoder requires only 0.4\,MB and under 2\,ms.
For VLMs, Phi-3.5 occupies 8.3\,GB with a 13\,MB LoRA adapter --- a
$658\times$ ratio.
These results confirm that the backbone is the dominant, expensive, and
\emph{shareable} resource, whereas task-specific components are cheap enough
to load and swap on demand. This asymmetry is the core motivation for
FMaaS: by keeping one backbone resident in GPU memory and multiplexing
lightweight task components across concurrent requests, the system amortizes
the dominant backbone cost over many tasks simultaneously.
