# Colocation experiment

Colocates `ecgclass` (tsfm / `momentlarge`) and `nyudepth` (vision / `dinobase-patch`)
on the same GPU with `max_batch_size=1`.

Conditions per RPS:
- `single_ecgclass` — one device server running momentlarge + ecgclass
- `single_nyudepth` — one device server running dinobase + nyudepth
- `no_sharing`      — two device servers (one per backbone) running concurrently

## Usage

```bash
RPS_SWEEP=1,5,10 bash experiments/colocation/run.sh
```

Override knobs via env (see top of [run.sh](run.sh)):

```bash
MAX_BATCH_SIZE=1 \
RPS_SWEEP=1,2,5,10,20 \
PHASE_DURATION=300 \
TSFM_BACKBONE=momentlarge \
VISION_BACKBONE=dinobase-patch \
bash experiments/colocation/run.sh
```

## Output layout

```
results/
  rps_<R>/
    trace.json
    single_ecgclass/   { latencies.csv, task_results.csv, run_config.json }
    single_nyudepth/   { ... }
    no_sharing/        { ... }
  logs/
```
