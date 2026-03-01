This experiment is the "phase 1 runs both `ecgclass` and `gestureclass`" version.

How it works:

- Each phase starts from a fresh deployment.
- The deployment for that phase includes the full cumulative task set.
- The request trace for that phase is one global deterministic arrival stream
  for all active tasks.
- Each arrival is assigned to a task according to the target phase rates.
- After the phase completes, the devices are cleaned up.
- The next phase then redeploys with the next cumulative workload.

Example:

- Phase 0:
  - `ecgclass @ 10`
- Phase 1:
  - `ecgclass @ 10`
  - `gestureclass @ 8`
- Phase 2:
  - `ecgclass @ 15`
  - `gestureclass @ 8`

So phase 1 does not run only `gestureclass`.
It runs both tasks together, because the phase trace is generated from the full
current workload.

Why this is different from `runtime_stepwise`:

- `runtime_stepwise` runs only the newly generated batch in each phase.
- `runtime_collective` regenerates one combined batch for all active tasks in
  each phase.

Why this helps:

- There is no overlap of independent runtime batches from earlier events.
- You get one phase-local trace that reflects the current total workload mix.
- You avoid independent per-task periodic clocks re-aligning with each other.
- It is easier to evaluate the scheduler under a cleaner workload model.

Important limitation:

- This is still a phased experiment, not one continuous timeline.
- Earlier traffic does not remain live across phase boundaries.
- Instead, each phase is a fresh run of the cumulative state.

Run:

```bash
cd serving
bash experiments/runtime_collective/run.sh
```

Results:

- Each phase writes its own deployment plan, deployment metadata, and latency CSV
  under `experiments/runtime_collective/results/.../phase*/`.
