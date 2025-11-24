# Scaling Experiments Bug Analysis

## Summary
The scaling experiments have several critical bugs that invalidate the results and plots:

1. **Throughput Overcounting Bug** - Massively inflates throughput values
2. **Duplicate Task Handling** - Results plateau at 9 tasks for VLM
3. **Expected vs Observed Trends** - Metrics don't follow expected patterns

---

## Bug #1: Throughput Overcounting (CRITICAL)

### Location
`run_scaling_experiments.py:461-465` in `collect_metrics()`

### The Bug
```python
for (m, d) in deployed:
    model_tasks = [t for t in tasks if support.get((m, t), 0) == 1]
    if model_tasks:
        capacities = [Ptmd.get((t, m, d), 0.0) for t in model_tasks]
        total_capacity += sum(capacities)  # ← WRONG!
```

If a model can serve 6 tasks, this sums the throughput for ALL 6 tasks.

### Example
- Model M1 deployed on device D1
- M1 supports 6 VLM tasks
- Each task has throughput ~8 req/s on M1+D1
- **Reported**: 6 × 8 = 48 req/s capacity
- **Actual**: ~8 req/s capacity (shared across tasks)

This is why throughput INCREASES with task count - more tasks = more capacities to sum!

### Impact
- Throughput values are inflated by 2-9x
- Makes it appear that adding tasks INCREASES capacity
- Completely invalidates throughput plots

### Expected Behavior
As tasks increase:
- **Throughput should**: Stay flat or slightly increase (if more deployments needed)
- **Actual**: Increases linearly with task count due to overcounting

### Fix Options
1. Use actual routing to calculate utilized capacity
2. Use average/min throughput across supported tasks
3. Report per-deployment capacity (not summed across tasks)

---

## Bug #2: Duplicate Task Handling

### Location
`run_scaling_experiments.py:83-93` (task selection) + ILP formulation

### The Problem
VLM has only **9 unique tasks**:
```python
VLM_TASKS = [
    "activity_recognition", "crowd_counting", "gesture_recognition",
    "image_classification", "object_detection", "ocr",
    "scene_classification", "traffic_classification", "vqa"
]
```

When requesting 10+ tasks, round-robin creates duplicates:
- 10 tasks: `["activity_recognition", ..., "vqa", "activity_recognition"]`
- But ILP only recognizes 9 unique task names

### Evidence
From `vlm_10tasks_deployments.json`:
- `num_tasks`: 10
- `task_list`: 10 items (with "activity_recognition" twice)
- `total_demand_req_s`: **9.0** (not 10.0!)

### Impact
- Results for 10, 12, 14, 16 tasks are **identical**
- All show demand = 9.0 req/s
- Plots show flatline after 9 tasks
- Scaling beyond 9 tasks is not actually tested

### Why This Happens
In `build_ilp_inputs_vlm()` and the ILP:
- Demands dict is keyed by task NAME: `demands = {t: 1.0 for t in task_list}`
- Python dict with duplicate keys only stores the last value
- ILP constraint (line 318): `lam = float(demands.get(t, 0.0))`
- Duplicate "activity_recognition" doesn't get 2.0 req/s demand

### Fix Options
1. Add more unique VLM tasks (10-16 total)
2. Modify task selection to avoid duplicates
3. Support duplicate tasks with aggregated demand
4. Limit VLM scaling to 9 tasks max

---

## Bug #3: Expected Trends Not Observed

### What SHOULD Happen as Tasks Increase

| Metric | Expected Trend | Actual Trend | Why It's Wrong |
|--------|---------------|--------------|----------------|
| **Memory** | Increase (more/larger models) | Increases then jumps/plateaus | Plateau due to duplicate tasks |
| **Latency** | Increase or stay flat | Decreases then plateaus | Bug in calculation or model selection |
| **Throughput** | Flat or slight increase | **Increases linearly** | Overcounting bug #1 |
| **Demand** | Increase linearly | Plateaus at 9.0 | Duplicate task bug #2 |

### VLM Data (O1: Deployments Only)
```
Tasks   Demand   Memory(MB)  Throughput(req/s)  Latency(ms)
2       2.0      16889       4.76               421.7
4       4.0      7154        28.57              141.9
6       6.0      8668        47.48              132.6
8       8.0      20585       130.40             128.6
10      9.0      22193       143.85             130.8    ← demand stuck
12      9.0      22193       143.85             130.8    ← identical
14      9.0      22193       143.85             130.8    ← identical
16      9.0      22193       143.85             130.8    ← identical
```

### Anomalies
1. **Latency DECREASES** from 422ms (2 tasks) to 131ms (10 tasks)
   - Should increase or stay flat as system gets loaded
   - Likely: ILP selects faster models when more tasks present

2. **Memory drops** from 16.9GB (2 tasks) to 7.2GB (4 tasks)
   - Should increase monotonically
   - Likely: Different models selected based on task mix

3. **Throughput increases 30x** from 4.8 to 144 req/s
   - With only 4x more deployments (1→2)
   - Explained by overcounting bug

---

## Bug #4: Latency Calculation Issues

### Location
`run_scaling_experiments.py:474-483`

### The Code
```python
weighted_latency = 0.0
weighted_demand = 0.0
for (t, m, d), frac in r.items():
    if frac > 0.001:
        lat = latency_ilp.get((t, m, d), 0.0)
        dem = demands.get(t, 0.0)
        weighted_latency += lat * frac * dem
        weighted_demand += dem * frac
avg_latency = weighted_latency / weighted_demand if weighted_demand > 0 else 0.0
```

### Issue
This weights by routing fraction AND demand, which may double-weight demand.

### Why Latency Decreases
When # tasks increases, ILP may:
- Select different (faster) models
- Route to faster model-device pairs
- Have more flexibility in optimization

This isn't necessarily a bug, but counterintuitive - need to verify.

---

## Bug #5: Memory Footprint Jumps

### Observed
For O1+O2+O3+O4 (all objectives):
```
6 tasks:  8.7 GB
8 tasks:  57.8 GB  (6.7x jump!)
10+ tasks: 58.8 GB (stays flat)
```

### Possible Causes
1. Waste+ModelSize objectives prefer larger models
2. At 8 tasks, system hits capacity threshold
3. Must deploy 2nd model, and waste optimization picks huge models

### Need to Investigate
- What models are selected at 6 vs 8 tasks?
- Why does waste optimization choose 58GB models?

---

## Recommendations

### Immediate Fixes
1. **Fix throughput calculation** (Bug #1)
   - Option A: Use routing-based actual throughput
   - Option B: Use min or average throughput across supported tasks

2. **Fix duplicate task handling** (Bug #2)
   - Option A: Add unique VLM tasks (10-16 total)
   - Option B: Aggregate demand for duplicate task names
   - Option C: Limit scaling experiments to 9 tasks for VLM

3. **Re-run all experiments** after fixes

### Deeper Investigation
1. Analyze why latency decreases with more tasks
2. Understand memory jump at 8 tasks for waste+modelsize objectives
3. Verify all three workloads (VLM, TSFM, Mixed) have same issues
4. Add validation: check that trends make sense

### Plot Improvements
1. Add total_demand line to throughput plot
2. Show actual vs deployed capacity
3. Annotate where duplicate tasks begin
4. Add error bars or confidence intervals if experiments are stochastic

---

## Next Steps
1. Create fix for throughput calculation
2. Create fix for duplicate task handling
3. Re-run VLM scaling experiments
4. Validate results make sense
5. Check TSFM and Mixed workloads
6. Regenerate plots
