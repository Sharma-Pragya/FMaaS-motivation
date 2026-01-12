import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = "./greedy_new"
SAVEFIG_DIR=f"./plots/{ROOT_DIR}"
REQ_RATES = sorted([int(d) for d in os.listdir(ROOT_DIR) if d.isdigit()])
def load_data(req_rate):
    path = os.path.join(ROOT_DIR, str(req_rate), "request_latency_results.csv")
    return pd.read_csv(path)

def compute_reqs_per_sec(df, time_col):
    out = {}
    for task, g in df.groupby("task"):
        t = g[time_col]
        duration = t.max() - t.min()
        out[task] = len(t) / duration if duration > 0 else 0.0
    return out

# def compute_reqs_per_sec(df, time_col):
#     out = {}

#     for task, g in df.groupby("task"):
#         t = g[time_col].values
#         t = t - t.min()

#         sec_bins = np.floor(t).astype(int)
#         counts = pd.Series(sec_bins).value_counts()

#         out[task] = counts.mean() if len(counts) > 0 else 0.0

#     return out

# def compute_reqs_per_sec(df, time_col):
#     out = {}
#     for task, g in df.groupby("task"):
#         t = g[time_col].to_numpy()
#         if len(t) == 0:
#             out[task] = 0.0
#             continue

#         t = t - t.min()
#         sec = np.floor(t).astype(np.int64)

#         last = int(sec.max())
#         counts = np.bincount(sec, minlength=last + 1)

#         out[task] = counts.mean()

#     return out

def stacked_bar(ax, req_rates, data_dicts, title, ylabel):
    x = np.arange(len(req_rates))
    tasks = sorted({k for d in data_dicts for k in d})
    bottom = np.zeros(len(req_rates))

    for task in tasks:
        vals = [d.get(task, 0.0) for d in data_dicts]
        ax.bar(x, vals, bottom=bottom, label=task)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(req_rates)
    ax.set_xlabel("req_rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

# -------- Plot 1: Workload from req_time --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    data.append(compute_reqs_per_sec(df, "req_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Workload from dataset", "reqs/sec")
plt.savefig(f"{SAVEFIG_DIR}/workload_from_req_time.png")

# -------- Plot 2: Workload from site_manager_send_time --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    data.append(compute_reqs_per_sec(df, "site_manager_send_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Workload from Sender", "reqs/sec")
plt.savefig(f"{SAVEFIG_DIR}/workload_from_site_manager_send_time.png")

# -------- Plot 3: Workload from device_start_time --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    data.append(compute_reqs_per_sec(df, "device_start_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Workload on device", "reqs/sec")
plt.savefig(f"{SAVEFIG_DIR}/workload_from_device_start_time.png")

# -------- Plot 4: Response time breakdown --------
tasks = sorted(load_data(REQ_RATES[0])["task"].unique())
x = np.arange(len(REQ_RATES))
width = 0.8 / len(tasks)

fig, ax = plt.subplots()

task_colors = {task: plt.rcParams["axes.prop_cycle"].by_key()["color"][i % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])] for i, task in enumerate(tasks)}
hatch_proc = ""
hatch_swap = "///"
hatch_other = "xx"

for i, task in enumerate(tasks):
    proc = []
    swap = []
    other = []

    for r in REQ_RATES:
        df = load_data(r)
        g = df[df["task"] == task]

        p = g["proc_time(ms)"].mean()
        s = g["swap_time(ms)"].mean()
        e = g["end_to_end_latency(ms)"].mean()

        proc.append(p)
        swap.append(s)
        other.append(max(e - p - s, 0.0))

    xpos = x + i * width
    c = task_colors[task]

    ax.bar(xpos, proc, width, color=c, hatch=hatch_proc, edgecolor="black")
    ax.bar(xpos, swap, width, bottom=proc, color=c, hatch=hatch_swap, edgecolor="black")
    ax.bar(xpos, other, width, bottom=np.array(proc) + np.array(swap), color=c, hatch=hatch_other, edgecolor="black")

ax.set_xticks(x + width * (len(tasks) - 1) / 2)
ax.set_xticklabels(REQ_RATES)
ax.set_title("Response Time Breakdown")
ax.set_ylabel("Time (ms)")
ax.set_xlabel("req_rate")

task_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=task_colors[t], edgecolor="black", label=t) for t in tasks]
comp_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatch_proc, label="proc_time"),
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatch_swap, label="swap_time"),
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatch_other, label="other"),
]

leg1 = ax.legend(handles=task_handles, title="Task", loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.add_artist(leg1)
ax.legend(handles=comp_handles, title="Component", loc="upper left", bbox_to_anchor=(1.02, 0.55))

plt.tight_layout()
plt.savefig(f"{SAVEFIG_DIR}/response_time_breakdown.png")
## box plot version
tasks = sorted(load_data(REQ_RATES[0])["task"].unique())
x = np.arange(len(REQ_RATES))
width = 0.8 / len(tasks)

fig, ax = plt.subplots()

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
task_colors = {task: colors[i % len(colors)] for i, task in enumerate(tasks)}

for i, task in enumerate(tasks):
    data = []
    positions = []

    for r_idx, r in enumerate(REQ_RATES):
        df = load_data(r)
        g = df[df["task"] == task]
        vals = g["end_to_end_latency(ms)"].to_numpy()

        pos = x[r_idx] + i * width
        data.append(vals)
        positions.append(pos)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.85,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black"),
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(task_colors[task])
        patch.set_edgecolor("black")
        patch.set_alpha(0.6)

    # for pos, vals in zip(positions, data):
    #     if len(vals) == 0:
    #         continue
    #     n = min(len(vals), 300)
    #     idx = np.random.choice(len(vals), size=n, replace=False)
    #     v = vals[idx]
        # jitter = (np.random.rand(n) - 0.5) * width * 0.55
        # ax.scatter(np.full(n, pos) + jitter, v, s=8, alpha=0.25)

ax.set_xticks(x + width * (len(tasks) - 1) / 2)
ax.set_xticklabels(REQ_RATES)
ax.set_xlabel("req_rate")
ax.set_ylabel("end_to_end_latency (ms)")
ax.set_title("Response Time")

task_handles = [
    plt.Line2D([0], [0], color=task_colors[t], lw=6, label=t)
    for t in tasks
]
ax.legend(handles=task_handles, title="Task", bbox_to_anchor=(1.02, 1.0), loc="upper left")

plt.tight_layout()
plt.savefig(f"{SAVEFIG_DIR}/response_time_box_jitter.png")

for r in REQ_RATES:
    df = load_data(r)
    vals = df["end_to_end_latency(ms)"]
    print(f"{r} req/s")
    print("p50:", np.percentile(vals, 50))
    print("p90:", np.percentile(vals, 90))
    print("p95:", np.percentile(vals, 95))
    print("p99:", np.percentile(vals, 99))
    print("mean:", vals.mean())
    print()

# -------- Plot 5: Device throughput --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    df["device_done_time"] = df["device_start_time"] + ((df["proc_time(ms)"] + df["swap_time(ms)"])/1000.0)
    data.append(compute_reqs_per_sec(df, "device_done_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Device Throughput", "reqs/sec")
plt.savefig(f"{SAVEFIG_DIR}/device_throughput.png")

# -------- Plot 6: Site manager throughput --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    df["sm_done_time"] = df["site_manager_send_time"] + (df["end_to_end_latency(ms)"]/1000.0)
    data.append(compute_reqs_per_sec(df, "sm_done_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Sender Throughput", "reqs/sec")
for r, d in zip(REQ_RATES, data):
    print(f"\n{r} req/s")
    total = 0.0
    for task, v in d.items():
        print(f"  {task}: {v:.2f} req/s")
        total += v
    print(f"  TOTAL: {total:.2f} req/s")
plt.savefig(f"{SAVEFIG_DIR}/site_manager_throughput.png")

# # -------- Plot 7: gpu time --------
# tasks = sorted(load_data(REQ_RATES[0])["task"].unique())
# x = np.arange(len(REQ_RATES))
# width = 0.8 / len(tasks)

# fig, ax = plt.subplots()

# task_colors = {task: plt.rcParams["axes.prop_cycle"].by_key()["color"][i % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])] for i, task in enumerate(tasks)}

# for i, task in enumerate(tasks):
#     gpu_time = []

#     for r in REQ_RATES:
#         df = load_data(r)
#         g = df[df["task"] == task]
#         p = g["gpu_time(ms)"].mean()
#         gpu_time.append(p)

#     xpos = x + i * width
#     c = task_colors[task]

#     ax.bar(xpos, p, width, color=c, hatch=hatch_proc, edgecolor="black")

# ax.set_xticks(x + width * (len(tasks) - 1) / 2)
# ax.set_xticklabels(REQ_RATES)
# ax.set_title("GPU Time Breakdown")
# ax.set_ylabel("Time (ms)")
# ax.set_xlabel("req_rate")

# task_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=task_colors[t], edgecolor="black", label=t) for t in tasks]
# leg1 = ax.legend(handles=task_handles, title="Task", loc="upper left", bbox_to_anchor=(1.02, 1.0))
# ax.add_artist(leg1)

# plt.tight_layout()
# plt.savefig(f"{SAVEFIG_DIR}/gpu_time.png")

# # -------- Plot 7: gpu sync --------
# tasks = sorted(load_data(REQ_RATES[0])["task"].unique())
# x = np.arange(len(REQ_RATES))
# width = 0.8 / len(tasks)

# fig, ax = plt.subplots()

# task_colors = {task: plt.rcParams["axes.prop_cycle"].by_key()["color"][i % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])] for i, task in enumerate(tasks)}

# for i, task in enumerate(tasks):
#     gpu_time = []

#     for r in REQ_RATES:
#         df = load_data(r)
#         g = df[df["task"] == task]
#         p = g["gpu_sync(ms)"].mean()
#         gpu_time.append(p)

#     xpos = x + i * width
#     c = task_colors[task]

#     ax.bar(xpos, p, width, color=c, hatch=hatch_proc, edgecolor="black")

# ax.set_xticks(x + width * (len(tasks) - 1) / 2)
# ax.set_xticklabels(REQ_RATES)
# ax.set_title("GPU Sync Breakdown")
# ax.set_ylabel("Time (ms)")
# ax.set_xlabel("req_rate")

# task_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=task_colors[t], edgecolor="black", label=t) for t in tasks]
# leg1 = ax.legend(handles=task_handles, title="Task", loc="upper left", bbox_to_anchor=(1.02, 1.0))
# ax.add_artist(leg1)

# plt.tight_layout()
# plt.savefig(f"{SAVEFIG_DIR}/gpu_sync.png")