import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = "./"
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
stacked_bar(ax, REQ_RATES, data, "Workload (from req_time)", "reqs/sec")
plt.savefig("workload_from_req_time.png")

# -------- Plot 2: Workload from site_manager_send_time --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    data.append(compute_reqs_per_sec(df, "site_manager_send_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Workload (from site_manager_send_time)", "reqs/sec")
plt.savefig("workload_from_site_manager_send_time.png")

# -------- Plot 3: Workload from device_start_time --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    data.append(compute_reqs_per_sec(df, "device_start_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Workload (from device_start_time)", "reqs/sec")
plt.savefig("workload_from_device_start_time.png")

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
ax.set_ylabel("seconds")
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
plt.savefig("response_time_breakdown.png")
# -------- Plot 5: Device throughput --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    df["device_done_time"] = df["device_start_time"] + ((df["proc_time(ms)"] + df["swap_time(ms)"])/1000.0)
    data.append(compute_reqs_per_sec(df, "device_done_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Device Throughput", "reqs/sec")
plt.savefig("device_throughput.png")

# -------- Plot 6: Site manager throughput --------
data = []
for r in REQ_RATES:
    df = load_data(r)
    df["sm_done_time"] = df["site_manager_send_time"] + (df["end_to_end_latency(ms)"]/1000.0)
    data.append(compute_reqs_per_sec(df, "sm_done_time"))

fig, ax = plt.subplots()
stacked_bar(ax, REQ_RATES, data, "Site Manager Throughput", "reqs/sec")
plt.savefig("site_manager_throughput.png")
