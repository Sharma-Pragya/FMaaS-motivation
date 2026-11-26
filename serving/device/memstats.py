"""memstats.py — GPU memory numbers for Control("stats").

Shared mode reads them in the server process when asked. In process mode
(isolated_app.py) each process writes its own numbers into a small shared
array every PERIOD_S, and the gRPC process reads the arrays when asked: a
control message to a task process would wait behind its queued requests.
"""
from __future__ import annotations

import threading
import time

PERIOD_S = 0.25
MB = 2 ** 20

# Backbone process array: time, allocated, reserved, device used, device total.
BB_FIELDS = ("t", "allocated_mb", "reserved_mb", "device_used_mb", "device_total_mb")
# Task process array: time, allocated, reserved, slots in its pool, slots in
# use (holding a request), MB per slot (input + mask + features), requests
# waiting on the CPU.
TASK_FIELDS = ("t", "allocated_mb", "reserved_mb", "slots_total", "slots_used",
               "slot_mb", "overflow_n")


def cuda_mem(dev) -> dict:
    """This process's torch allocated / reserved MB, and the device's used /
    total MB as CUDA reports them (cudaMemGetInfo: the whole GPU, or the MIG
    slice, every process and CUDA context included). Zeros on CPU."""
    import torch
    if not (str(dev).startswith("cuda") and torch.cuda.is_available()):
        return {"allocated_mb": 0.0, "reserved_mb": 0.0,
                "device_used_mb": 0.0, "device_total_mb": 0.0}
    free, total = torch.cuda.mem_get_info(dev)
    return {"allocated_mb": torch.cuda.memory_allocated(dev) / MB,
            "reserved_mb": torch.cuda.memory_reserved(dev) / MB,
            "device_used_mb": (total - free) / MB, "device_total_mb": total / MB}


def new_array(ctx, fields: tuple):
    """A shared array for one process's numbers (no lock: one writer)."""
    return ctx.Array("d", len(fields), lock=False)


def start_writer(arr, fields: tuple, read, stop=None) -> threading.Thread:
    """Write read() (a dict over fields, without "t") into arr every PERIOD_S."""
    def loop() -> None:
        while stop is None or not stop.is_set():
            try:
                vals = read()
                arr[0] = time.time()
                for k, f in enumerate(fields[1:], start=1):
                    arr[k] = float(vals.get(f, 0.0) or 0.0)
            except Exception:
                pass
            time.sleep(PERIOD_S)
    th = threading.Thread(target=loop, name="memstats", daemon=True)
    th.start()
    return th


def task_mem_mb(spec: str | None, task: str | None) -> float | None:
    """A vFM's memory budget in MB from --task-mem-mb, or None (no budget).

    spec is "MB" (every task) or "task=MB,...", where "*=MB" sets the
    budget of tasks not named, e.g. "victim=2048,*=1024"."""
    if not spec:
        return None
    default, per = None, {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, mb = part.split("=", 1)
            if name.strip() == "*":
                default = float(mb)
            else:
                per[name.strip()] = float(mb)
        else:
            default = float(part)
    return per.get(task, default)


def read_array(arr, fields: tuple) -> dict:
    out = {f: arr[k] for k, f in enumerate(fields)}
    out["age_s"] = time.time() - out.pop("t") if out.get("t") else None
    return out
