import math
from typing import List, Dict


class Request:
    def __init__(self, req_id, task, site_manager, device, req_time):
        self.req_id = req_id
        self.task = task
        self.site_manager = site_manager
        self.device = device
        self.req_time = req_time
        self.backbone = None

    def __repr__(self):
        return (f"req_id={self.req_id}, task={self.task}, "
                f"site_manager={self.site_manager}, device={self.device}, "
                f"req_time={self.req_time}")

    def to_dict(self):
        return {
            "req_id": self.req_id,
            "task": self.task,
            "site_manager": self.site_manager,
            "device": self.device,
            "backbone": self.backbone,
            "req_time": self.req_time,
        }


def generate_requests(
    req_rate: float,
    duration: float,
    tasks: list,
    seed: int = 42,
    req_id_offset: int = 0,
) -> tuple:
    """Generate a perfectly deterministic (D/D/1) workload trace.

    Arrivals are evenly spaced at interval = 1/req_rate seconds.
    Tasks are assigned round-robin across all task slots, giving each
    task an equal share of the total request rate.

    This is the baseline for the D/D/1 assumption implicit in the
    heuristic util = Σ rps × latency / 1000.  Use it to isolate
    whether latency blowup is caused by workload burstiness (CV > 0)
    or by true over-utilisation (rho >= 1).

    Args:
        req_rate:      Total requests per second across all tasks.
        duration:      Experiment duration in seconds.
        tasks:         List of (task_name, site_manager, device, backbone)
                       tuples, one entry per task.
        seed:          Ignored (deterministic trace needs no RNG).
        req_id_offset: Starting request ID (for multi-phase experiments).

    Returns:
        (requests, mean_rps_per_task, peak_rps_per_task)
    """
    k = len(tasks)
    tot_req = int(req_rate * duration)
    interval = 1.0 / req_rate          # fixed inter-arrival gap (seconds)

    requests: List[Request] = []
    counts: Dict[str, int] = {}
    bins: Dict[str, Dict[int, int]] = {}

    for i in range(tot_req):
        req_time = (i + 1) * interval  # first arrival at t=interval, not t=0
        task_name, site_manager, device, backbone = tasks[i % k]

        requests.append(
            Request(req_id_offset + i, task_name, site_manager, device, req_time)
        )

        counts[task_name] = counts.get(task_name, 0) + 1
        sec = int(math.floor(req_time))
        bins.setdefault(task_name, {})[sec] = bins.get(task_name, {}).get(sec, 0) + 1

    mean_rps = {t: cnt / float(duration) for t, cnt in counts.items()}
    peak_rps = (
        {t: float(max(sec_bins.values())) for t, sec_bins in bins.items()}
        if bins else {}
    )

    return requests, mean_rps, peak_rps
