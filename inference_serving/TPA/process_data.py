import random
import string
from tpa_simulation import Request
def generate_synthetic_requests(num_requests, available_tasks):
    """
    Simulates 'num_requests' multi-task inference requests.
    Each request will contain 1–3 randomly selected tasks.
    """
    task_requests = []
    for i in range(num_requests):
        request_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        num_tasks = random.randint(1, min(3, len(available_tasks)))  # Multi-task
        tasks = random.sample(available_tasks, num_tasks)
        req = Request(id=request_id, tasks=tasks, input_size=10, tolerance=5)
        task_requests.append(req)
    return task_requests

available_tasks = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Can be real model names too

top_k_intervals = [
    (0, 10_000_000, 144339),
    (1_430_000_000, 1_440_000_000, 3168),
    (2_170_000_000, 2_180_000_000, 2131),
    (2_130_000_000, 2_140_000_000, 2122),
    (1_420_000_000, 1_430_000_000, 2115),
]

# For each interval, generate requests
for start, end, count in top_k_intervals:
    task_requests = generate_synthetic_requests(count, available_tasks)
    print(f"Interval {start}-{end} → {len(task_requests)} requests generated.")
    # Here you'd call your TPA algorithm with task_requests
