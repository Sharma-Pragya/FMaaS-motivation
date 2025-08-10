
import random
from tpa_simulation import RSU, Model, Request, TPA, get_rsu_by_id, get_model_for_task, compute_total_delay

task_pool = ['perception', 'planning', 'localization', 'segmentation', 'detection']

models = [
    Model('M1', {'perception', 'planning'}, 300, {'perception': 200, 'planning': 250}, {'perception': 100, 'planning': 150}, load_time=0.2),
    Model('M2', {'localization', 'detection'}, 250, {'localization': 150, 'detection': 200}, {'localization': 70, 'detection': 120}, load_time=0.1),
    Model('M3', {'segmentation'}, 200, {'segmentation': 180}, {'segmentation': 90}, load_time=0.15)
]

rsus = [
    RSU(1, 2.5, 5.0, True, 1000),
    RSU(2, 2.0, None, False, 800),
    RSU(3, 2.3, 4.5, True, 900),
    RSU(4, 2.1, None, False, 850)
]

def generate_requests(num_requests, task_pool):
    requests = []
    for i in range(num_requests):
        task_count = random.randint(1, 3)
        tasks = set(random.sample(task_pool, task_count))
        input_size = random.uniform(5, 15)
        tolerance = random.uniform(3.5, 6.0)
        requests.append(Request(id=f'R{i}', tasks=tasks, input_size=input_size, tolerance=tolerance))
    return requests

def simulate_intervals(num_intervals=5, base_requests=10):
    results = []
    for t in range(num_intervals):
        print(f"Running interval {t}...")
        num_reqs = base_requests + t * 5
        reqs = generate_requests(num_reqs, task_pool)
        deployment, assignment = TPA(reqs, rsus, models, T_max=10.0, T_min=0.1, T_tol=0.01)

        total_rt = 0.0
        for req in reqs:
            max_delay = 0
            for task in req.tasks:
                rsu_id = assignment.get((req.id, task))
                if rsu_id is None:
                    continue
                rsu = get_rsu_by_id(rsu_id, rsus)
                model = get_model_for_task(task, models)
                delay = compute_total_delay(req, task, rsu, model)
                max_delay = max(max_delay, delay)
            total_rt += max_delay
        results.append(total_rt)
    return results

if __name__ == "__main__":
    random.seed(42)
    results = simulate_intervals(num_intervals=5, base_requests=10)
    print("\nTotal Response Times by Time Interval:")
    for i, val in enumerate(results):
        print(f"Interval {i}: {val:.2f} seconds")
