from tpa_simulation import TPA, RSU, Model
from process_data import generate_synthetic_requests

# === Define RSUs ===
rsus = [
    RSU(id='RSU1', cpu_freq=2.5, gpu_flops=5.0, has_gpu=True, storage_capacity=1000),
    RSU(id='RSU2', cpu_freq=2.0, gpu_flops=0.0, has_gpu=False, storage_capacity=800),
    RSU(id='RSU3', cpu_freq=3.0, gpu_flops=6.0, has_gpu=True, storage_capacity=1200),
]

# === Define Models ===
models = [
    Model(
        id='M1',
        tasks={'perception', 'planning'},
        storage=300,
        cpu_costs={'perception': 200, 'planning': 250},
        gpu_costs={'perception': 100, 'planning': 150},
        load_time=0.2
    ),
    Model(
        id='M2',
        tasks={'prediction', 'localization'},
        storage=250,
        cpu_costs={'prediction': 180, 'localization': 200},
        gpu_costs={'prediction': 90, 'localization': 100},
        load_time=0.15
    ),
    Model(
        id='M3',
        tasks={'obstacle_avoidance'},
        storage=150,
        cpu_costs={'obstacle_avoidance': 120},
        gpu_costs={'obstacle_avoidance': 60},
        load_time=0.1
    )
]

# === Available tasks for requests ===
available_tasks = ['perception', 'planning', 'prediction', 'localization', 'obstacle_avoidance']

# === Top intervals from Google trace (format: (start, end, count)) ===
top_k_intervals = [
    (0, 10_000_000, 144339),
    (1_430_000_000, 1_440_000_000, 3168),
    (2_170_000_000, 2_180_000_000, 2131),
    (2_130_000_000, 2_140_000_000, 2122),
    (1_420_000_000, 1_430_000_000, 2115),
]

# === Run experiment for each interval ===
for idx, (start, end, count) in enumerate(top_k_intervals):
    print(f"\n--- Interval {idx + 1} | Time {start // 1_000_000}s–{end // 1_000_000}s | {count} requests ---")

    task_requests = generate_synthetic_requests(count, available_tasks)

    deployment, assignment = TPA(task_requests, rsus, models, T_max=10.0, T_min=0.1, T_tol=0.01)

    # === Log summary ===
    print(f"Deployed Models per RSU:")
    for rsu_id, deployed_models in deployment.items():
        print(f"  RSU {rsu_id}: {list(deployed_models)}")

    total_assignments = len(assignment)
    print(f"Total task assignments made: {total_assignments}")
    print(f"Assignment coverage: {total_assignments / (count * 2):.2%} (assuming avg 2 tasks/request)")
