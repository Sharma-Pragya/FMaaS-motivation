
from tpa_simulation import RSU, Model, Request, TPA
from process_data import generate_synthetic_requests

# Create RSUs
rsus = [
    RSU(id=2, cpu_freq=2.0, gpu_flops=None, has_gpu=False, storage_capacity=800),
    RSU(id=1, cpu_freq=2.5, gpu_flops=5.0, has_gpu=True, storage_capacity=1000),
    
]

# Create Models
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
        tasks={'localization'},
        storage=200,
        cpu_costs={'localization': 100},
        gpu_costs={'localization': 50},
        load_time=0.1
    )
]

# Create Requests
requests = [
    Request(id='R1', tasks={'perception', 'localization'}, input_size=10, tolerance=5),
    Request(id='R2', tasks={'planning'}, input_size=8, tolerance=5),
    Request(id='R3', tasks={'localization'}, input_size=12, tolerance=5)
]

# Run TPA
deployment, assignments = TPA(requests, rsus, models, T_max=10.0, T_min=0.1, T_tol=0.01)

# Output results
print("Model Deployment per RSU:")
for rsu_id, model_ids in deployment.items():
    print(f"  RSU {rsu_id}: Models {list(model_ids)}")

print("\nTask Assignments:")
for (req_id, task), rsu_id in assignments.items():
    print(f"  Task '{task}' from Request '{req_id}' -> RSU {rsu_id}")
