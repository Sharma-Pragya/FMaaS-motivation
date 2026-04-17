import subprocess
import json
from config import tasks, tpc_counts, tpc_mode, log_file
import sys

if __name__ == "__main__":
    for task_name, task_info in tasks.items():
        print(f"\n=== Running task: {task_name} ===")

        for p in task_info["pipelines"]:
            backbone = p["backbone"]
            dataset  = task_info["datasets"][0]

            for tpc_count in tpc_counts:
                print(f"--- Running model {backbone} on dataset {dataset} tpc={tpc_count} ---")

                subprocess.run([
                    "python3",
                    "worker.py",
                    json.dumps({
                        "task_name": task_name,
                        "task_info": task_info,
                        "pipeline": p,
                        "file_name": log_file,
                        "tpc_count": tpc_count,
                        "tpc_mode": tpc_mode,
                    })
                ])
