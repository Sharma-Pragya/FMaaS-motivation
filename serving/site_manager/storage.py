DEPLOYMENTS = []
RUNTIME_REQUESTS = []
PROCESSED_COUNT = 0  # Track how many requests have been processed
OUTPUT_DIR = None  # Set by orchestrator to save results in the right experiment directory
DEPLOYING_TASKS = set()  # Track (task, device) pairs whose deployment is currently in progress


def clear_state():
    """Reset all stored state between experiments.

    Must be called at the start of each new deployment to prevent
    request/state accumulation across experiments.
    """
    global DEPLOYMENTS, RUNTIME_REQUESTS, PROCESSED_COUNT, OUTPUT_DIR, DEPLOYING_TASKS
    DEPLOYMENTS = []
    RUNTIME_REQUESTS = []
    PROCESSED_COUNT = 0
    OUTPUT_DIR = None
    DEPLOYING_TASKS.clear()
    print("[RuntimeBuffer] State cleared for new experiment.")


def store_plan(payload):
    global DEPLOYMENTS, OUTPUT_DIR
    DEPLOYMENTS = payload.get("deployments", [])
    OUTPUT_DIR = payload.get("output_dir", None)
    print(f"[RuntimeBuffer] Stored {len(DEPLOYMENTS)} deployments"
          f"{f', output_dir={OUTPUT_DIR}' if OUTPUT_DIR else ''}")


def store_requests(payload):
    global RUNTIME_REQUESTS
    RUNTIME_REQUESTS.extend(payload.get("runtime_requests", []))
    print(f"[RuntimeBuffer] Stored {len(RUNTIME_REQUESTS)} runtime requests")


def get_requests():
    """Get all requests (including newly added ones during runtime)."""
    return RUNTIME_REQUESTS


def get_new_requests():
    """Get only requests that haven't been scheduled yet.

    Returns requests from PROCESSED_COUNT onwards, and updates the counter.
    This allows the inference loop to pick up newly added requests during runtime.
    """
    global PROCESSED_COUNT
    new_reqs = RUNTIME_REQUESTS[PROCESSED_COUNT:]
    PROCESSED_COUNT = len(RUNTIME_REQUESTS)
    if new_reqs:
        print(f"[RuntimeBuffer] Retrieved {len(new_reqs)} new request(s) (total={PROCESSED_COUNT})")
    return new_reqs


def get_deployments():
    return DEPLOYMENTS


def get_output_dir():
    return OUTPUT_DIR


def replace_deployment(old_backbone: str, new_spec: dict):
    """Replace an existing deployment (identified by backbone) with a new spec.

    Used during backbone migration — removes the old deployment entry and
    inserts the new one so cleanup later targets the right servers.

    Args:
        old_backbone: Backbone name of the deployment to remove.
        new_spec: New deployment spec dict to insert.
    """
    global DEPLOYMENTS
    DEPLOYMENTS = [d for d in DEPLOYMENTS if d.get("backbone") != old_backbone]
    DEPLOYMENTS.append(new_spec)
    print(f"[RuntimeBuffer] Replaced deployment backbone '{old_backbone}' → '{new_spec.get('backbone')}'. "
          f"Total: {len(DEPLOYMENTS)}")


def append_deployments(new_specs: list):
    """Append new deployment specs to the existing DEPLOYMENTS list.

    Used when new tasks are added at runtime — the site manager needs
    to track all active deployments for cleanup.

    Args:
        new_specs: List of deployment spec dicts to append.
    """
    global DEPLOYMENTS
    DEPLOYMENTS.extend(new_specs)
    print(f"[RuntimeBuffer] Appended {len(new_specs)} deployment(s). "
          f"Total: {len(DEPLOYMENTS)}")


def mark_task_deploying(task_name: str, device: str):
    """Mark a (task, device) pair as currently being deployed.

    Used to defer requests for this task on this specific device
    until deployment completes. Other devices serving the same task
    are unaffected.
    """
    DEPLOYING_TASKS.add((task_name, device))
    print(f"[RuntimeBuffer] Task '{task_name}' on '{device}' marked as DEPLOYING (total deploying: {len(DEPLOYING_TASKS)})")


def mark_task_deployed(task_name: str, device: str):
    """Mark a (task, device) pair as deployment complete.

    Requests for this task on this device can now be executed.
    """
    DEPLOYING_TASKS.discard((task_name, device))
    print(f"[RuntimeBuffer] Task '{task_name}' on '{device}' marked as DEPLOYED (remaining deploying: {len(DEPLOYING_TASKS)})")


def is_task_deploying(task_name: str, device: str) -> bool:
    """Check if a (task, device) pair is currently being deployed.

    Returns True only if this specific task+device combination is
    deploying — other devices serving the same task are unaffected.
    """
    return (task_name, device) in DEPLOYING_TASKS