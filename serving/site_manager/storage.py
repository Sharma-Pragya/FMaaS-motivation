import threading

DEPLOYMENTS = []
RUNTIME_REQUESTS = []
PROCESSED_COUNT = 0  # Track how many requests have been handed to the executor
OUTPUT_DIR = None  # Set by orchestrator to save results in the right experiment directory
_STATE_LOCK = threading.Lock()
_NEW_REQUESTS_EVENT = threading.Event()


def clear_state():
    """Reset all stored state between experiments."""
    global DEPLOYMENTS, RUNTIME_REQUESTS, PROCESSED_COUNT, OUTPUT_DIR
    with _STATE_LOCK:
        DEPLOYMENTS = []
        RUNTIME_REQUESTS = []
        PROCESSED_COUNT = 0
        OUTPUT_DIR = None
    _NEW_REQUESTS_EVENT.clear()
    print("[RuntimeBuffer] State cleared for new experiment.")


def store_plan(payload):
    global DEPLOYMENTS, OUTPUT_DIR
    with _STATE_LOCK:
        DEPLOYMENTS = payload.get("deployments", [])
        OUTPUT_DIR = payload.get("output_dir", None)
        dep_count = len(DEPLOYMENTS)
        out_dir = OUTPUT_DIR
    print(f"[RuntimeBuffer] Stored {dep_count} deployments"
          f"{f', output_dir={out_dir}' if out_dir else ''}")


def store_requests(payload):
    """Append incoming runnable request chunks."""
    global RUNTIME_REQUESTS
    new_reqs = payload.get("runtime_requests", [])
    with _STATE_LOCK:
        RUNTIME_REQUESTS.extend(new_reqs)
        total = len(RUNTIME_REQUESTS)
    if new_reqs:
        _NEW_REQUESTS_EVENT.set()
    print(f"[RuntimeBuffer] Stored {len(new_reqs)} request(s) (total={total})")


def get_requests():
    """Get all requests (including newly added ones during runtime)."""
    with _STATE_LOCK:
        return list(RUNTIME_REQUESTS)


def get_new_requests():
    """Get only requests that haven't been scheduled yet."""
    global PROCESSED_COUNT
    with _STATE_LOCK:
        new_reqs = list(RUNTIME_REQUESTS[PROCESSED_COUNT:])
        PROCESSED_COUNT = len(RUNTIME_REQUESTS)
        total = PROCESSED_COUNT
        if PROCESSED_COUNT >= len(RUNTIME_REQUESTS):
            _NEW_REQUESTS_EVENT.clear()
    if new_reqs:
        print(f"[RuntimeBuffer] Retrieved {len(new_reqs)} new request(s) (total={total})")
    return new_reqs


def wait_for_new_requests(timeout: float) -> bool:
    """Block until request storage is updated or timeout expires."""
    triggered = _NEW_REQUESTS_EVENT.wait(timeout)
    if triggered:
        _NEW_REQUESTS_EVENT.clear()
    return triggered


def get_deployments():
    with _STATE_LOCK:
        return list(DEPLOYMENTS)


def get_output_dir():
    with _STATE_LOCK:
        return OUTPUT_DIR


def replace_deployment(old_backbone: str, new_spec: dict):
    """Replace an existing deployment (identified by backbone) with a new spec."""
    global DEPLOYMENTS
    with _STATE_LOCK:
        DEPLOYMENTS = [d for d in DEPLOYMENTS if d.get("backbone") != old_backbone]
        DEPLOYMENTS.append(new_spec)
        total = len(DEPLOYMENTS)
    print(f"[RuntimeBuffer] Replaced deployment backbone '{old_backbone}' → '{new_spec.get('backbone')}'. "
          f"Total: {total}")


def append_deployments(new_specs: list):
    """Append new deployment specs to the existing DEPLOYMENTS list."""
    global DEPLOYMENTS
    with _STATE_LOCK:
        DEPLOYMENTS.extend(new_specs)
        total = len(DEPLOYMENTS)
    print(f"[RuntimeBuffer] Appended {len(new_specs)} deployment(s). "
          f"Total: {total}")
