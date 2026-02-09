DEPLOYMENTS = []
RUNTIME_REQUESTS = []
OUTPUT_DIR = None  # Set by orchestrator to save results in the right experiment directory


def clear_state():
    """Reset all stored state between experiments.
    
    Must be called at the start of each new deployment to prevent
    request/state accumulation across experiments.
    """
    global DEPLOYMENTS, RUNTIME_REQUESTS, OUTPUT_DIR
    DEPLOYMENTS = []
    RUNTIME_REQUESTS = []
    OUTPUT_DIR = None
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
    return RUNTIME_REQUESTS


def get_deployments():
    return DEPLOYMENTS


def get_output_dir():
    return OUTPUT_DIR