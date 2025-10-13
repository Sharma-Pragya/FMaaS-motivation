# orchestrator/router.py
ROUTING_TABLE = {
    "hr": ("http://10.100.1.3:8000/predict", "http://10.100.20.18:8000/predict"),
    "diasbp": ("http://10.100.1.3:8000/predict", "http://10.100.20.18:8000/predict"),
    "sysbp": ("http://10.100.1.3:8000/predict", "http://10.100.20.18:8000/predict"),
    # "etth1": ("http://10.100.1.3:8000/predict", "http://10.100.115.7:8000/predict"),
    # "weather": ("http://10.100.1.3:8000/predict", "http://10.100.115.7:8000/predict"),
}

def get_route(task: str):
    """Return (site_manager, device) for a given task"""
    if task not in ROUTING_TABLE:
        raise ValueError(f"No routing defined for task {task}")
    return ROUTING_TABLE[task]
