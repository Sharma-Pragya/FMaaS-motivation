DEPLOYMENTS=[]
RUNTIME_REQUESTS=[]

def store_plan_and_requests(payload:dict):
    global DEPLOYMENTS, RUNTIME_REQUESTS
    DEPLOYMENTS = payload.get("deployments",[])
    RUNTIME_REQUESTS = payload.get("runtime_requests", [])
    print(f"[RuntimeBuffer] Stored {len(DEPLOYMENTS)}  \
          deployment and {len(RUNTIME_REQUESTS)} runtime requests")

def get_requests():
    return RUNTIME_REQUESTS