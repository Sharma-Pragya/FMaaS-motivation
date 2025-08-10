
class RSU:
    def __init__(self, id, cpu_freq, gpu_flops, has_gpu, storage_capacity):
        self.id = id
        self.cpu_freq = cpu_freq
        self.gpu_flops = gpu_flops if has_gpu else None
        self.has_gpu = has_gpu
        self.storage_capacity = storage_capacity
        self.model_cache = set()

class Model:
    def __init__(self, id, tasks, storage, cpu_costs, gpu_costs, load_time):
        self.id = id
        self.tasks = tasks
        self.storage = storage
        self.cpu_costs = cpu_costs
        self.gpu_costs = gpu_costs
        self.load_time = load_time

class Request:
    def __init__(self, id, tasks, input_size, tolerance):
        self.id = id
        self.tasks = tasks
        self.input_size = input_size
        self.tolerance = tolerance

def compute_inference_delay(rsu, model, task, use_gpu):
    if use_gpu and rsu.has_gpu:
        return model.gpu_costs[task] / rsu.gpu_flops
    else:
        return model.cpu_costs[task] / rsu.cpu_freq

def communication_delay(input_size, bandwidth):
    return input_size / bandwidth

def model_download_delay(model, bandwidth):
    return model.storage / bandwidth

def model_loading_delay(model, model_cached):
    return 0 if model_cached else model.load_time

def all_task_types(requests):
    return set(t for req in requests for t in req.tasks)

def ITA(task_requests, rsus, models, T):
    deployment = {}
    task_assignment = {}
    for task_type in all_task_types(task_requests):
        best_rsu = None
        best_model = None
        max_covered = -1
        for rsu in rsus:
            for model in models:
                if task_type not in model.tasks or model.storage > rsu.storage_capacity:
                    continue
                covered_tasks = sum(1 for req in task_requests if task_type in req.tasks)
                if covered_tasks > max_covered:
                    max_covered = covered_tasks
                    best_rsu = rsu
                    best_model = model
        if best_rsu and best_model:
            deployment.setdefault(best_rsu.id, set()).add(best_model.id)
            for req in task_requests:
                if task_type in req.tasks:
                    task_assignment[(req.id, task_type)] = best_rsu.id
    return deployment, task_assignment

def get_rsu_by_id(rsu_id, rsus):
    return next(r for r in rsus if r.id == rsu_id)

def get_model_for_task(task, models):
    for model in models:
        if task in model.tasks:
            return model
    return None

def compute_total_delay(req, task, rsu, model):
    use_gpu = rsu.has_gpu
    inference = compute_inference_delay(rsu, model, task, use_gpu)
    transfer = communication_delay(req.input_size, 100)
    model_dl = model_download_delay(model, 500)
    load = model_loading_delay(model, model.id in rsu.model_cache)
    return inference + transfer + model_dl + load

def TPA(task_requests, rsus, models, T_max, T_min, T_tol):
    while T_max - T_min > T_tol:
        T_mid = (T_max + T_min) / 2
        deployment, task_assignment = ITA(task_requests, rsus, models, T_mid)
        all_completed = True
        for req in task_requests:
            max_delay = 0
            for task in req.tasks:
                rsu_id = task_assignment.get((req.id, task))
                if rsu_id is None:
                    all_completed = False
                    break
                rsu = get_rsu_by_id(rsu_id, rsus)
                model = get_model_for_task(task, models)
                delay = compute_total_delay(req, task, rsu, model)
                max_delay = max(max_delay, delay)
            if not all_completed or max_delay > req.tolerance:
                all_completed = False
                break
        if all_completed:
            T_max = T_mid
        else:
            T_min = T_mid
    final_deployment, final_assignment = ITA(task_requests, rsus, models, T_max)
    return final_deployment, final_assignment
