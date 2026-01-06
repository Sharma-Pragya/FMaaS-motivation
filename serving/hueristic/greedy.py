from hueristic.parser.profiler import *
import math
import copy
import json

class TreeNode:
    """Node for representing a component in the pipeline tree."""
    def __init__(self, data, mem):
        self.data = data
        self.mem = mem
        self.children = []
    def add_child(self, child):
        self.children.append(child)

def create_tree(tasks,server_type):
    """Create a tree for each backbone, representing the pipeline architecture."""
    covered_backbones = {}
    for id, pipeline in pipelines.items():
        backbone = pipeline['backbone']
        if backbone not in covered_backbones:
            root = TreeNode(backbone, components[backbone]['mem'])
            covered_backbones[backbone] = root
        else:
            root = covered_backbones[backbone]
        decoder = TreeNode(f"{pipeline['decoder']}_{pipeline['backbone']}_{pipeline['task']}", components[f"{pipeline['decoder']}_{pipeline['backbone']}_{pipeline['task']}"]['mem'])
        task = TreeNode(pipeline['task'], components[f"{pipeline['task']}_{pipeline['backbone']}_{pipeline['decoder']}"]['mem'])
        if task.data in tasks:
            if tasks[task.data]['metric']=='accuracy' and metric[id] < tasks[task.data]['value']:
                continue  # Skip if pipeline metric is less than task requirement
            elif tasks[task.data]['metric']=='mae' and metric[id] > tasks[task.data]['value']:
                continue  # Skip if pipeline metric is less than task requirement
            elif latency[id][server_type]>tasks[task.data]['latency']:
                continue
            print(f"Adding task {task.data} under backbone {backbone}")
            decoder.add_child(task)
            root.add_child(decoder)
    return covered_backbones

def lower_bound_mem(node, P, ancestor_size=None, child_size=None, count=None, cap=None):
    # count(v) is the number of servers needed to pack the tree rooted at v. (v is not included)
    # child size(v) represent the total size (in MB) required by the LB process to pack the subtree rooted at v (v is not included).
    # ancestor size(v) represent the total size (in MB) of the ancestor before v
    # cap(v) is the capacity size (in MB) needed by ancester (v is not included)
    """Recursively compute memory lower bounds for each node."""
    if ancestor_size is None:
        ancestor_size, child_size, count, cap = {node.data:0}, {}, {}, {}

    # Leaf node
    if not node.children:
        child_size[node.data] = node.mem
        cap[node.data] = P - ancestor_size[node.data]
        count[node.data] = 1
        return ancestor_size,child_size, count
    
    total_size = 0
    cap[node.data] = P - ancestor_size[node.data]

    for child in node.children:
        ancestor_size[child.data]=ancestor_size[node.data]+node.mem
        ancestor_size, child_size, count = lower_bound_mem(child, P, ancestor_size, child_size, count, cap)
        total_size += child_size[child.data]
    
    if total_size == 0:
        count[node.data] = 1
    else:
        count[node.data] = math.ceil(total_size / (cap[node.data] - node.mem))
    child_size[node.data] = total_size + (count[node.data] * node.mem)
    return ancestor_size, child_size, count

def first_fit_binpack(bin, children, capacity):
    """First-fit bin packing for children under ancestor constraints."""
    #pack children into bin consisting of ancestor"

    for child in children:
        if child[1] + sum(bin.values()) > capacity:
            raise ValueError(f"Child {child[0].data} with size {child[1]} cannot fit in the server with ancestor size {sum(bin.values())} and capacity {capacity}")
        else:
            bin[child[0].data] = child[0].mem
            for c in child[0].children:
                bin[c.data] = c.mem
    return [bin]


def greedy_pack(root, tasks, server, ancestor_size, child_size, count):
    """Greedy packing of nodes into servers using bin packing."""
    servers = []
    server_id = 0
    if count[root.data] == 1:
        # Single server, all components in child_size
        # components_dict = {comp: components[comp]['mem'] for comp in child_size.keys()}
        # mem = sum(components_dict.values())
        components_dict = [comp for comp in child_size.keys()]

        tasks_dict = {task: {'type':info['type'],'total_requested_workload':info['peak_workload'],'request_per_sec':info['peak_workload']} for task, info in tasks.items() if task in child_size.keys()}

        servers.append({
            'site_manager':server['site_manager'],
            'name': server['name'],
            'ip': server['ip'],
            'type': server['type'],
            'total_memory': server['memory'],
            'tasks': tasks_dict,
            'components': components_dict
        })
        return servers
    while root is not None:
        stack = [root]
        bottleneck_node = None
        while stack:
            node = stack.pop()
            check_child = [count[c.data] == 1 for c in node.children]
            if count[node.data] > 1 and all(check_child):
                bottleneck_node = node
                break
            stack.extend(node.children)
        if bottleneck_node is None:
            break
        child_subtrees = [[c, child_size[c.data]] for c in bottleneck_node.children]
        bin={bottleneck_node.data: ancestor_size[bottleneck_node.data] + bottleneck_node.mem}
        bin = first_fit_binpack(bin, child_subtrees, server['memory'])

        # components_dict = {comp: components[comp]['mem'] for comp in bin.keys()}
        # mem = sum(bin.values())
        components_dict = [comp for comp in child_size.keys()]

        tasks_dict = {task: {'type':info['type'],'total_requested_workload':info['peak_workload'],'request_per_sec':info['peak_workload']} for task, info in tasks.items() if task in bin.keys()}
        servers.append({
            'site_manager':server['site_manager'],
            'name': server['name'],
            'ip': server['ip'],
            'type': server['type'],
            'total_memory': server['memory'],
            'tasks': tasks_dict,
            'components': components_dict
        }) 

    return servers


def get_task_latency(backbone,tasks):
    """Helper to get task latencies for a server."""
    task_latency = {}
    for t in tasks:
        for id,pipeline in pipelines.items():
            if t==pipeline['task'] and backbone==pipeline['backbone']:
                task_latency[t] = latency[id]
    return task_latency


def check_workload(backbone,task_manifest,device_type):
    """Check if for all task in manifest combined sum of all latency*workload <= 1."""
    for srv in task_manifest:
        task_latency = get_task_latency(backbone,srv['tasks'])
        # as latency is in ms convert to seconds
        cap = sum(l[device_type] * srv['tasks'][t]['total_requested_workload'] for t, l in task_latency.items())/1000
        if cap > 1:
            # print(f"Warning: Server {srv['name']} exceeds capacity with {cap}")
            redundant_servers = math.ceil(cap)
            print(f"Splitting server {srv['name']} into {redundant_servers} servers to meet workload requirements.")
            # reduce the workload for each task in manifest by dividing by redundant_servers
            # store it in % of total workload for each task in task_manifest
            for t in srv['tasks']:
                #store it in % of total workload for each task in task_manifest
                srv['tasks'][t]['type'] = srv['tasks'][t]['type']
                srv['tasks'][t]['total_requested_workload'] = srv['tasks'][t]['total_requested_workload']
                srv['tasks'][t]['request_per_sec'] = srv['tasks'][t]['total_requested_workload'] / redundant_servers
    return task_manifest
        

def shared_packing(devices, tasks):
    """Main function to run shared packing and print results."""
    # #descending order of devices for config based on memory
    servers = []
    for device_name, device_info in sorted(devices.items(), key=lambda x: x[1]['mem'], reverse=True):
        servers.append({'name': device_name,'type':device_info['type'], 'memory': device_info['mem'],'ip':device_info['ip'],'site_manager':device_info['site_manager']})

    final_task_manifest = []
    # use a deep copy so nested dicts (like per-task dicts) are not shared
    # between tasks_copy and the original tasks mapping
    # tasks_copy = copy.deepcopy(tasks)
    while tasks and servers:
        server=servers.pop()
        print(f"Device: {server['name']}, Memory: {server['memory']} MB, Type: {server['type']}")
        device_memory = server['memory']
        device_type = server['type']
        device_name = server['name']
        #create trees for each backbone, keep task on backbone which statisfies it.
        trees = create_tree(tasks,device_type)

        #sort the trees based on number of tasks (leaves) on root (descending)
        trees = sorted(trees.items(), key=lambda x: sum(1 for decoder in x[1].children for task in decoder.children), reverse=True)
        
        backbone, root=trees[0]
        #based on memory packing
        ancestor_size, child_size, count = lower_bound_mem(root, device_memory)
        task_manifest = greedy_pack(root,tasks, server, ancestor_size, child_size, count)
        #check workload for all task in task_manifest 
        task_manifest = check_workload(backbone,task_manifest,device_type)

        #only if 100% workload is covered then remove packed tasks from tasks list and trees
        #else update the peak workload of tasks based on packed workload in task_manifest
        packed_tasks = {}
        for srv in task_manifest:
            for t in srv['tasks']:
                if srv['tasks'][t]['total_requested_workload'] == srv['tasks'][t]['request_per_sec']:
                    packed_tasks[t] = True
                else:
                    tasks[t]['peak_workload'] -= srv['tasks'][t]['request_per_sec']
        
        #remove packed tasks from tasks and trees
        for t in packed_tasks:
            if t in tasks:
                del tasks[t]
            #remove from tree not just this root all other roots where this task is present
            for _, tree_root in trees:
                tree_root.children = [d for d in tree_root.children if all(task.data != t for task in d.children)]

        trees = [t for t in trees if t[1].children]
        final_task_manifest.extend(task_manifest)
        print(f"Remaining tasks to pack: {list(tasks.keys())}")
    return final_task_manifest


def build_final_json(device_list):
    """
    Build deployment JSON grouped by site_manager using pipeline config.
    Each device's components are matched against pipeline definitions
    to infer backbone and decoders properly.
    """

    sites = {}
    # reverse lookup: decoder -> backbone
    decoder_to_task = {f"{v['decoder']}_{v['backbone']}_{v['task']}": v['task'] for v in pipelines.values()}
    decoder_to_fulltask = {f"{v['decoder']}_{v['backbone']}_{v['task']}":f"{v['task']}_{v['backbone']}_{v['decoder']}"  for v in pipelines.values()}
    decoder_to_backbone = {f"{v['decoder']}_{v['backbone']}_{v['task']}": v['backbone'] for v in pipelines.values()}
    port = 8001
    for d in device_list:
        # port += 1
        site_id = d["site_manager"]
        device_url = f"{d['ip']}:{port}"
        device_type = d['type']
        tasks_info = d["tasks"]

        components = set(d["components"])

        # find all decoders on this device that match known ones
        decoders = [dec for dec in components if dec in decoder_to_backbone]

        # infer backbone: any backbone that corresponds to these decoders
        backbone = None
        for dec in decoders:
            possible_backbone = decoder_to_backbone[dec]
            if possible_backbone in components:
                backbone = possible_backbone
                break
        
        decoders_list = []
        for decoder in decoders:
            task=decoder_to_task[decoder]
            full_task=decoder_to_fulltask[decoder]
            decoders_list.append({"task": task, "type": tasks_info[task]['type'], "path":full_task})
            # build deployment entry
        deployment = {
            "device": device_url,
            "device_type": device_type,
            "backbone": backbone,
            "decoders": decoders_list,
            "tasks": tasks_info

        }

        # initialize site entry
        if site_id not in sites:
            sites[site_id] = {
                "id": site_id,
                "deployments": []
            }

        sites[site_id]["deployments"].append(deployment)

    final_json = {"sites": list(sites.values())}
    return final_json


if __name__ == "__main__":
    from serving.orchestrator.user_config import devices, tasks
    task_manifest=shared_packing(devices,tasks)
    final_json = build_final_json(task_manifest)
    output_path = "../deployment_plan.json"
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=2)
    # print("---\n")
    # normal_packing()

# def generate_redundancy(servers):
#     """Update servers list to split servers if latency*workload > 1, and store all info in servers."""
#     new_servers = []
#     for server in servers:
#         task_latency = get_task_latency(server['tasks'])
#         cap = sum(l * server['tasks'][t] for t, l in task_latency.items())
#         if cap > 1:
#             redundant_servers = math.ceil(cap)
#             for i in range(redundant_servers):
#                 split_tasks = {t: server['tasks'][t] / redundant_servers for t in server['tasks']}
#                 new_servers.append({
#                     'name': f"{server['name']}_{i}",
#                     'tasks': split_tasks,
#                     'memory': server['memory'],
#                     'components': server['components']
#                 })
#         else:
#             new_servers.append(server)
#     return new_servers


# def first_fit_binpack(ancestor, children, capacity):
#     """First-fit bin packing for children under ancestor constraints."""
#     #pack children into bin consisting of ancestor"
#     bins = [ancestor.copy()]
#     for child in children:
#         placed = False
#         for b in bins:
#             s = sum(b.values())
#             if s + child[1] <= capacity:
#                 b[child[0].data] = child[0].mem
#                 for c in child[0].children:
#                     b[c.data] = c.mem
#                 placed = True
#                 break
#         if not placed:
#             cpy_bin = ancestor.copy()
#             cpy_bin[child[0].data] = child[0].mem
#             for c in child[0].children:
#                 cpy_bin[c.data] = c.mem
#             bins.append(cpy_bin)
#     return bins

# def normal_packing():
#     """Create a server for each task and its components, then apply redundancy."""
#     servers = []
#     server_id = 0
#     for task in tasks:
#         # Find the pipeline that contains this task
#         found_pipeline = None
#         for pipeline in pipelines.values():
#             if task == pipeline['architecture'][2]:
#                 found_pipeline = pipeline
#                 break
#         if found_pipeline is None:
#             continue  # skip if not found
#         backbone = found_pipeline['architecture'][0]
#         decoder = found_pipeline['architecture'][1]
#         components_list = [backbone, decoder, task]
#         components_dict = {comp: components[comp]['mem'] for comp in components_list}
#         tasks_dict = {task: tasks[task]['peak_workload']}
#         mem = sum(components[comp]['mem'] for comp in components_list)
#         servers.append({
#             'name': f'server{server_id}',
#             'tasks': tasks_dict,
#             'memory': mem,
#             'components': components_dict
#         })
#         server_id += 1
#     servers = generate_redundancy(servers)
#     print("Normal Packing:")
#     for server in servers:
#         print(server)
        

# def find_parent(node, target):
#     """Find parent of a given node in the tree."""
#     for c in node.children:
#         if c == target:
#             return node
#         p = find_parent(c, target)
#         if p:
#             return p
#     return None

# def remove_subtree(node, parent=None):
#     """Remove a subtree from its parent."""
#     if parent:
#         parent.children = [c for c in parent.children if c != node]
