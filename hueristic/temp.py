from config import *
import math

class TreeNode:
    """Node for representing a component in the pipeline tree."""
    def __init__(self, data, mem):
        self.data = data
        self.mem = mem
        self.children = []
    def add_child(self, child):
        self.children.append(child)

def create_tree():
    """Create a tree for each backbone, representing the pipeline architecture."""
    covered_backbones = {}
    for _, pipeline in pipelines.items():
        backbone = pipeline['architecture'][0]
        if backbone not in covered_backbones:
            root = TreeNode(backbone, components[backbone]['mem'])
            covered_backbones[backbone] = root
        else:
            root = covered_backbones[backbone]
        decoder = TreeNode(pipeline['architecture'][1], components[pipeline['architecture'][1]]['mem'])
        task = TreeNode(pipeline['architecture'][2], components[pipeline['architecture'][2]]['mem'])
        if tasks[task.data]['metric']=='accuracy' and pipeline['metric'] < tasks[task.data]['value']:
            continue  # Skip if pipeline metric is less than task requirement
        elif tasks[task.data]['metric']=='mae' and pipeline['metric'] > tasks[task.data]['value']:
            continue  # Skip if pipeline metric is less than task requirement
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
        child_size[node.data] = components[node.data]['mem']
        cap[node.data] = P - ancestor_size[node.data]
        count[node.data] = 1
        return ancestor_size,child_size, count
    
    total_size = 0
    cap[node.data] = P - ancestor_size[node.data]

    for child in node.children:
        ancestor_size[child.data]=ancestor_size[node.data]+components[node.data]['mem']
        ancestor_size, child_size, count = lower_bound_mem(child, P, ancestor_size, child_size, count, cap)
        total_size += child_size[child.data]
    
    if total_size == 0:
        count[node.data] = 1
    else:
        count[node.data] = math.ceil(total_size / (cap[node.data] - components[node.data]['mem']))
    child_size[node.data] = total_size + (count[node.data] * components[node.data]['mem'])
    return ancestor_size, child_size, count

def first_fit_binpack(bin, children, capacity):
    """First-fit bin packing for children under ancestor constraints."""
    #pack children into bin consisting of ancestor"

    for child in children:
        if child[1] + sum(bin.values()) > capacity:
            raise ValueError(f"Child {child[0].data} with size {child[1]} cannot fit in the server with ancestor size {sum(ancestor.values())} and capacity {capacity}")
        else:
            bin[child[0].data] = child[0].mem
            for c in child[0].children:
                bin[c.data] = c.mem
    return [bin]


def greedy_pack(root, P, ancestor_size, child_size, count):
    """Greedy packing of nodes into servers using bin packing."""
    servers = []
    server_id = 0
    if count[root.data] == 1:
        # Single server, all components in child_size
        components_dict = {comp: components[comp]['mem'] for comp in child_size.keys()}
        mem = sum(components_dict.values())
        tasks_dict = {task: [info['peak_workload'],100] for task, info in tasks.items() if task in child_size.keys()}
        servers.append({
            'name': f'server{server_id}',
            'tasks': tasks_dict,
            'memory': mem,
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
        bin = first_fit_binpack(bin, child_subtrees, P)

        components_dict = {comp: components[comp]['mem'] for comp in bin.keys()}
        mem = sum(bin.values())
        tasks_dict = {task: [info['peak_workload'],100] for task, info in tasks.items() if task in bin.keys()}
        servers.append({
                'name': f'server{server_id}',
                'tasks': tasks_dict,
                'memory': mem,
                'components': components_dict
            })  
          
        # for b in bins:
        #     components_dict = {comp: components[comp]['mem'] for comp in b.keys()}
        #     mem = sum(b.values())
        #     tasks_dict = {task: info['peak_workload'] for task, info in tasks.items() if task in b.keys()}
        #     servers.append({
        #         'name': f'server{server_id}',
        #         'tasks': tasks_dict,
        #         'memory': mem,
        #         'components': components_dict
        #     })
        #     server_id += 1
        # parent = find_parent(root, bottleneck_node)
        # if parent:
        #     remove_subtree(bottleneck_node, parent)
        # else:
        #     root = None
    return servers


def get_task_latency(tasks):
    """Helper to get task latencies for a server."""
    task_latency = {}
    for t in tasks:
        for pipeline in pipelines.values():
            if t in pipeline['architecture']:
                task_latency[t] = pipeline['latency']
    return task_latency


def check_workload(task_manifest):
    """Check if for all task in manifest combined sum of all latency*workload <= 1."""
    for srv in task_manifest:
        task_latency = get_task_latency(srv['tasks'])
        cap = sum(l * srv['tasks'][t][0] for t, l in task_latency.items())
        if cap > 1:
            # print(f"Warning: Server {srv['name']} exceeds capacity with {cap}")
            redundant_servers = math.ceil(cap)
            # reduce the workload for each task in manifest by dividing by redundant_servers
            # store it in % of total workload for each task in task_manifest
            for t in srv['tasks']:
                #store it in % of total workload for each task in task_manifest
                srv['tasks'][t] = [srv['tasks'][t][0],100 / redundant_servers]
    return task_manifest
        

def shared_packing():
    """Main function to run shared packing and print results."""
    #create trees for each backbone, keep task on backbone which statisfies it.
    trees = create_tree()

    # #descending order of devices for config based on memory
    servers = []
    for device_name, device_info in sorted(devices.items(), key=lambda x: x[1]['mem'], reverse=True):
        servers.append({'name': device_name, 'memory': device_info['mem']})

    #sort the trees based on number of tasks (leaves) on root (descending)
    trees = sorted(trees.items(), key=lambda x: sum(1 for decoder in x[1].children for task in decoder.children), reverse=True)

    final_task_manifest = []
    while tasks:
        server=servers.pop()
        backbone, root=trees[0]
        print(f"Device: {server['name']}, Memory: {server['memory']} MB")
        device_memory = server['memory']
        
        #based on memory packing
        ancestor_size, child_size, count = lower_bound_mem(root, device_memory)
        task_manifest = greedy_pack(root, device_memory, ancestor_size, child_size, count)
        #check workload for all task in task_manifest 
        task_manifest = check_workload(task_manifest)

        #only if 100% workload is covered then remove packed tasks from tasks list and trees
        #else update the peak workload of tasks based on packed workload in task_manifest
        packed_tasks = {}
        for srv in task_manifest:
            for t in srv['tasks']:
                if srv['tasks'][t][1] == 100:
                    packed_tasks[t] = True
                else:
                    tasks[t]['peak_workload'] -= srv['tasks'][t][0] * (srv['tasks'][t][1] / 100)
        
        #remove packed tasks from tasks and trees
        for t in packed_tasks:
            if t in tasks:
                del tasks[t]
            #remove from tree not just this root all other roots where this task is present
            for backbone_key, tree_root in trees:
                root_to_check = tree_root
                for decoder in root.children:
                    root.children = [d for d in root.children if all(task.data != t for task in d.children)]
        trees[0]=(backbone, root)
        final_task_manifest.extend(task_manifest)

    return final_task_manifest
    # for device in devices:
    #     print(f"Device: {device['name']}, Memory: {device['memory']} MB")
    #     device_memory = device['memory']

        
    #     for backbone, root in shared_trees.items():
    #         ancestor_size, child_size, count = lower_bound_mem(root, P)
    #         servers = greedy_pack(root, P, ancestor_size, child_size, count)
    #         servers = generate_redundancy(servers)
    #         print(f"Backbone: {backbone}")
    #         for server in servers:
    #             print(server)


if __name__ == "__main__":
    task_manifest=shared_packing()
    print(task_manifest)
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
