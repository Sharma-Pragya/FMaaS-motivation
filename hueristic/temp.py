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
        decoder.add_child(task)
        root.add_child(decoder)
    return covered_backbones

def lower_bound_mem(node, P, ancestors_weight=0, ancestor_size=None, child_size=None, count=None, cap=None):
    """Recursively compute memory lower bounds for each node."""
    if ancestor_size is None:
        ancestor_size, child_size, count, cap = {}, {}, {}, {}
    # Leaf node
    if not node.children:
        child_size[node.data] = components[node.data]['mem']
        count[node.data] = 1
        cap[node.data] = P - ancestors_weight
        ancestor_size[node.data] = ancestors_weight
        return ancestor_size, child_size, count
    total_size = 0
    for child in node.children:
        ancestor_size, child_size, count = lower_bound_mem(child, P, ancestors_weight + components[node.data]['mem'], ancestor_size, child_size, count,cap)
        total_size += child_size[child.data]
    cap[node.data] = P - ancestors_weight
    ancestor_size[node.data] = ancestors_weight
    if total_size == 0:
        count[node.data] = 1
    else:
        count[node.data] = math.ceil(total_size / (cap[node.data] - components[node.data]['mem']))
    child_size[node.data] = total_size + (count[node.data] * components[node.data]['mem'])
    return ancestor_size, child_size, count

def first_fit_binpack(ancestor, children, capacity):
    """First-fit bin packing for children under ancestor constraints."""
    bins = [ancestor.copy()]
    for child in children:
        placed = False
        for b in bins:
            s = sum(b.values())
            if s + child[1] <= capacity:
                b[child[0].data] = child[0].mem
                for c in child[0].children:
                    b[c.data] = c.mem
                placed = True
                break
        if not placed:
            cpy_bin = ancestor.copy()
            cpy_bin[child[0].data] = child[0].mem
            for c in child[0].children:
                cpy_bin[c.data] = c.mem
            bins.append(cpy_bin)
    return bins

def find_parent(node, target):
    """Find parent of a given node in the tree."""
    for c in node.children:
        if c == target:
            return node
        p = find_parent(c, target)
        if p:
            return p
    return None

def remove_subtree(node, parent=None):
    """Remove a subtree from its parent."""
    if parent:
        parent.children = [c for c in parent.children if c != node]

def greedy_pack(root, P, ancestor_size, child_size, count):
    """Greedy packing of nodes into servers using bin packing."""
    servers = []
    server_id = 0
    if count[root.data] == 1:
        # Single server, all components in child_size
        components_dict = {comp: components[comp]['mem'] for comp in child_size.keys()}
        mem = sum(components_dict.values())
        tasks_dict = {task: info['peak_workload'] for task, info in tasks.items() if task in child_size.keys()}
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
        bins = first_fit_binpack({bottleneck_node.data: ancestor_size[bottleneck_node.data] + bottleneck_node.mem}, child_subtrees, P)
        for b in bins:
            components_dict = {comp: components[comp]['mem'] for comp in b.keys()}
            mem = sum(b.values())
            tasks_dict = {task: info['peak_workload'] for task, info in tasks.items() if task in b.keys()}
            servers.append({
                'name': f'server{server_id}',
                'tasks': tasks_dict,
                'memory': mem,
                'components': components_dict
            })
            server_id += 1
        parent = find_parent(root, bottleneck_node)
        if parent:
            remove_subtree(bottleneck_node, parent)
        else:
            root = None
    return servers


def get_task_latency(tasks):
    """Helper to get task latencies for a server."""
    task_latency = {}
    for t in tasks:
        for pipeline in pipelines.values():
            if t in pipeline['architecture']:
                task_latency[t] = pipeline['latency']
    return task_latency



def generate_redundancy(servers):
    """Update servers list to split servers if latency*workload > 1, and store all info in servers."""
    new_servers = []
    for server in servers:
        task_latency = get_task_latency(server['tasks'])
        cap = sum(l * server['tasks'][t] for t, l in task_latency.items())
        if cap > 1:
            redundant_servers = math.ceil(cap)
            for i in range(redundant_servers):
                split_tasks = {t: server['tasks'][t] / redundant_servers for t in server['tasks']}
                new_servers.append({
                    'name': f"{server['name']}_{i}",
                    'tasks': split_tasks,
                    'memory': server['memory'],
                    'components': server['components']
                })
        else:
            new_servers.append(server)
    return new_servers

def shared_packing():
    """Main function to run shared packing and print results."""
    shared_trees = create_tree()
    for backbone, root in shared_trees.items():
        ancestor_size, child_size, count = lower_bound_mem(root, P)
        servers = greedy_pack(root, P, ancestor_size, child_size, count)
        servers = generate_redundancy(servers)
        print(f"Backbone: {backbone}")
        for server in servers:
            print(server)
def normal_packing():
    """Create a server for each task and its components, then apply redundancy."""
    servers = []
    server_id = 0
    for task in tasks:
        # Find the pipeline that contains this task
        found_pipeline = None
        for pipeline in pipelines.values():
            if task == pipeline['architecture'][2]:
                found_pipeline = pipeline
                break
        if found_pipeline is None:
            continue  # skip if not found
        backbone = found_pipeline['architecture'][0]
        decoder = found_pipeline['architecture'][1]
        components_list = [backbone, decoder, task]
        components_dict = {comp: components[comp]['mem'] for comp in components_list}
        tasks_dict = {task: tasks[task]['peak_workload']}
        mem = sum(components[comp]['mem'] for comp in components_list)
        servers.append({
            'name': f'server{server_id}',
            'tasks': tasks_dict,
            'memory': mem,
            'components': components_dict
        })
        server_id += 1
    servers = generate_redundancy(servers)
    print("Normal Packing:")
    for server in servers:
        print(server)
        
if __name__ == "__main__":
    P = 16000
    shared_packing()
    print("---\n")
    normal_packing()