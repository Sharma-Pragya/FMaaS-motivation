from config import *
import math
class TreeNode:
    def __init__(self,data,mem):
        self.data=data
        self.mem=mem
        self.children=[]
    def add_child(self,child):
        self.children.append(child)
        

def create_tree():
    covered_backbones={}
    for _,pipeline in pipelines.items():
        backbone=pipeline['architecture'][0]
        if backbone not in covered_backbones:
            root=TreeNode(backbone,components[backbone]['mem'])
            covered_backbones.update({backbone:root})            
        else:
            root=covered_backbones[backbone]
        decoder=TreeNode(pipeline['architecture'][1],components[pipeline['architecture'][1]]['mem'])
        task=TreeNode(pipeline['architecture'][2],components[pipeline['architecture'][2]]['mem'])
        decoder.add_child(task)
        root.add_child(decoder)
    return covered_backbones

def lower_bound_mem(node, P, ancestors_weight=0,ancestor_size={},child_size={},count={},cap={}):
    #leaf
    if not node.children:
        child_size[node.data]=components[node.data]['mem']
        count[node.data]=1
        cap[node.data]=P-ancestors_weight
        ancestor_size[node.data]=ancestors_weight
        return ancestor_size,child_size,count
    
    total_size=0
    for child in node.children:
        ancestor_size,child_size, count = lower_bound_mem(child,P,ancestors_weight+components[node.data]['mem'],ancestor_size,child_size,count)
        total_size+=child_size[child.data]
    cap[node.data]=P-ancestors_weight
    ancestor_size[node.data]=ancestors_weight
    if total_size==0:
        count[node.data]=1
    else:
        count[node.data]=math.ceil(total_size/(cap[node.data]-components[node.data]['mem']))
    child_size[node.data]=total_size+(count[node.data]*components[node.data]['mem'])
    return ancestor_size,child_size, count


def first_fit_binpack(ancestor,children,capacity):
    print('ancestor')
    print(ancestor)
    bins=[ancestor.copy()]
    for child in children:
        placed = False
        for b in bins:
            s=sum([value for value in b.values()])
            if s + child[1] <=capacity:
                #this should be recursive like dfs even ancestor should be recursive
                b.update({child[0].data:child[0].mem})
                for c in child[0].children:
                    b.update({c.data:c.mem})
                placed = True
                break
        if not placed:
            cpy_bin=ancestor.copy()
            cpy_bin.update({child[0].data:child[0].mem})
            for c in child[0].children:
                cpy_bin.update({c.data:c.mem})
            bins.append(cpy_bin)
    return bins


def greedy_pack(root,P,ancestor_size,child_size,count):
    servers=[]
    if count[root.data]==1:
        print(child_size)

        servers.append(child_size)

        return servers
    # curr_root=root
    while root!=None:
        stack=[root]
        while stack:
            node=stack.pop()
            check_child=[count[c.data]==1 for c in node.children]
            if count[node.data] > 1 and all(check_child):
                bottleneck_node=node
                break
            stack.extend(node.children)
        
        #first fit bin pack
        child_subtrees=[[c,child_size[c.data]] for c in bottleneck_node.children]
        bins = first_fit_binpack({bottleneck_node.data:ancestor_size[bottleneck_node.data]+bottleneck_node.mem},child_subtrees, P)
        servers.extend(bins)
        def find_parent(node,bottleneck_node):
            for c in node.children:
                if c==bottleneck_node:
                    return node
                p=find_parent(c,bottleneck_node)
                if p:
                    return p
            return None
        
        def remove_subtree(node, parent=None):
            if parent:
                parent.children = [c for c in parent.children if c != node]
            # node.children = []

        parent = find_parent(root, bottleneck_node)
        if parent:
            remove_subtree(bottleneck_node, parent)
        else: 
            root=None
    return servers

def generate_redundancy(servers):
    s={}
    for i,server in enumerate(servers):
        server_task={}
        for task, info in tasks.items():
            if task in server:
                server_task.update({task:info['peak_workload']})
        s[f'server{i}']=server_task
    workload_server={}
    for index,(server_name,server) in enumerate(s.items()):
        task_latency={}
        for t in server.keys():
            for pipeline in pipelines.values():
                if t in pipeline['architecture']:
                    task_latency[t]=pipeline['latency']
        cap=0
        for t,l in task_latency.items():
            cap+=l*tasks[t]['peak_workload']
        print(cap)
        if cap>1:
            redundant_servers=math.ceil(cap)
            for i in range(redundant_servers):
                workload_server[f'server_{index}_{i}']={}
                for t,l in task_latency.items():
                    workload_server[f'server_{index}_{i}'].update({t:tasks[t]['peak_workload']/redundant_servers})
        else:
            workload_server[f'server_{index}_0']={}
            for t,l in task_latency.items():
                workload_server[f'server_{index}_0'].update({t:tasks[t]['peak_workload']})
    return workload_server       
    
def shared_packing():
    P=16000
    shared_trees=create_tree()
    for backbone,root in shared_trees.items():
        ancestor_size,child_size,count=lower_bound_mem(root,P)

        servers = greedy_pack(root, P,ancestor_size,child_size, count)
        workload_server=generate_redundancy(servers)
        print(servers)
        print(workload_server)

shared_packing()