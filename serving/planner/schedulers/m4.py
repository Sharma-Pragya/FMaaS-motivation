"""M4-style deployment scheduler.

This module implements an M4-style scheduling algorithm where:
1. The first task selects the best backbone (by latency or accuracy)
2. ALL subsequent tasks use that SAME backbone
3. Replication on multiple servers if capacity is exceeded
"""

import heapq
import logging
from typing import Dict, List, Optional, Tuple

from ..models import Server, Deployment, TaskInfo, TaskSpec
from ..data_loader import ProfileData
from ..state import DeploymentState
from ..config import SchedulerConfig, TaskConstraints, DEFAULT_CONFIG
from .base import BaseScheduler


logger = logging.getLogger(__name__)


class M4Scheduler(BaseScheduler):
    """M4-style scheduler - all tasks share ONE backbone selected by first task.
    
    This scheduler deploys tasks to devices by:
    1. First task selects the BEST backbone (by latency or accuracy)
    2. ALL subsequent tasks use that SAME backbone
    3. Tasks can share deployments on the same backbone
    4. Replication to multiple servers if capacity exceeded
    
    Attributes:
        data: ProfileData instance with component/pipeline information.
        config: SchedulerConfig with scheduling parameters.
    """
    
    def schedule(
        self, 
        devices: Dict[str, Dict], 
        tasks: Dict[str, Dict],
        accuracy_mode: bool = False
    ) -> List[Deployment]:
        """Schedule tasks onto devices.
        
        Args:
            devices: Dictionary of device configurations.
            tasks: Dictionary of task specifications.
            accuracy_mode: If True, select backbone by accuracy; else by latency.
            
        Returns:
            List of Deployment objects representing the deployment plan.
        """
        # Initialize state with servers
        servers = self._create_servers(devices)
        state = DeploymentState(servers)
        
        # Sort tasks by peak workload (highest first)
        sorted_tasks = self._sort_tasks_by_workload(tasks)
        
        # First task determines the backbone for ALL tasks
        self._selected_backbone = None
        
        # Deploy each task
        for task_name, task_spec in sorted_tasks:
            task = self._create_task_spec(task_name, task_spec)
            
            # First task selects the backbone
            if self._selected_backbone is None:
                self._selected_backbone = self._select_best_backbone(task, accuracy_mode)
                logger.info(f"M4: Selected backbone '{self._selected_backbone}' based on first task '{task_name}'")
            
            temp_plan, demand_left = self._deploy_task(state, task)
            
            # Log admission failure if demand not fully satisfied
            if demand_left is not None and demand_left > self.config.demand_epsilon:
                logger.warning(
                    f"M4: task '{task_name}' has {demand_left:.4f} rps "
                    f"unsatisfied demand out of {task.peak_workload:.4f} rps"
                )
            
            # Commit the plan for this task
            if temp_plan:
                for deployment in temp_plan.values():
                    state.add_deployment(
                        deployment, 
                        self.config.base_port, 
                        self.config.port_increment
                    )
        
        logger.info(f"Final deployment count: {state.get_deployment_count()}")
        return state.get_all_deployments()
    
    def _sort_tasks_by_workload(
        self, 
        tasks: Dict[str, Dict]
    ) -> List[Tuple[str, Dict]]:
        """Sort tasks by peak workload, highest first."""
        return sorted(
            tasks.items(),
            key=lambda x: x[1].get('peak_workload', 0),
            reverse=True
        )
    
    def _create_task_spec(self, name: str, spec: Dict) -> TaskSpec:
        """Create TaskSpec from dictionary specification."""
        return TaskSpec(
            name=name,
            type=spec['type'],
            peak_workload=spec['peak_workload'],
            latency=spec.get('latency', float('inf')),
            metric=spec.get('metric', 'mae'),
            value=spec.get('value', 0)
        )
    
    def _select_best_backbone(self, task: TaskSpec, accuracy_mode: bool) -> str:
        """Select the best backbone based on the first task."""
        pid_backbones = self.data.get_backbones_for_task(task.name)
        
        if accuracy_mode:
            backbones = self._sort_backbones_by_accuracy(pid_backbones, task)
        else:
            backbones = self._sort_backbones_by_latency(pid_backbones, task)
        
        return backbones[0] if backbones else None
    
    def _deploy_task(
        self, 
        state: DeploymentState, 
        task: TaskSpec
    ) -> Tuple[Optional[Dict], Optional[float]]:
        """Deploy a task using the pre-selected backbone.
        
        M4 behavior: All tasks use the same backbone selected by the first task.
        Tasks CAN share deployments on the same (server, backbone).
        """
        backbone = self._selected_backbone
        if not backbone:
            return None, task.peak_workload
        
        # Check if this backbone can serve this task
        pid = self.data.find_pipeline_id(task.name, backbone)
        if not pid:
            logger.warning(f"M4: Backbone '{backbone}' cannot serve task '{task.name}'")
            return None, task.peak_workload
        
        # First try to use existing deployments with this backbone
        temp_plan = {}
        task_demand = task.peak_workload
        util_tracker = {}
        
        # Try existing deployments first (to share)
        existing_deployments = state.find_active_deployments(backbone, self.config.util_factor)
        for deployment in existing_deployments:
            if task_demand <= self.config.demand_epsilon:
                break
            
            server = state.get_server(deployment.server_name)
            if not server:
                continue
            
            if server.name not in util_tracker:
                util_tracker[server.name] = server.util
            
            total_util = util_tracker[server.name]
            left_cap = self.config.util_factor - total_util
            
            if left_cap <= 1e-6:
                continue
            
            latency = self.data.get_pipeline_latency(pid, server.type)
            if latency is None:
                continue
            
            # Calculate allocation
            task_cap_needed = task_demand * latency / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * 1000.0 / latency
            
            task_demand -= allocated_demand
            util_tracker[server.name] += allocated_cap
            
            # Get pipeline components
            pipeline = self.data.get_pipeline(pid)
            components = self.data.get_pipeline_components_mem(pipeline)
            
            # Create/update deployment - sharing with same backbone
            deployment_obj = Deployment(
                server_name=server.name,
                backbone=backbone,
                ip=server.ip,
                site_manager=server.site_manager,
                device_type=server.type,
                mem=server.mem,
                util=util_tracker[server.name],
                components=components,
                task_info={
                    task.name: TaskInfo(
                        type=task.type,
                        total_requested_workload=task.peak_workload,
                        request_per_sec=allocated_demand
                    )
                }
            )
            
            temp_plan[(server.name, backbone)] = deployment_obj
        
        # If demand not satisfied, add new servers with the same backbone
        for server in state.get_all_servers():
            if task_demand <= self.config.demand_epsilon:
                break
            
            if server.mem < self.data.get_component_mem(backbone):
                continue
            
            # Skip if already processed
            if (server.name, backbone) in temp_plan:
                continue
            
            if server.name not in util_tracker:
                util_tracker[server.name] = server.util
            
            total_util = util_tracker[server.name]
            left_cap = self.config.util_factor - total_util
            
            if left_cap <= 1e-6:
                continue
            
            latency = self.data.get_pipeline_latency(pid, server.type)
            if latency is None:
                continue
            
            # Calculate allocation
            task_cap_needed = task_demand * latency / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * 1000.0 / latency
            
            task_demand -= allocated_demand
            util_tracker[server.name] += allocated_cap
            
            # Get pipeline components
            pipeline = self.data.get_pipeline(pid)
            components = self.data.get_pipeline_components_mem(pipeline)
            
            # Create deployment
            deployment = Deployment(
                server_name=server.name,
                backbone=backbone,
                ip=server.ip,
                site_manager=server.site_manager,
                device_type=server.type,
                mem=server.mem,
                util=util_tracker[server.name],
                components=components,
                task_info={
                    task.name: TaskInfo(
                        type=task.type,
                        total_requested_workload=task.peak_workload,
                        request_per_sec=allocated_demand
                    )
                }
            )
            
            temp_plan[(server.name, backbone)] = deployment
        
        return temp_plan, max(0, task_demand)
    
    def _sort_backbones_by_latency(
        self,
        pid_backbones: Dict[str, str],
        task: TaskSpec
    ) -> List[str]:
        """Sort backbones by latency (lowest first)."""
        heap = []
        for pid, backbone in pid_backbones.items():
            latency = self.data.get_pipeline_latency(pid, 'A16')
            if latency is not None:
                heapq.heappush(heap, (latency, backbone))
        return [backbone for _, backbone in heap]
    
    def _sort_backbones_by_accuracy(
        self,
        pid_backbones: Dict[str, str],
        task: TaskSpec
    ) -> List[str]:
        """Sort backbones by accuracy (best first)."""
        heap = []
        for pid, backbone in pid_backbones.items():
            metric = self.data.get_pipeline_metric(pid)
            if task.type == 'classification':
                heapq.heappush(heap, (-metric, backbone))  # Higher is better
            else:
                heapq.heappush(heap, (metric, backbone))   # Lower is better (MAE)
        return [backbone for _, backbone in heap]


def build_final_json(deployments: List[Deployment], pipelines: Dict) -> Dict:
    """Build final JSON output from deployments."""
    sites = {}
    
    for deployment in deployments:
        backbone = deployment.backbone
        decoders = []
        site_id = deployment.site_manager
        
        for comp in deployment.components:
            for v in pipelines.values():
                d_key = f"{v['decoder']}_{v['backbone']}_{v['task']}"
                if comp == d_key and v['backbone'] == backbone:
                    task_name = v['task']
                    if task_name in deployment.task_info:
                        decoders.append({
                            "task": task_name,
                            "type": deployment.task_info[task_name].type,
                            "path": f"{task_name}_{v['backbone']}_{v['decoder']}"
                        })
                        break
        
        task_info_dict = {}
        for task_name, info in deployment.task_info.items():
            task_info_dict[task_name] = {
                'type': info.type,
                'total_requested_workload': info.total_requested_workload,
                'request_per_sec': info.request_per_sec
            }
        
        deployment_entry = {
            "device": deployment.ip,
            "device_name": deployment.server_name,
            "device_type": deployment.device_type,
            "backbone": backbone,
            "decoders": decoders,
            "tasks": task_info_dict,
            "util": round(deployment.util, 6)
        }
        
        if site_id not in sites:
            sites[site_id] = {"id": site_id, "deployments": []}
        sites[site_id]["deployments"].append(deployment_entry)
    
    return {"sites": list(sites.values())}
