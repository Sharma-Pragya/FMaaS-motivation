"""Clipper-style deployment scheduler.

This module implements a Clipper-style scheduling algorithm where each task
gets its own best backbone (by latency or accuracy) with NO sharing between tasks.
Each task = 1 deployment (or multiple if replication needed for capacity).
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


class ClipperScheduler(BaseScheduler):
    """Clipper-style scheduler - each task gets its own best backbone.
    
    This scheduler deploys tasks to devices by:
    1. For each task, select the BEST backbone (by latency or accuracy)
    2. Deploy that task using ONLY that backbone (no sharing with other tasks)
    3. Each task gets its own deployment(s)
    
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
        
        # Deploy each task with its own best backbone
        for task_name, task_spec in sorted_tasks:
            task = self._create_task_spec(task_name, task_spec)
            temp_plan, demand_left = self._deploy_task(state, task, accuracy_mode)
            
            # Log admission failure if demand not fully satisfied
            if demand_left is not None and demand_left > self.config.demand_epsilon:
                logger.warning(
                    f"Clipper: task '{task_name}' has {demand_left:.4f} rps "
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
    
    def _deploy_task(
        self, 
        state: DeploymentState, 
        task: TaskSpec,
        accuracy_mode: bool = False
    ) -> Tuple[Optional[Dict], Optional[float]]:
        """Deploy a single task using its best backbone only.
        
        Clipper behavior: Each task gets its own best backbone.
        No sharing with other tasks - each task gets a SEPARATE deployment.
        """
        # Get best backbone for this task
        if accuracy_mode:
            all_backbones = self._sort_backbones_by_accuracy(
                self.data.get_backbones_for_task(task.name), task
            )
        else:
            all_backbones = self._sort_backbones_by_latency(
                self.data.get_backbones_for_task(task.name), task
            )
        
        if not all_backbones:
            logger.warning(f"Clipper: no candidate backbones for task '{task.name}'")
            return None, task.peak_workload
        
        # TODO: SLO enforcement — uncomment to filter backbones by latency/metric bound
        # all_backbones = [
        #     b for b in all_backbones
        #     if self._satisfies_slo(task, b)
        # ]
        # if not all_backbones:
        #     logger.warning(f"Clipper: no backbone meets SLO for task '{task.name}'")
        #     return None, task.peak_workload
        
        # Use ONLY the best backbone for this task
        best_backbone = all_backbones[0]
        
        # Clipper: use a unique backbone name per task to prevent sharing
        # This ensures each task gets its own deployment
        task_backbone = f"{best_backbone}__clipper__{task.name}"
        
        # Try to deploy on available servers using ONLY this backbone
        temp_plan = {}
        task_demand = task.peak_workload
        util_tracker = {}
        print(state.get_servers_by_free_capacity(self.data.get_component_mem(best_backbone), max_util=self.config.util_factor))
        for server in state.get_servers_by_free_capacity(self.data.get_component_mem(best_backbone), max_util=self.config.util_factor):
            if task_demand <= self.config.demand_epsilon:
                break
                
            if server.mem < self.data.get_component_mem(best_backbone):
                continue
            
            # Check if we can use this server (has capacity)
            if server.name not in util_tracker:
                util_tracker[server.name] = server.util
            
            total_util = util_tracker[server.name]
            left_cap = self.config.util_factor - total_util
            
            if left_cap <= 1e-6:
                continue
            
            # Get pipeline info using the real backbone name
            pid = self.data.find_pipeline_id(task.name, best_backbone)
            if not pid:
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
            
            # Create deployment with unique backbone name per task
            deployment = Deployment(
                server_name=server.name,
                backbone=task_backbone,  # Unique per task
                ip=server.ip,
                site_manager=server.site_manager,
                device_type=server.type,
                mem=server.mem,
                util=util_tracker[server.name],
                cuda=server.cuda,
                components=components,
                task_info={
                    task.name: TaskInfo(
                        type=task.type,
                        total_requested_workload=task.peak_workload,
                        request_per_sec=allocated_demand
                    )
                }
            )
            
            temp_plan[(server.name, task_backbone)] = deployment
        
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
    """Build final JSON output from deployments.

    The __clipper__<task> suffix used to prevent backbone sharing is
    stripped transparently by DeploymentState.to_plan_json().
    """
    state = DeploymentState([])
    for d in deployments:
        state._deployments[(d.server_name, d.backbone)] = d
    return state.to_plan_json(pipelines)
