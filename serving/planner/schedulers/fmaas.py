"""FMaaS deployment scheduler with shared backbone optimization.

This module implements the FMaaS scheduling algorithm that deploys tasks
to devices while maximizing backbone sharing across tasks.
"""

import heapq
import json
import logging
from typing import Dict, List, Optional, Tuple

from ..models import Server, Deployment, TaskInfo, TaskSpec
from ..data_loader import ProfileData
from ..state import DeploymentState
from ..config import SchedulerConfig, TaskConstraints, DEFAULT_CONFIG
from .base import BaseScheduler


logger = logging.getLogger(__name__)


class FMaaSScheduler(BaseScheduler):
    """FMaaS deployment scheduler with shared backbone optimization.
    
    This scheduler deploys tasks to devices by:
    1. Prioritizing high-workload tasks
    2. Maximizing backbone sharing when possible
    3. Attempting backbone downsizing (fit) when resources are constrained
    
    Attributes:
        data: ProfileData instance with component/pipeline information.
        config: SchedulerConfig with scheduling parameters.
    """
    
    def schedule(
        self, 
        devices: Dict[str, Dict], 
        tasks: Dict[str, Dict],
        share_mode: bool = False
    ) -> List[Deployment]:
        """Schedule tasks onto devices.
        
        Args:
            devices: Dictionary of device configurations.
            tasks: Dictionary of task specifications.
            share_mode: If True, prioritize using existing backbones.
            
        Returns:
            List of Deployment objects representing the deployment plan.
        """
        # Initialize state with servers
        servers = self._create_servers(devices)
        state = DeploymentState(servers)
        
        # Sort tasks by peak workload (highest first)
        sorted_tasks = self._sort_tasks_by_workload(tasks)
        
        # Deploy each task
        for task_name, task_spec in sorted_tasks:
            task = self._create_task_spec(task_name, task_spec)
            temp_plan, demand_left = self._deploy_task(state, task, share_mode)
            logger.info(f"Deployed task {task.name}, remaining demand: {demand_left}, temp plan: {temp_plan}, current deployments: {state.get_all_deployments()}")
            # Commit the plan for this task
            if temp_plan:
                logger.info(f"Committing deployment for task {task.name}: {temp_plan}")
                for deployment in temp_plan.values():
                    state.add_deployment(
                        deployment, 
                        self.config.base_port, 
                        self.config.port_increment
                    )
            logger.info(f"State after deploying task {task.name}: {state.get_all_deployments()}")
        
        logger.info(f"Final deployment count: {state.get_deployment_count()}")
        return state.get_all_deployments()
    
    def _sort_tasks_by_workload(
        self, 
        tasks: Dict[str, Dict]
    ) -> List[Tuple[str, Dict]]:
        """Sort tasks by peak workload, highest first.
        
        Args:
            tasks: Dictionary of task specifications.
            
        Returns:
            List of (task_name, task_spec) tuples sorted by workload.
        """
        return sorted(
            tasks.items(),
            key=lambda x: x[1].get('peak_workload', 0),
            reverse=True
        )
    
    def _create_task_spec(self, name: str, spec: Dict) -> TaskSpec:
        """Create TaskSpec from dictionary specification.
        
        Args:
            name: Task name.
            spec: Task specification dictionary.
            
        Returns:
            TaskSpec object.
        """
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
        share_mode: bool = False,
        do_fit: bool = True
    ) -> Tuple[Optional[Dict], Optional[float]]:
        """Deploy a single task.
        
        Args:
            state: Current deployment state.
            task: Task specification.
            share_mode: If True, prioritize existing backbones.
            do_fit: If True, attempt fit when resources constrained.
            
        Returns:
            Tuple of (deployment plan, remaining demand).
        """
        # Get backbones that can serve this task
        if not share_mode:
            pid_backbones = self.data.get_backbones_for_task(task.name)
        else:
            # In share mode, first try backbones already deployed
            all_backbones = self._sort_backbones_by_accuracy(
                self.data.get_backbones_for_task(task.name), 
                task
            )
            pid_backbones = {}
            for deployment in state.get_all_deployments():
                pid = self.data.find_pipeline_id(task.name, deployment.backbone)
                if pid:
                    pid_backbones[pid] = deployment.backbone
        
        # Sort backbones by accuracy
        sorted_backbones = self._sort_backbones_by_accuracy(pid_backbones, task)
        
        # Try deploying with each backbone - each attempt is independent
        # Only return if a backbone can satisfy FULL demand (like legacy)
        temp_plan: Dict = {}
        demand_left = task.peak_workload
        last_plan = None
        last_demand_left = task.peak_workload
        
        for backbone in sorted_backbones:
            # Each backbone attempt starts fresh (independent util tracking)
            backbone_plan, backbone_demand_left, _ = self._deploy_with_backbone(
                state, task, backbone, share_mode,
                existing_plan=None,  # Fresh plan for each backbone
                remaining_demand=None,  # Full demand for each backbone
                util_tracker=None  # Fresh util tracking
            )
            # Keep track of last attempt for fit() fallback
            if backbone_plan:
                last_plan = backbone_plan
                last_demand_left = backbone_demand_left
            
            if backbone_demand_left is not None and backbone_demand_left <= self.config.demand_epsilon:
                # This backbone can satisfy the full demand
                return backbone_plan, backbone_demand_left
        
        # No backbone satisfied full demand - use last result for fit() fallback
        temp_plan = last_plan if last_plan is not None else {}
        demand_left = last_demand_left
        
        # In share mode, try all backbones if existing ones don't satisfy demand
        if share_mode:
            # Legacy behavior: 
            # 1. First collect ALL active deployments from sorted_backbones
            # 2. Then iterate through all backbones and servers, appending to the same list
            # 3. distribute_demand consumes from this list via heappop
            
            # Step 1: Collect all active deployments from sorted_backbones
            active_endpoints = []
            for backbone in sorted_backbones:
                for d in state.find_active_deployments(backbone, self.config.util_factor):
                    active_endpoints.append((d.server_name, backbone))
            
            # Step 2: Iterate through all backbones and servers, appending and distributing
            for backbone in all_backbones:
                for server in state.get_all_servers():
                    if server.mem >= self.data.get_component_mem(backbone):
                        active_endpoints.append((server.name, backbone))
                        temp_plan, demand_left = self._distribute_demand(
                            state, task, active_endpoints,
                            remaining_demand=None,  # Start fresh with full demand
                            existing_plan=None,  # Fresh plan
                            util_tracker=None  # Fresh util tracker
                        )
                        if demand_left is not None and demand_left <= self.config.demand_epsilon:
                            return temp_plan, demand_left
        
        # Attempt fit if enabled and demand not satisfied
        if do_fit and (demand_left is None or demand_left > self.config.demand_epsilon):
            logger.info(f"Running fit for task {task.name}")
            fit_plan, fit_demand = self._fit(state, task, share_mode)
            # Use fit result if it improved over the pre-fit result
            pre_fit_demand = demand_left if demand_left is not None else task.peak_workload
            if fit_demand is not None and fit_demand < pre_fit_demand:
                return fit_plan, fit_demand

        return temp_plan, demand_left
    
    def _deploy_with_backbone(
        self,
        state: DeploymentState,
        task: TaskSpec,
        backbone: str,
        share_mode: bool,
        existing_plan: Optional[Dict] = None,
        remaining_demand: Optional[float] = None,
        util_tracker: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[Dict], Optional[float], Dict[str, float]]:
        """Deploy task using a specific backbone.
        
        Args:
            state: Current deployment state.
            task: Task specification.
            backbone: Backbone name to use.
            share_mode: If True, only use existing deployments.
            existing_plan: Existing deployment plan to extend.
            remaining_demand: Remaining demand to allocate.
            util_tracker: Utilization tracker across calls.
            
        Returns:
            Tuple of (deployment plan, remaining demand, util_tracker).
        """
        temp_plan = existing_plan if existing_plan is not None else {}
        demand_left = remaining_demand if remaining_demand is not None else task.peak_workload
        if util_tracker is None:
            util_tracker = {}
        
        # Find active deployments with this backbone
        active_endpoints = [
            (d.server_name, backbone)
            for d in state.find_active_deployments(backbone, self.config.util_factor)
        ]
        
        temp_plan, demand_left = self._distribute_demand(
            state, task, active_endpoints,
            remaining_demand=demand_left,
            existing_plan=temp_plan,
            util_tracker=util_tracker
        )
        
        if demand_left is not None and demand_left <= self.config.demand_epsilon:
            return temp_plan, demand_left, util_tracker
        
        # If not in share mode, try adding new servers
        if not share_mode:
            backbone_mem = self.data.get_component_mem(backbone)
            for server in state.get_servers_with_memory(backbone_mem):
                # Skip if already in endpoints
                if (server.name, backbone) in temp_plan:
                    continue
                active_endpoints = [(server.name, backbone)]
                temp_plan, demand_left = self._distribute_demand(
                    state, task, active_endpoints,
                    remaining_demand=demand_left,
                    existing_plan=temp_plan,
                    util_tracker=util_tracker
                )
                if demand_left is not None and demand_left <= self.config.demand_epsilon:
                    return temp_plan, demand_left, util_tracker
        
        return temp_plan, demand_left, util_tracker
    
    def _distribute_demand(
        self,
        state: DeploymentState,
        task: TaskSpec,
        endpoints: List[Tuple[str, str]],
        remaining_demand: Optional[float] = None,
        existing_plan: Optional[Dict] = None,
        util_tracker: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict, float]:
        """Distribute task demand across endpoints.
        
        Args:
            state: Current deployment state.
            task: Task specification.
            endpoints: List of (server_name, backbone) tuples.
            remaining_demand: Remaining demand to allocate. If None, uses task.peak_workload.
            existing_plan: Existing deployment plan to extend. If None, creates new.
            util_tracker: Existing utilization tracker. If None, creates new.
            
        Returns:
            Tuple of (temporary deployment plan, remaining demand).
        """
        task_demand = remaining_demand if remaining_demand is not None else task.peak_workload
        temp_plan = existing_plan if existing_plan is not None else {}
        if util_tracker is None:
            util_tracker = {}  # server_name -> util
        
        # Build heap of endpoints sorted by priority
        heap = []
        for server_name, backbone in endpoints:
            pid = self.data.find_pipeline_id(task.name, backbone)
            if not pid:
                continue
            
            # Check constraints
            pipeline = self.data.get_pipeline(pid)
            server = state.get_server(server_name)
            if not server:
                continue
            
            latency = self.data.get_pipeline_latency(pid, server.type)
            if latency is None:
                continue
            
            metric = self.data.get_pipeline_metric(pid)
            constraints = TaskConstraints(
                metric_type=task.metric,
                metric_threshold=task.value,
                max_latency=task.latency
            )
            
            # if not constraints.satisfies_metric(metric):
            #     continue
            # if not constraints.satisfies_latency(latency):
            #     continue
            
            # Track utilization
            if server_name not in util_tracker:
                util_tracker[server_name] = server.util
            
            # Priority: higher accuracy = lower priority value for classification
            priority = -metric if task.type == 'classification' else metric
            heapq.heappush(heap, (priority, server_name, backbone, pid))
        
        # Distribute demand
        while task_demand > self.config.demand_epsilon and heap:
            priority, server_name, backbone, pid = heapq.heappop(heap)
            server = state.get_server(server_name)
            
            total_util = util_tracker[server_name]
            left_cap = self.config.util_factor - total_util
            
            if left_cap > 1e-6:
                latency = self.data.get_pipeline_latency(pid, server.type)
                task_cap_needed = task_demand * latency / 1000.0
                allocated_cap = min(left_cap, task_cap_needed)
                allocated_demand = allocated_cap * 1000.0 / latency
                logger.debug(f"Task {task.name} on {server_name}/{backbone}: "
                           f"latency={latency}ms, left_cap={left_cap:.4f}, "
                           f"allocated_cap={allocated_cap:.4f}, allocated_demand={allocated_demand:.2f} req/s, "
                           f"remaining_demand={max(0, task_demand - allocated_demand):.2f}")                
                task_demand -= allocated_demand
                
                # Get pipeline components
                pipeline = self.data.get_pipeline(pid)
                components = self.data.get_pipeline_components_mem(pipeline)
                
                # Don't allocate port here - do it during commit to avoid
                # consuming ports for temp plans that get rolled back.
                # Use server IP as placeholder; actual port allocated in add_deployment
                placeholder_ip = server.ip
                
                # Create deployment
                deployment = Deployment(
                    server_name=server_name,
                    backbone=backbone,
                    ip=placeholder_ip,
                    site_manager=server.site_manager,
                    device_type=server.type,
                    mem=server.mem,
                    util=total_util + allocated_cap,
                    components=components,
                    task_info={
                        task.name: TaskInfo(
                            type=task.type,
                            total_requested_workload=task.peak_workload,
                            request_per_sec=allocated_demand
                        )
                    }
                )
                
                temp_plan[(server_name, backbone)] = deployment
                util_tracker[server_name] += allocated_cap
        
        # Don't commit here - let the caller commit once per task
        return temp_plan, max(0, task_demand)
    
    def _sort_backbones_by_accuracy(
        self,
        pid_backbones: Dict[str, str],
        task: TaskSpec
    ) -> List[str]:
        """Sort backbones by accuracy metric.
        
        Args:
            pid_backbones: Mapping of pipeline IDs to backbone names.
            task: Task specification.
            
        Returns:
            List of backbone names sorted by accuracy.
        """
        heap = []
        for pid, backbone in pid_backbones.items():
            metric = self.data.get_pipeline_metric(pid)
            if task.type == 'classification':
                heapq.heappush(heap, (-metric, backbone))
            else:
                heapq.heappush(heap, (metric, backbone))
        
        return [backbone for _, backbone in heap]
    
    def _fit(
        self,
        state: DeploymentState,
        task: TaskSpec,
        share_mode: bool
    ) -> Tuple[Optional[Dict], Optional[float]]:
        """Attempt to fit task by downsizing backbones cumulatively.

        Tries downsizing multiple backbones in sequence. Each downsize
        frees capacity, and the cumulative effect is checked after each one.
        If fully satisfied at any point, returns immediately. Otherwise,
        keeps changes if they improved demand, or rolls back everything.

        Args:
            state: Current deployment state.
            task: Task specification.
            share_mode: Share mode flag.

        Returns:
            Tuple of (deployment plan, remaining demand).
        """
        logger.info(f"Attempting fit for task {task.name}")
        demand_left = task.peak_workload
        temp_plan = None

        # Track all changes for potential rollback
        # Each entry: (old_deployment_copy, existing_copy_or_None, new_backbone)
        rollback_info = []

        # Get deployments sorted by number of tasks (ascending)
        deployments = sorted(
            state.get_all_deployments(),
            key=lambda d: len(d.task_info)
        )

        for deployment in deployments:
            # Find smaller backbone
            new_backbone = self.data.find_smaller_backbone(deployment.backbone)
            if not new_backbone:
                continue

            # Check if this deployment still exists in state — a previous
            # iteration's update_deployment_backbone may have removed it
            # (e.g., if it was merged into another deployment's key).
            current = state.get_deployment(deployment.server_name, deployment.backbone)
            if not current:
                continue

            logger.info(f"Trying to fit by replacing backbone {current.backbone} of deployment {current} with {new_backbone} on server {current.server_name}")

            # Save current state for potential rollback (use current from
            # state, not snapshot, since previous iterations may have
            # mutated it via util sync or merge).
            old_deployment_copy = Deployment(
                server_name=current.server_name,
                backbone=current.backbone,
                ip=current.ip,
                site_manager=current.site_manager,
                device_type=current.device_type,
                mem=current.mem,
                util=current.util,
                components=dict(current.components),
                task_info={k: TaskInfo(
                    type=v.type,
                    total_requested_workload=v.total_requested_workload,
                    request_per_sec=v.request_per_sec
                ) for k, v in current.task_info.items()}
            )

            # Save existing deployment at (server_name, new_backbone) if any,
            # since update_deployment_backbone may merge into it and we need
            # to restore it on rollback.
            existing_at_new_key = state.get_deployment(current.server_name, new_backbone)
            existing_copy = None
            if existing_at_new_key:
                existing_copy = Deployment(
                    server_name=existing_at_new_key.server_name,
                    backbone=existing_at_new_key.backbone,
                    ip=existing_at_new_key.ip,
                    site_manager=existing_at_new_key.site_manager,
                    device_type=existing_at_new_key.device_type,
                    mem=existing_at_new_key.mem,
                    util=existing_at_new_key.util,
                    components=dict(existing_at_new_key.components),
                    task_info={k: TaskInfo(
                        type=v.type,
                        total_requested_workload=v.total_requested_workload,
                        request_per_sec=v.request_per_sec
                    ) for k, v in existing_at_new_key.task_info.items()}
                )

            rollback_info.append((old_deployment_copy, existing_copy, new_backbone))

            # Calculate new components and utilization
            new_components = {new_backbone: self.data.get_component_mem(new_backbone)}
            new_util = 0.0
            old_util = current.util

            for t_name, t_info in current.task_info.items():
                # Find pipeline for new backbone
                new_pid = self.data.find_pipeline_id(t_name, new_backbone)
                if new_pid:
                    pipeline = self.data.get_pipeline(new_pid)
                    comp_mem = self.data.get_pipeline_components_mem(pipeline)
                    for k, v in comp_mem.items():
                        if k != new_backbone:
                            new_components[k] = v

                    latency = self.data.get_pipeline_latency(new_pid, current.device_type)
                    if latency:
                        new_util += (t_info.request_per_sec * latency) / 1000.0

                # Subtract old backbone contribution
                old_pid = self.data.find_pipeline_id(t_name, current.backbone)
                if old_pid:
                    old_latency = self.data.get_pipeline_latency(old_pid, current.device_type)
                    if old_latency:
                        old_util -= (t_info.request_per_sec * old_latency) / 1000.0

            # Update deployment backbone (cumulative — don't rollback yet)
            state.update_deployment_backbone(
                current.server_name,
                current.backbone,
                new_backbone,
                new_components,
                old_util + new_util
            )
            logger.info(f"Current deployments after backbone change: {state.get_all_deployments()}")

            # Try to deploy task with cumulative changes
            temp_plan, demand_left = self._deploy_task(
                state, task, share_mode, do_fit=False
            )
            logger.info(f"Post-fit deployment attempt for task {task.name}, remaining demand: {demand_left}, temp plan: {temp_plan}, current deployments: {state.get_all_deployments()}")

            if demand_left is not None and demand_left <= self.config.demand_epsilon:
                # Fully satisfied — keep all backbone changes
                logger.info(f"Fit fully succeeded with: {state.get_all_deployments()}")
                return temp_plan, demand_left

            # Not fully satisfied — continue trying more backbone downsizes

        # After trying all backbone downsizes: check if enough demand was satisfied
        # to justify keeping the backbone changes.
        satisfied_fraction = 1.0 - (demand_left / task.peak_workload) if task.peak_workload > 0 else 0.0
        if satisfied_fraction >= self.config.fit_keep_threshold and demand_left < task.peak_workload:
            logger.info(f"Fit partially helped: demand reduced from {task.peak_workload} to {demand_left} "
                        f"({satisfied_fraction:.1%} satisfied, threshold={self.config.fit_keep_threshold:.1%}), keeping backbone changes")
            return temp_plan, demand_left
        else:
            # Fit didn't satisfy enough demand — rollback ALL changes in reverse order
            logger.info(f"Fit insufficient: {satisfied_fraction:.1%} satisfied (threshold={self.config.fit_keep_threshold:.1%}), rolling back all backbone changes")
            for old_copy, existing_copy, new_bb in reversed(rollback_info):
                state.remove_deployment(old_copy.server_name, new_bb)
                if existing_copy:
                    state.add_deployment(
                        existing_copy,
                        self.config.base_port,
                        self.config.port_increment
                    )
                state.add_deployment(
                    old_copy,
                    self.config.base_port,
                    self.config.port_increment
                )
            return temp_plan, demand_left


def build_final_json(deployments: List[Deployment], pipelines: Dict) -> Dict:
    """Build final JSON output from deployments."""
    state = DeploymentState([])
    for d in deployments:
        state._deployments[(d.server_name, d.backbone)] = d
    return state.to_plan_json(pipelines)


