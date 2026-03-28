"""FMaaS placement scheduler with user-specified backbones.

This module implements a simplified FMaaS scheduling algorithm where the
user specifies which backbone to use per task in their config. The scheduler's
only job is placement and sharing: tasks that specify the same backbone are
co-located on the same deployment; tasks with different backbones get separate
deployments.

No model selection, no backbone downsizing (fit), no migrate diffs.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..models import Deployment, TaskInfo, TaskSpec
from ..state import DeploymentState
from ..config import DEFAULT_CONFIG, SchedulerConfig
from .base import BaseScheduler


logger = logging.getLogger(__name__)


class FMaaSPlacementScheduler(BaseScheduler):
    """FMaaS scheduler where users specify the backbone per task.

    Tasks sharing the same backbone are co-located on the same deployment
    when capacity allows. No model selection is performed — backbone choice
    is entirely up to the user.

    Attributes:
        data: ProfileData instance with component/pipeline information.
        config: SchedulerConfig with scheduling parameters.
    """

    def schedule(
        self,
        devices: Dict[str, Dict],
        tasks: Dict[str, Dict],
    ) -> List[Deployment]:
        """Schedule tasks onto devices.

        Each task spec must include a 'backbone' key. Tasks that share a
        backbone will be co-located when possible (sharing). Tasks with
        different backbones always get separate deployments.

        Args:
            devices: Dictionary of device configurations.
            tasks: Dictionary of task specifications. Each task must include:
                   - backbone: backbone model name to use (required)
                   - type: 'classification' or 'regression'
                   - peak_workload: maximum requests per second
                   - latency: maximum acceptable latency (optional)
                   - metric: 'accuracy' or 'mae' (optional)
                   - value: required metric threshold (optional)

        Returns:
            List of Deployment objects representing the deployment plan.
        """
        servers = self._create_servers(devices)
        state = DeploymentState(servers)

        # Sort tasks by peak workload (highest first) so high-demand tasks
        # claim capacity before lower-demand ones.
        sorted_tasks = sorted(
            tasks.items(),
            key=lambda x: x[1].get('peak_workload', 0),
            reverse=True,
        )

        for task_name, task_spec in sorted_tasks:
            task = self._create_task_spec(task_name, task_spec)
            if not task.backbone:
                raise ValueError(
                    f"Task '{task_name}' has no 'backbone' specified. "
                    f"fmaas_place requires a backbone per task in user_config."
                )
            temp_plan, demand_left = self._deploy_task(state, task)

            if demand_left is not None and demand_left > self.config.demand_epsilon:
                logger.warning(
                    f"FMaaSPlacement: task '{task_name}' has {demand_left:.4f} rps "
                    f"unsatisfied demand out of {task.peak_workload:.4f} rps"
                )

            if temp_plan:
                for deployment in temp_plan.values():
                    state.add_deployment(
                        deployment,
                        self.config.base_port,
                        self.config.port_increment,
                    )

        logger.info(f"Final deployment count: {state.get_deployment_count()}")
        return state.get_all_deployments()

    def _create_task_spec(self, name: str, spec: Dict) -> TaskSpec:
        """Create TaskSpec from dictionary specification."""
        return TaskSpec(
            name=name,
            type=spec['type'],
            peak_workload=spec['peak_workload'],
            latency=spec.get('latency', float('inf')),
            metric=spec.get('metric', 'mae'),
            value=spec.get('value', 0),
            backbone=spec.get('backbone', None),
        )

    def _deploy_task(
        self,
        state: DeploymentState,
        task: TaskSpec,
        share_mode: bool = True,  # accepted for interface compatibility, ignored
    ) -> Tuple[Optional[Dict], float]:
        """Deploy a single task using its user-specified backbone.

        Phase 1: Exhaust capacity on existing deployments that already have
                 task.backbone loaded (sharing opportunity).
        Phase 2: If demand remains, place on new servers with task.backbone.

        Args:
            state: Current deployment state.
            task: TaskSpec with backbone set.

        Returns:
            Tuple of (deployment plan dict, remaining demand).
        """
        backbone = task.backbone
        util_tracker: Dict[str, float] = {}
        temp_plan: Dict = {}
        demand_left = task.peak_workload

        # Phase 1: share with existing deployments that already use this backbone.
        active_endpoints = [
            (d.server_name, backbone)
            for d in state.find_active_deployments(backbone, self.config.util_factor)
        ]
        if active_endpoints:
            temp_plan, demand_left = self._distribute_demand(
                state, task, active_endpoints,
                remaining_demand=demand_left,
                existing_plan=temp_plan,
                util_tracker=util_tracker,
            )
            if demand_left <= self.config.demand_epsilon:
                return temp_plan, demand_left

        # Phase 2: place on new servers with enough free capacity.
        backbone_mem = self.data.get_component_mem(backbone)
        for server in state.get_servers_by_least_capacity(backbone_mem, max_util=self.config.util_factor):
            if (server.name, backbone) in temp_plan:
                continue  # already allocated in Phase 1
            temp_plan, demand_left = self._distribute_demand(
                state, task, [(server.name, backbone)],
                remaining_demand=demand_left,
                existing_plan=temp_plan,
                util_tracker=util_tracker,
            )
            if demand_left <= self.config.demand_epsilon:
                return temp_plan, demand_left

        return temp_plan, demand_left

    def _distribute_demand(
        self,
        state: DeploymentState,
        task: TaskSpec,
        endpoints: List[Tuple[str, str]],
        remaining_demand: Optional[float] = None,
        existing_plan: Optional[Dict] = None,
        util_tracker: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict, float]:
        """Distribute task demand across endpoints.

        Same allocation logic as FMaaSScheduler._distribute_demand: allocates
        demand proportionally based on available utilization capacity and
        pipeline latency.

        Args:
            state: Current deployment state.
            task: Task specification.
            endpoints: List of (server_name, backbone) tuples.
            remaining_demand: Demand left to allocate. Defaults to task.peak_workload.
            existing_plan: Deployment plan to extend. Defaults to empty dict.
            util_tracker: Utilization tracker across calls. Defaults to empty dict.

        Returns:
            Tuple of (updated deployment plan, remaining demand).
        """
        task_demand = remaining_demand if remaining_demand is not None else task.peak_workload
        temp_plan = existing_plan if existing_plan is not None else {}
        if util_tracker is None:
            util_tracker = {}

        for server_name, backbone in endpoints:
            if task_demand <= self.config.demand_epsilon:
                break

            pid = self.data.find_pipeline_id(task.name, backbone)
            if not pid:
                logger.warning(f"No pipeline found for task '{task.name}' with backbone '{backbone}'")
                continue

            server = state.get_server(server_name)
            if not server:
                continue

            latency = self.data.get_pipeline_latency(pid, server.type)
            if latency is None:
                logger.warning(f"No latency data for pipeline '{pid}' on device type '{server.type}'")
                continue

            if server_name not in util_tracker:
                util_tracker[server_name] = server.util

            total_util = util_tracker[server_name]
            left_cap = self.config.util_factor - total_util

            if left_cap <= 1e-6:
                continue

            task_cap_needed = task_demand * latency / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * 1000.0 / latency

            logger.debug(
                f"Task '{task.name}' on {server_name}/{backbone}: "
                f"latency={latency}ms, left_cap={left_cap:.4f}, "
                f"allocated_cap={allocated_cap:.4f}, allocated_demand={allocated_demand:.2f} req/s, "
                f"remaining_demand={max(0, task_demand - allocated_demand):.2f}"
            )

            task_demand -= allocated_demand
            util_tracker[server_name] += allocated_cap

            pipeline = self.data.get_pipeline(pid)
            components = self.data.get_pipeline_components_mem(pipeline)

            deployment = Deployment(
                server_name=server_name,
                backbone=backbone,
                ip=server.ip,
                site_manager=server.site_manager,
                device_type=server.type,
                mem=server.mem,
                util=util_tracker[server_name],
                cuda=server.cuda,
                components=components,
                task_info={
                    task.name: TaskInfo(
                        type=task.type,
                        total_requested_workload=task.peak_workload,
                        request_per_sec=allocated_demand,
                    )
                },
            )

            temp_plan[(server_name, backbone)] = deployment

        return temp_plan, max(0, task_demand)


def build_final_json(deployments: List[Deployment], pipelines: Dict) -> Dict:
    """Build final JSON output from deployments."""
    state = DeploymentState([])
    for d in deployments:
        state._deployments[(d.server_name, d.backbone)] = d
    return state.to_plan_json(pipelines)
