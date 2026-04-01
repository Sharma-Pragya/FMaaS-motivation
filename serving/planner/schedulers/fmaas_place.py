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
from ..data_loader import BatchProfile, ProfileData
from .base import BaseScheduler


logger = logging.getLogger(__name__)


class FMaaSPlacementScheduler(BaseScheduler):
    """FMaaS scheduler where users specify the backbone per task.

    Tasks sharing the same backbone are co-located on the same deployment
    when capacity allows. No model selection is performed — backbone choice
    is entirely up to the user.

    When a BatchProfile is provided, the scheduler accounts for cross-task
    batching: tasks sharing a backbone amortize backbone cost across a
    selected batch size, reducing per-request utilization.

    Attributes:
        data: ProfileData instance with component/pipeline information.
        config: SchedulerConfig with scheduling parameters.
        batch_profile: Optional BatchProfile for batching-aware scheduling.
    """

    def __init__(
        self,
        profile_data: ProfileData,
        config: Optional[SchedulerConfig] = None,
        batch_profile: Optional[BatchProfile] = None,
    ):
        super().__init__(profile_data, config)
        self.batch_profile = batch_profile
        # Maps (server_name, backbone) -> saturation batch size (for device max_batch_size)
        self.batch_size_map: Dict[Tuple[str, str], int] = {}
        # Maps (server_name, backbone) -> selected batch size (expected utilization)
        self.expected_batch_size_map: Dict[Tuple[str, str], int] = {}

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
                    key = (deployment.server_name, deployment.backbone)
                    existing = state.get_deployment(deployment.server_name, deployment.backbone)
                    if existing and self.batch_profile is not None:
                        # Batching logic builds fully merged deployments —
                        # replace directly to avoid double-counting in add_deployment's merge.
                        if ':' not in deployment.ip:
                            port = state.get_next_port(
                                deployment.ip, self.config.base_port, self.config.port_increment,
                            )
                            deployment.ip = f"{deployment.ip}:{port}"
                        state._deployments[key] = deployment
                        state._sync_server_utilization(deployment.server_name, deployment.util)
                    else:
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
        # demand_tracker: accumulates total demand per (server, backbone) across
        # all tasks placed so far (including from state and this scheduling round).
        # Seeded from already-committed deployments in state.
        base_temp_plan: Dict = {}
        base_util_tracker: Dict[str, float] = {}
        base_demand_tracker: Dict[Tuple[str, str], Dict[str, float]] = {}
        # deploy_util_tracker: tracks utilization contributed by each (server, backbone)
        base_deploy_util_tracker: Dict[Tuple[str, str], float] = {}
        for d in state.get_all_deployments():
            key = (d.server_name, d.backbone)
            base_demand_tracker[key] = {
                t_name: t_info.request_per_sec
                for t_name, t_info in d.task_info.items()
            }
            # The util on the deployment includes all deployments on that server,
            # so we estimate this deployment's share from its task demands
            # (will be recomputed accurately when we touch it)
            base_deploy_util_tracker[key] = d.util  # initial estimate

        active_endpoints = [
            (d.server_name, backbone)
            for d in state.find_active_deployments(backbone, self.config.util_factor)
        ]
        backbone_mem = self.data.get_component_mem(backbone)
        new_endpoints = [
            (server.name, backbone)
            for server in state.get_servers_by_least_capacity(
                backbone_mem, max_util=self.config.util_factor,
            )
            if (server.name, backbone) not in active_endpoints
        ]

        # Prefer the smallest candidate pool that can satisfy the full task:
        # 1) one active endpoint, 2) active endpoints together, then 3) add one
        # new endpoint at a time and retry those same two checks.
        candidate_pool = list(active_endpoints)
        candidate_demand_left = task.peak_workload
        candidate_plan: Dict = {}

        while True:
            single_plan, single_demand_left = self._try_full_fit_on_single_endpoint(
                state, task, candidate_pool,
                existing_plan=base_temp_plan,
                util_tracker=base_util_tracker,
                demand_tracker=base_demand_tracker,
                deploy_util_tracker=base_deploy_util_tracker,
            )
            if single_demand_left <= self.config.demand_epsilon:
                return single_plan, single_demand_left

            if candidate_pool:
                (
                    candidate_plan,
                    candidate_demand_left,
                    candidate_batch_size_map,
                    candidate_expected_batch_size_map,
                ) = self._run_isolated_distribution(
                    state, task, candidate_pool,
                    remaining_demand=task.peak_workload,
                    existing_plan=base_temp_plan,
                    util_tracker=base_util_tracker,
                    demand_tracker=base_demand_tracker,
                    deploy_util_tracker=base_deploy_util_tracker,
                )
                if candidate_demand_left <= self.config.demand_epsilon:
                    self.batch_size_map = candidate_batch_size_map
                    self.expected_batch_size_map = candidate_expected_batch_size_map
                    return candidate_plan, candidate_demand_left

            if not new_endpoints:
                return candidate_plan, candidate_demand_left

            candidate_pool.append(new_endpoints.pop(0))

    def _try_full_fit_on_single_endpoint(
        self,
        state: DeploymentState,
        task: TaskSpec,
        endpoints: List[Tuple[str, str]],
        existing_plan: Dict,
        util_tracker: Dict[str, float],
        demand_tracker: Dict[Tuple[str, str], Dict[str, float]],
        deploy_util_tracker: Dict[Tuple[str, str], float],
    ) -> Tuple[Dict, float]:
        """Try each endpoint in isolation and return the first full-fit plan."""
        for endpoint in endpoints:
            solo_plan, solo_demand_left, batch_size_map, expected_batch_size_map = self._run_isolated_distribution(
                state, task, [endpoint],
                remaining_demand=task.peak_workload,
                existing_plan=existing_plan,
                util_tracker=util_tracker,
                demand_tracker=demand_tracker,
                deploy_util_tracker=deploy_util_tracker,
            )
            if solo_demand_left <= self.config.demand_epsilon:
                self.batch_size_map = batch_size_map
                self.expected_batch_size_map = expected_batch_size_map
                return solo_plan, solo_demand_left

        return dict(existing_plan), task.peak_workload

    def _run_isolated_distribution(
        self,
        state: DeploymentState,
        task: TaskSpec,
        endpoints: List[Tuple[str, str]],
        remaining_demand: float,
        existing_plan: Dict,
        util_tracker: Dict[str, float],
        demand_tracker: Dict[Tuple[str, str], Dict[str, float]],
        deploy_util_tracker: Dict[Tuple[str, str], float],
    ) -> Tuple[Dict, float, Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
        """Run distribution on a trial copy so placement policy stays explicit."""
        saved_batch_size_map = dict(self.batch_size_map)
        saved_expected_batch_size_map = dict(self.expected_batch_size_map)

        trial_plan, trial_demand_left = self._distribute_demand(
            state, task, endpoints,
            remaining_demand=remaining_demand,
            existing_plan=dict(existing_plan),
            util_tracker=dict(util_tracker),
            demand_tracker={
                key: dict(task_demands)
                for key, task_demands in demand_tracker.items()
            },
            deploy_util_tracker=dict(deploy_util_tracker),
        )

        trial_batch_size_map = dict(self.batch_size_map)
        trial_expected_batch_size_map = dict(self.expected_batch_size_map)

        self.batch_size_map = saved_batch_size_map
        self.expected_batch_size_map = saved_expected_batch_size_map

        return (
            trial_plan,
            trial_demand_left,
            trial_batch_size_map,
            trial_expected_batch_size_map,
        )

    def _distribute_demand(
        self,
        state: DeploymentState,
        task: TaskSpec,
        endpoints: List[Tuple[str, str]],
        remaining_demand: Optional[float] = None,
        existing_plan: Optional[Dict] = None,
        util_tracker: Optional[Dict[str, float]] = None,
        demand_tracker: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
        deploy_util_tracker: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[Dict, float]:
        """Distribute task demand across endpoints with batching awareness.

        When batch_profile is set and a deployment already has tasks, this
        method recomputes the batch size using aggregate demand across all
        tasks on the deployment. The larger batch size reduces per-request
        utilization for ALL tasks, freeing capacity for the new task.

        Args:
            state: Current deployment state.
            task: Task specification.
            endpoints: List of (server_name, backbone) tuples.
            remaining_demand: Demand left to allocate. Defaults to task.peak_workload.
            existing_plan: Deployment plan to extend. Defaults to empty dict.
            util_tracker: Utilization tracker across calls. Defaults to empty dict.
            demand_tracker: Per-deployment per-task demand tracker. Defaults to empty dict.
            deploy_util_tracker: Per-deployment utilization tracker. Defaults to empty dict.

        Returns:
            Tuple of (updated deployment plan, remaining demand).
        """
        task_demand = remaining_demand if remaining_demand is not None else task.peak_workload
        temp_plan = existing_plan if existing_plan is not None else {}
        if util_tracker is None:
            util_tracker = {}
        if demand_tracker is None:
            demand_tracker = {}
        if deploy_util_tracker is None:
            deploy_util_tracker = {}

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

            latency_bs1 = self.data.get_pipeline_latency(pid, server.type)
            if latency_bs1 is None:
                logger.warning(f"No latency data for pipeline '{pid}' on device type '{server.type}'")
                continue

            if server_name not in util_tracker:
                util_tracker[server_name] = server.util

            deploy_key = (server_name, backbone)

            # Get existing demand on this deployment
            existing_demands = demand_tracker.get(deploy_key, {})
            existing_total_demand = sum(existing_demands.values())

            # Compute aggregate demand (existing + new task)
            aggregate_demand = existing_total_demand + task_demand

            # Compute batch size and latency using aggregate demand
            selected_bs = 1
            sat_bs = 1
            latency = latency_bs1
            if self.batch_profile is not None:
                try:
                    latency, selected_bs, sat_bs = self._compute_batched_latency(
                        backbone, server.type, aggregate_demand, latency_bs1,
                    )
                except KeyError:
                    logger.debug(
                        f"No batch profile for {backbone}/{server.type}, "
                        f"falling back to bs=1"
                    )

            # Recompute utilization for existing tasks at the new batch size/latency
            existing_util_new = (existing_total_demand / selected_bs) * latency / 1000.0

            # Base utilization = server util WITHOUT this deployment's old contribution
            old_deploy_util = deploy_util_tracker.get(deploy_key, 0.0)
            base_util = max(0.0, util_tracker[server_name] - old_deploy_util)

            left_cap = self.config.util_factor - base_util - existing_util_new
            if left_cap <= 1e-6:
                continue

            # Allocate new task
            new_task_cap_needed = (task_demand / selected_bs) * latency / 1000.0
            allocated_cap = min(left_cap, new_task_cap_needed)
            allocated_demand = allocated_cap * selected_bs * 1000.0 / latency

            # Total utilization for this server
            total_server_util = base_util + existing_util_new + allocated_cap

            print(
                f"[Alloc] Task '{task.name}' on {server_name}/{backbone}: "
                f"latency={latency:.2f}ms, batch_size={selected_bs}, "
                f"aggregate_demand={aggregate_demand:.2f}, "
                f"existing_util_new={existing_util_new:.6f}, "
                f"new_task_cap={new_task_cap_needed:.6f}, left_cap={left_cap:.4f}, "
                f"allocated_cap={allocated_cap:.6f}, allocated_demand={allocated_demand:.2f} req/s, "
                f"total_util={total_server_util:.6f}, "
                f"remaining_demand={max(0, task_demand - allocated_demand):.2f}"
            )

            task_demand -= allocated_demand
            util_tracker[server_name] = total_server_util

            # Update demand tracker
            if deploy_key not in demand_tracker:
                demand_tracker[deploy_key] = {}
            demand_tracker[deploy_key][task.name] = allocated_demand

            # Update deploy util tracker with new total for this deployment
            deploy_util_tracker[deploy_key] = existing_util_new + allocated_cap

            # Track batch sizes per deployment
            self.batch_size_map[deploy_key] = sat_bs
            self.expected_batch_size_map[deploy_key] = selected_bs

            # Build deployment with ALL tasks (existing + new)
            pipeline = self.data.get_pipeline(pid)
            components = self.data.get_pipeline_components_mem(pipeline)

            # Merge components from existing deployment if present
            existing_deployment = temp_plan.get(deploy_key) or state.get_deployment(server_name, backbone)
            if existing_deployment:
                components = dict(existing_deployment.components)
                components.update(self.data.get_pipeline_components_mem(pipeline))

            # Build merged task_info: existing tasks with recalculated demand + new task
            task_info = {}
            for t_name, t_demand in demand_tracker[deploy_key].items():
                if t_name == task.name:
                    task_info[t_name] = TaskInfo(
                        type=task.type,
                        total_requested_workload=task.peak_workload,
                        request_per_sec=t_demand,
                    )
                elif existing_deployment and t_name in existing_deployment.task_info:
                    old_info = existing_deployment.task_info[t_name]
                    task_info[t_name] = TaskInfo(
                        type=old_info.type,
                        total_requested_workload=old_info.total_requested_workload,
                        request_per_sec=t_demand,
                    )

            deployment = Deployment(
                server_name=server_name,
                backbone=backbone,
                ip=existing_deployment.ip if existing_deployment else server.ip,
                site_manager=server.site_manager,
                device_type=server.type,
                mem=server.mem,
                util=total_server_util,
                cuda=server.cuda,
                components=components,
                task_info=task_info,
            )

            temp_plan[deploy_key] = deployment

        return temp_plan, max(0, task_demand)

    def _compute_batched_latency(
        self,
        backbone: str,
        device_type: str,
        task_demand: float,
        latency_bs1: float,
    ) -> Tuple[float, int, int]:
        """Compute effective latency and batch size for cross-task batching.

        Steps:
        1. Find saturation batch size and backbone_mean_ms at saturation.
        2. Compute util_dummy = task_demand * latency_b_max / 1000.
        3. Select batch size = util_dummy * max_batch_size, snapped to profile.
        4. Estimate batched latency = latency_bs1 - backbone_mean_ms(bs=1)
           + backbone_mean_ms(selected_bs).

        Args:
            backbone: Backbone name.
            device_type: GPU device type string.
            task_demand: Current demand in req/s.
            latency_bs1: Pipeline avg_latency_ms at batch_size=1 from profiler.

        Returns:
            (batched_latency_ms, selected_batch_size, saturation_batch_size)
        """
        bp = self.batch_profile

        # Step 1: saturation point
        sat_bs, latency_b_max = bp.get_saturation_batch_size(backbone, device_type)

        # Step 2: utilization proxy (per-sample cost at saturation)
        util_dummy = task_demand * (latency_b_max / sat_bs) / 1000.0

        # Step 3: select batch size (capped at saturation batch size)
        target_bs = util_dummy * sat_bs
        target_bs = max(1.0, min(target_bs, float(sat_bs)))
        selected_bs = bp.snap_to_profile(backbone, device_type, target_bs)

        # Step 4: estimate batched latency
        backbone_ms_bs1 = bp.get_backbone_mean_ms(backbone, device_type, 1)
        backbone_ms_selected = bp.get_backbone_mean_ms(backbone, device_type, selected_bs)
        batched_latency = latency_bs1 - backbone_ms_bs1 + backbone_ms_selected

        print(
            f"[BatchInfo] {backbone}/{device_type}: "
            f"sat_bs={sat_bs}, latency_b_max={latency_b_max:.2f}ms, "
            f"util_dummy={util_dummy:.4f}, "
            f"target_bs={target_bs:.1f}, selected_bs={selected_bs}, "
            f"backbone_ms(bs=1)={backbone_ms_bs1:.2f}, "
            f"backbone_ms(bs={selected_bs})={backbone_ms_selected:.2f}, "
            f"latency_bs1={latency_bs1:.2f}, batched_latency={batched_latency:.2f}ms"
        )

        return batched_latency, selected_bs, sat_bs


def build_final_json(
    deployments: List[Deployment],
    pipelines: Dict,
    batch_size_map: Optional[Dict[Tuple[str, str], int]] = None,
    expected_batch_size_map: Optional[Dict[Tuple[str, str], int]] = None,
) -> Dict:
    """Build final JSON output from deployments."""
    state = DeploymentState([])
    for d in deployments:
        state._deployments[(d.server_name, d.backbone)] = d
    plan = state.to_plan_json(pipelines)

    # Inject batch sizes per deployment
    for site in plan.get("sites", []):
        for dep in site.get("deployments", []):
            key = (dep["device_name"], dep["backbone"])
            if batch_size_map and key in batch_size_map:
                dep["max_batch_size"] = batch_size_map[key]
            if expected_batch_size_map and key in expected_batch_size_map:
                dep["expected_batch_size"] = expected_batch_size_map[key]

    return plan
