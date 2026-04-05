"""Clipper placement scheduler with user-specified backbones.

Like fmaas_place, the user specifies which backbone to use per task.
Unlike fmaas_place, each task gets its own model instance — no sharing
between tasks even if they specify the same backbone. Multiple model
instances can co-exist on the same GPU as long as memory allows.

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


class ClipperPlacementScheduler(BaseScheduler):
    """Clipper-style scheduler where users specify the backbone per task.

    Each task gets its own isolated model instance (1 task : 1 model).
    Multiple model instances can share a GPU when memory permits.
    The __clipper__<task> suffix in the deployment key prevents tasks
    from being co-located on the same model instance; to_plan_json
    strips it transparently so the output JSON shows the real backbone.

    When a BatchProfile is provided, the scheduler accounts for same-task
    batching to compute more accurate utilization estimates.

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
        batch_mode: str = "util_dummy",
    ):
        super().__init__(profile_data, config)
        self.batch_profile = batch_profile
        self.batch_mode = batch_mode
        # Maps (server_name, backbone_key) -> saturation batch size
        self.batch_size_map: Dict[Tuple[str, str], int] = {}
        # Maps (server_name, backbone_key) -> selected (expected) batch size
        self.expected_batch_size_map: Dict[Tuple[str, str], int] = {}

    def schedule(
        self,
        devices: Dict[str, Dict],
        tasks: Dict[str, Dict],
    ) -> List[Deployment]:
        """Schedule tasks onto devices.

        Each task spec must include a 'backbone' key. Every task gets its
        own model instance — no sharing between tasks.

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
                    f"clipper_place requires a backbone per task in user_config."
                )
            temp_plan, demand_left = self._deploy_task(state, task)

            if demand_left is not None and demand_left > self.config.demand_epsilon:
                logger.warning(
                    f"ClipperPlacement: task '{task_name}' has {demand_left:.4f} rps "
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
        accuracy_mode: bool = False,  # accepted for interface compatibility, ignored
    ) -> Tuple[Optional[Dict], float]:
        """Deploy a single task using its user-specified backbone.

        No sharing — skips existing deployments entirely. Places only on
        new servers using a per-task backbone key so each task gets its own
        model instance. Multiple instances can share a GPU if memory allows.

        Args:
            state: Current deployment state.
            task: TaskSpec with backbone set.

        Returns:
            Tuple of (deployment plan dict, remaining demand).
        """
        backbone = task.backbone
        # Unique key per task: prevents merging with any other task's deployment
        # in add_deployment, so each task gets its own model instance + port.
        task_backbone = f"{backbone}__clipper__{task.name}"

        util_tracker: Dict[str, float] = {}
        temp_plan: Dict = {}
        demand_left = task.peak_workload

        backbone_mem = self.data.get_component_mem(backbone)
        candidate_servers = state.get_servers_by_least_capacity(
            backbone_mem, max_util=self.config.util_factor,
        )

        # First, check whether any single new server can satisfy the full task
        # before falling back to multi-device distribution.
        for server in candidate_servers:
            saved_batch_size_map = dict(self.batch_size_map)
            saved_expected_batch_size_map = dict(self.expected_batch_size_map)

            solo_plan, solo_demand_left = self._distribute_demand(
                state, task, [(server.name, task_backbone)],
                remaining_demand=demand_left,
                existing_plan=dict(temp_plan),
                util_tracker=dict(util_tracker),
                real_backbone=backbone,
            )
            if solo_demand_left <= self.config.demand_epsilon:
                return solo_plan, solo_demand_left

            self.batch_size_map = saved_batch_size_map
            self.expected_batch_size_map = saved_expected_batch_size_map

        for server in candidate_servers:
            temp_plan, demand_left = self._distribute_demand(
                state, task, [(server.name, task_backbone)],
                remaining_demand=demand_left,
                existing_plan=temp_plan,
                util_tracker=util_tracker,
                real_backbone=backbone,
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
        real_backbone: Optional[str] = None,
    ) -> Tuple[Dict, float]:
        """Distribute task demand across endpoints.

        Same allocation logic as fmaas_place._distribute_demand, with an
        extra real_backbone parameter for pipeline lookup when the endpoint
        backbone key carries the __clipper__<task> suffix.

        Args:
            state: Current deployment state.
            task: Task specification.
            endpoints: List of (server_name, backbone_key) tuples.
            remaining_demand: Demand left to allocate. Defaults to task.peak_workload.
            existing_plan: Deployment plan to extend. Defaults to empty dict.
            util_tracker: Utilization tracker across calls. Defaults to empty dict.
            real_backbone: Actual backbone name for pipeline/memory lookup when
                           backbone_key carries a __clipper__ suffix.

        Returns:
            Tuple of (updated deployment plan, remaining demand).
        """
        task_demand = remaining_demand if remaining_demand is not None else task.peak_workload
        temp_plan = existing_plan if existing_plan is not None else {}
        if util_tracker is None:
            util_tracker = {}

        for server_name, backbone_key in endpoints:
            if task_demand <= self.config.demand_epsilon:
                break

            lookup_backbone = real_backbone if real_backbone is not None else backbone_key

            pid = self.data.find_pipeline_id(task.name, lookup_backbone)
            if not pid:
                logger.warning(f"No pipeline found for task '{task.name}' with backbone '{lookup_backbone}'")
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

            total_util = util_tracker[server_name]
            left_cap = self.config.util_factor - total_util

            if left_cap <= 1e-6:
                continue

            # Batching-aware scheduling (single-task only for clipper)
            selected_bs = 1
            sat_bs = 1
            latency = latency_bs1
            if self.batch_profile is not None:
                try:
                    if self.batch_mode == "fixedpoint":
                        latency, selected_bs, sat_bs = self._compute_batched_latency_fixedpoint(
                            lookup_backbone, server.type, task_demand, latency_bs1,
                        )
                    else:
                        latency, selected_bs, sat_bs = self._compute_batched_latency(
                            lookup_backbone, server.type, task_demand, latency_bs1,
                        )
                except KeyError:
                    logger.debug(
                        f"No batch profile for {lookup_backbone}/{server.type}, "
                        f"falling back to bs=1"
                    )

            task_cap_needed = (task_demand / selected_bs) * latency / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * selected_bs * 1000.0 / latency

            print(
                f"[Alloc] Task '{task.name}' on {server_name}/{lookup_backbone}: "
                f"latency={latency:.2f}ms, batch_size={selected_bs}, sat_bs={sat_bs}, "
                f"task_cap_needed={task_cap_needed:.6f}, left_cap={left_cap:.4f}, "
                f"allocated_cap={allocated_cap:.6f}, allocated_demand={allocated_demand:.2f} req/s, "
                f"remaining_demand={max(0, task_demand - allocated_demand):.2f}"
            )

            task_demand -= allocated_demand
            util_tracker[server_name] += allocated_cap

            # Track batch sizes per deployment
            self.batch_size_map[(server_name, backbone_key)] = sat_bs
            self.expected_batch_size_map[(server_name, backbone_key)] = selected_bs

            pipeline = self.data.get_pipeline(pid)
            components = self.data.get_pipeline_components_mem(pipeline)

            deployment = Deployment(
                server_name=server_name,
                backbone=backbone_key,
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

            temp_plan[(server_name, backbone_key)] = deployment

        return temp_plan, max(0, task_demand)

    def _compute_batched_latency(
        self,
        backbone: str,
        device_type: str,
        task_demand: float,
        latency_bs1: float,
    ) -> Tuple[float, int, int]:
        """Compute effective latency and batch size for same-task batching.

        Same algorithm as FMaaSPlacementScheduler._compute_batched_latency.

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

    def _compute_batched_latency_fixedpoint(
        self,
        backbone: str,
        device_type: str,
        task_demand: float,
        latency_bs1: float,
        max_iters: int = 10,
    ) -> Tuple[float, int, int]:
        """Compute expected batch size via fixed-point iteration (single task).

        arrivals = task_demand * batched_latency / 1000

        where batched_latency = (latency_bs1 - backbone_ms(1)) + backbone_ms(bs).

        Iterates until bs converges.

        Returns:
            (batched_latency_ms, selected_batch_size, saturation_batch_size)
        """
        import math
        bp = self.batch_profile

        sat_bs, _ = bp.get_saturation_batch_size(backbone, device_type)
        backbone_ms_bs1 = bp.get_backbone_mean_ms(backbone, device_type, 1)
        decoder_overhead = latency_bs1 - backbone_ms_bs1

        bs = 1
        for i in range(max_iters):
            backbone_ms_bs = bp.get_backbone_mean_ms(backbone, device_type, bs)
            batched_latency = decoder_overhead + backbone_ms_bs

            arrivals = task_demand * (batched_latency / 1000.0)
            new_bs_raw = max(1, math.ceil(arrivals))
            new_bs = bp.snap_ceil_to_profile(backbone, device_type, min(new_bs_raw, sat_bs))

            print(
                f"[FixedPoint] {backbone}/{device_type} iter={i+1}: "
                f"bs={bs}, backbone_ms={backbone_ms_bs:.2f}, "
                f"batched_latency={batched_latency:.2f}ms, "
                f"arrivals={arrivals:.3f}, ceil={new_bs_raw}, snapped={new_bs}"
            )

            if new_bs == bs:
                break
            bs = new_bs

        backbone_ms_final = bp.get_backbone_mean_ms(backbone, device_type, bs)
        batched_latency = decoder_overhead + backbone_ms_final

        print(
            f"[FixedPoint] {backbone}/{device_type}: CONVERGED "
            f"expected_bs={bs}, batched_latency={batched_latency:.2f}ms, "
            f"sat_bs={sat_bs}, task_demand={task_demand:.2f} req/s"
        )

        return batched_latency, bs, sat_bs


def build_final_json(
    deployments: List[Deployment],
    pipelines: Dict,
    batch_size_map: Optional[Dict[Tuple[str, str], int]] = None,
    expected_batch_size_map: Optional[Dict[Tuple[str, str], int]] = None,
) -> Dict:
    """Build final JSON output from deployments.

    The __clipper__<task> suffix is stripped transparently by
    DeploymentState.to_plan_json().
    """
    state = DeploymentState([])
    for d in deployments:
        state._deployments[(d.server_name, d.backbone)] = d
    plan = state.to_plan_json(pipelines)

    # Inject max_batch_size per deployment from scheduler's batch_size_map
    if batch_size_map:
        for site in plan.get("sites", []):
            for dep in site.get("deployments", []):
                # batch_size_map keys use __clipper__ suffix
                bb = dep["backbone"]
                tasks = list(dep.get("tasks", {}).keys())
                for task_name in tasks:
                    key = (dep["device_name"], f"{bb}__clipper__{task_name}")
                    if key in batch_size_map:
                        dep["max_batch_size"] = batch_size_map[key]
                        if expected_batch_size_map and key in expected_batch_size_map:
                            dep["expected_batch_size"] = expected_batch_size_map[key]
                        break

    return plan
