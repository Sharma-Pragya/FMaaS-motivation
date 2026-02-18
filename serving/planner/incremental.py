"""Incremental planning for runtime task addition.

This module provides the ability to plan a new task incrementally on top
of an existing DeploymentState, without re-running the full scheduler.
It calls the scheduler's _deploy_task() externally, then computes a
deployment "diff" describing what changed.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .data_loader import ProfileData
from .models import Deployment
from .state import DeploymentState

logger = logging.getLogger(__name__)


@dataclass
class DeploymentDiff:
    """Describes a single deployment action resulting from incremental planning.

    Attributes:
        action: "add_full" (new server + backbone) or "add_decoder" (hot-add to existing).
        site_manager: Site manager ID that owns this device.
        server_name: Device name.
        backbone: Backbone model name.
        ip: Device endpoint (ip:port).
        device_type: GPU type.
        new_decoders: List of decoder specs to deploy (task, type, path).
        full_deployment: Complete deployment spec for "add_full" actions.
    """
    action: str
    site_manager: str
    server_name: str
    backbone: str
    ip: str
    device_type: str
    new_decoders: List[dict] = field(default_factory=list)
    full_deployment: Optional[dict] = None


def plan_new_task(
    scheduler,
    state: DeploymentState,
    task_name: str,
    task_spec: dict,
    profile_data: ProfileData,
    pipelines: dict,
    **scheduler_kwargs
) -> Tuple[DeploymentState, List[DeploymentDiff]]:
    """Plan a new task incrementally on existing state.

    Calls the scheduler's _deploy_task() on the current state, commits
    the result, and computes a diff of what changed.

    Args:
        scheduler: A scheduler instance (e.g. FMaaSScheduler) with
            _create_task_spec() and _deploy_task() methods.
        state: Current DeploymentState (will be mutated).
        task_name: Name of the new task.
        task_spec: Task specification dict with keys: type, peak_workload,
            latency, metric, value.
        profile_data: ProfileData instance for pipeline lookups.
        pipelines: Raw pipelines dict (from profiler) for decoder path resolution.
        **scheduler_kwargs: Scheduler-specific parameters (e.g., share_mode for FMaaS,
            accuracy_mode for Clipper/M4).

    Returns:
        Tuple of (updated state, list of DeploymentDiff actions).
    """
    # 1. Snapshot existing deployment keys and their tasks
    old_snapshot: Dict[Tuple[str, str], set] = {}
    for deployment in state.get_all_deployments():
        key = (deployment.server_name, deployment.backbone)
        old_snapshot[key] = set(deployment.task_info.keys())

    # 2. Create TaskSpec and plan
    task = scheduler._create_task_spec(task_name, task_spec)
    temp_plan, demand_left = scheduler._deploy_task(state, task, **scheduler_kwargs)
    logger.info(f"Incremental plan for '{task_name}': demand_left={demand_left}, "
                f"temp_plan keys={list(temp_plan.keys()) if temp_plan else []}")

    # 3. Commit temp_plan to state
    if temp_plan:
        for deployment in temp_plan.values():
            state.add_deployment(
                deployment,
                scheduler.config.base_port,
                scheduler.config.port_increment,
            )

    # 4. Compute diff
    diffs = []
    for deployment in state.get_all_deployments():
        key = (deployment.server_name, deployment.backbone)

        # Strip Clipper suffix for use in diff (e.g., "momentbase__clipper__task" → "momentbase")
        backbone_for_diff = deployment.backbone
        if '__clipper__' in deployment.backbone:
            backbone_for_diff = deployment.backbone.split('__clipper__')[0]

        # Build deployment spec via to_plan_json (single source of truth)
        temp_state = DeploymentState([])
        temp_state._deployments[(deployment.server_name, deployment.backbone)] = deployment
        spec = temp_state.to_plan_json(pipelines)["sites"][0]["deployments"][0]

        if key not in old_snapshot:
            # Brand new deployment — add_full
            diffs.append(DeploymentDiff(
                action="add_full",
                site_manager=deployment.site_manager,
                server_name=deployment.server_name,
                backbone=backbone_for_diff,
                ip=deployment.ip,
                device_type=deployment.device_type,
                new_decoders=[d for d in spec["decoders"] if d["task"] == task_name],
                full_deployment=spec,
            ))
        else:
            # Existing deployment — check if new task was added
            old_tasks = old_snapshot[key]
            new_tasks = set(deployment.task_info.keys()) - old_tasks
            if task_name in new_tasks:
                diffs.append(DeploymentDiff(
                    action="add_decoder",
                    site_manager=deployment.site_manager,
                    server_name=deployment.server_name,
                    backbone=backbone_for_diff,
                    ip=deployment.ip,
                    device_type=deployment.device_type,
                    new_decoders=[d for d in spec["decoders"] if d["task"] == task_name],
                ))

    logger.info(f"Computed {len(diffs)} diff(s) for task '{task_name}'")
    return state, diffs
