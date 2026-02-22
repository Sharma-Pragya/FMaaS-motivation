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
        action: "add_full" (new server + backbone), "add_decoder" (hot-add to existing),
                or "migrate" (backbone swap on existing server — kill old, start new).
        site_manager: Site manager ID that owns this device.
        server_name: Device name.
        backbone: New backbone model name.
        old_backbone: Previous backbone name (only set for "migrate" action).
        ip: Device endpoint (ip:port).
        device_type: GPU type.
        new_decoders: List of decoder specs to deploy (task, type, path).
        full_deployment: Complete deployment spec for "add_full" and "migrate" actions.
    """
    action: str
    site_manager: str
    server_name: str
    backbone: str
    ip: str
    device_type: str
    old_backbone: Optional[str] = None
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
    # 1. Snapshot existing deployment keys and their tasks.
    #    Also build a per-server set of old backbones so we can detect backbone
    #    migrations produced by _fit() (old key disappears, new key appears on same server).
    #    We use a set per server (not a single value) so that when a server gets multiple
    #    new deployments after _fit, only the first new key consumes the migration slot;
    #    subsequent new keys on the same server are treated as add_full.
    old_snapshot: Dict[Tuple[str, str], set] = {}
    old_server_backbones: Dict[str, set] = {}  # server_name → set of old backbones
    for deployment in state.get_all_deployments():
        key = (deployment.server_name, deployment.backbone)
        old_snapshot[key] = set(deployment.task_info.keys())
        old_server_backbones.setdefault(deployment.server_name, set()).add(deployment.backbone)

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

    # 3b. Fix total_requested_workload: after merging, compute the true total
    #     request_per_sec for task_name across all deployments and write it back
    #     so deployment_plan.json reflects the real aggregate demand.
    true_total_rps = sum(
        d.task_info[task_name].request_per_sec
        for d in state.get_all_deployments()
        if task_name in d.task_info
    )
    for d in state.get_all_deployments():
        if task_name in d.task_info:
            d.task_info[task_name].total_requested_workload = true_total_rps

    # 4. Compute which old keys were removed by _fit (disappeared from state).
    #    These are the only keys eligible to be matched as migration sources.
    #    We build a mutable map: server_name → [removed_backbone, ...] so each
    #    removed backbone can be consumed at most once.
    current_keys = {(d.server_name, d.backbone) for d in state.get_all_deployments()}
    removed_backbones: Dict[str, List[str]] = {}
    for (server_name, backbone) in old_snapshot:
        if (server_name, backbone) not in current_keys:
            removed_backbones.setdefault(server_name, []).append(backbone)

    # 5. Compute diff
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
            # Key is new — check if there is an unmatched removed backbone on this server.
            # If yes, consume one: this new key is a migration of that old backbone.
            # If no removed backbones remain, it is a genuinely new add_full deployment.
            server_removed = removed_backbones.get(deployment.server_name, [])
            if server_removed:
                prev_backbone = server_removed.pop(0)  # consume one migration slot
                old_backbone_for_diff = prev_backbone
                if '__clipper__' in prev_backbone:
                    old_backbone_for_diff = prev_backbone.split('__clipper__')[0]
                diffs.append(DeploymentDiff(
                    action="migrate",
                    site_manager=deployment.site_manager,
                    server_name=deployment.server_name,
                    backbone=backbone_for_diff,
                    old_backbone=old_backbone_for_diff,
                    ip=deployment.ip,
                    device_type=deployment.device_type,
                    new_decoders=spec["decoders"],  # all decoders — full restart
                    full_deployment=spec,
                ))
            else:
                # Genuinely new server+backbone — add_full
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
