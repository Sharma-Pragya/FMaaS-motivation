"""Scheduler dispatch and deployment state management for the orchestrator."""

import json
import os
from orchestrator.config import DEPLOYMENT_PLAN_PATH


def run_scheduler(scheduler_name: str, devices: dict, tasks_slo: dict, output_dir: str = None):
    """Run the requested scheduler and save the deployment plan.

    Args:
        scheduler_name: One of 'fmaas', 'fmaas_share', 'clipper-ht', 'clipper-ha',
                        'm4-ht', 'm4-ha'.
        devices: Device configuration dict.
        tasks_slo: Task specification dict with SLO info.
        output_dir: Directory to save deployment_plan.json. Uses default path if None.

    Returns:
        (scheduler, profile, pipelines, plan) — scheduler and profile data are
        stored on the Orchestrator for later use in handle_add_task.
    """
    from planner import FMaaSScheduler, ClipperScheduler, M4Scheduler, ProfileData, SchedulerConfig
    from planner.schedulers import fmaas as fmaas_mod
    from planner.schedulers import clipper as clipper_mod
    from planner.schedulers import m4 as m4_mod
    from planner.parser.profiler import components, pipelines, latency, metric

    profile = ProfileData(components, pipelines, latency, metric)
    config = SchedulerConfig()

    if scheduler_name == 'fmaas':
        scheduler = FMaaSScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, share_mode=False)
        plan = fmaas_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'fmaas_share':
        scheduler = FMaaSScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, share_mode=True)
        plan = fmaas_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'clipper-ht':
        scheduler = ClipperScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=False)
        plan = clipper_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'clipper-ha':
        scheduler = ClipperScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=True)
        plan = clipper_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'm4-ht':
        scheduler = M4Scheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=False)
        plan = m4_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'm4-ha':
        scheduler = M4Scheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=True)
        plan = m4_mod.build_final_json(deployments, pipelines)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. "
                         f"Use: fmaas, fmaas_share, clipper-ht, clipper-ha, m4-ht, m4-ha")

    plan_path = (os.path.join(output_dir, "deployment_plan.json") if output_dir
                 else f"{DEPLOYMENT_PLAN_PATH}.json")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"[Orchestrator] Saved deployment plan to {plan_path} (scheduler={scheduler_name})")

    return scheduler, profile, pipelines, plan


def load_state(plan_path: str, devices: dict = None):
    """Load DeploymentState from deployment_plan.json.

    Args:
        plan_path: Path to deployment_plan.json.
        devices: Full devices config dict (from user_config). When provided,
                 idle servers that have no deployments are included in the
                 returned state so the incremental planner can place tasks on them.

    Returns DeploymentState, or None if file not found.
    """
    from planner.state import DeploymentState

    print(f"[Orchestrator] Loading state from: {plan_path}")
    if not os.path.isfile(plan_path):
        print(f"[Orchestrator] No state file found at {plan_path}")
        return None

    with open(plan_path, 'r') as f:
        plan = json.load(f)

    return DeploymentState.from_deployment_plan(plan, devices=devices)


def save_state(state, plan: dict, pipelines: dict, plan_path: str):
    """Serialize DeploymentState back into deployment_plan.json format and write to disk.

    Args:
        state: Current DeploymentState.
        plan: Existing plan dict (provides top-level metadata preserved in output).
        pipelines: Pipeline registry for decoder path lookup.
        plan_path: File path to write.
    """
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    updated = state.to_plan_json(pipelines)
    # Preserve any top-level keys from the original plan (e.g. metadata)
    plan['sites'] = updated['sites']
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"[Orchestrator] Saved updated state to {plan_path}")


def scheduler_kwargs(scheduler_name: str, scheduler_mode: bool) -> tuple[dict, str]:
    """Map scheduler name + mode flag to the correct keyword argument.

    Returns:
        (kwargs dict for plan_new_task, human-readable mode string)
    """
    name = scheduler_name.lower()
    if name in ('fmaas', 'fmaas_share'):
        return {'share_mode': scheduler_mode}, f"share_mode={scheduler_mode}"
    elif name in ('clipper-ht', 'clipper-ha', 'm4-ht', 'm4-ha'):
        return {'accuracy_mode': scheduler_mode}, f"accuracy_mode={scheduler_mode}"
    else:
        return {'share_mode': scheduler_mode}, f"share_mode={scheduler_mode}"
