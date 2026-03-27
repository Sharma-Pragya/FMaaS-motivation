"""Orchestrator — deployment planning and runtime task management.

Responsible for:
  - Running the scheduler to produce a deployment plan
  - Planning incremental task additions at runtime
  - Persisting plan state to disk

Transport (MQTT or local) is handled by a BaseSiteManager implementation
passed in from server.py (MQTT mode) or local_runner.py (local mode).
"""

import os
import threading


class Orchestrator:
    """Stateful orchestrator: manages deployment planning and runtime task addition."""

    def __init__(self, devices, tasks, scheduler_name, state_dir):
        self.devices = devices
        self.tasks = tasks
        self.scheduler_name = scheduler_name
        self.state_dir = os.path.abspath(state_dir)
        self.plan = None

        self._add_task_lock = threading.Lock()
        self._scheduler = None
        self._profile_data = None
        self._pipelines = None
        self._total_requests_generated = 0  # for unique req_id generation across add-task calls

    # ------------------------------------------------------------------ #
    # State helpers                                                        #
    # ------------------------------------------------------------------ #

    def _state_path(self):
        return os.path.join(self.state_dir, "deployment_plan.json")

    def _load_state(self):
        from orchestrator.planner import load_state
        return load_state(self._state_path(), devices=self.devices)

    def _save_state(self, state):
        from orchestrator.planner import save_state
        if self.plan is None:
            print("[Orchestrator] WARNING: Cannot save state — no plan exists")
            return
        save_state(state, self.plan, self._pipelines, self._state_path())

    # ------------------------------------------------------------------ #
    # Planning                                                             #
    # ------------------------------------------------------------------ #

    def run_deployment_plan(self, devices, tasks_slo, scheduler_name='fmaas', output_dir=None):
        """Run the scheduler and save deployment_plan.json.

        Stores scheduler, profile, and pipelines for later use in handle_add_task.
        Returns the deployment plan dict.
        """
        from orchestrator.planner import run_scheduler
        scheduler, profile, pipelines, plan = run_scheduler(
            scheduler_name, devices, tasks_slo, output_dir
        )
        self._scheduler = scheduler
        self._profile_data = profile
        self._pipelines = pipelines
        self.plan = plan
        return plan

    # ------------------------------------------------------------------ #
    # Runtime task addition                                                #
    # ------------------------------------------------------------------ #

    def handle_add_task(self, task_name, task_spec, scheduler_mode):
        """Add a new task at runtime: plan → compute diffs → save state.

        Returns:
            (success: bool, message: str, diffs: list, incremental_plan: dict)

        The caller (server.py or local_runner.py) is responsible for executing
        the diffs via site_manager.apply_diff() and sending new requests.
        """
        with self._add_task_lock:
            from planner.incremental import plan_new_task
            from orchestrator.planner import scheduler_kwargs

            state = self._load_state()
            if state is None:
                return False, "No deployment state; run initial deploy first.", [], {}

            task_spec_dict = {
                "type": task_spec.get("type", "classification"),
                "peak_workload": task_spec.get("peak_workload", 0),
                "latency": task_spec.get("latency", float("inf")),
                "metric": task_spec.get("metric", "mae"),
                "value": task_spec.get("value", 0),
            }

            kwargs, mode_str = scheduler_kwargs(self.scheduler_name, scheduler_mode)

            state, diffs, incremental_plan = plan_new_task(
                self._scheduler, state, task_name, task_spec_dict,
                self._profile_data, self._pipelines, **kwargs
            )
            print(f"[Orchestrator] Planned '{task_name}' with {mode_str}. Diffs: {diffs}")
            print(f"[Orchestrator] incremental_plan: {incremental_plan}")

            self._save_state(state)

            if not diffs:
                return True, "Task planned (no runtime deployment).", [], incremental_plan

            return True, "Deployment diffs ready.", diffs, incremental_plan

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def get_total_requests_generated(self):
        return self._total_requests_generated

    def increment_requests_generated(self, count: int):
        self._total_requests_generated += count

    def plan_only(self, plan, routed_trace, output_dir=None):
        import json
        print(f"[Orchestrator] Plan-only mode. Deployment plan saved to {output_dir}")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "trace_summary.json"), "w") as f:
                json.dump({
                    "scheduler": self.scheduler_name,
                    "num_sites": len(plan.get("sites", [])),
                    "num_routed": len(routed_trace),
                }, f, indent=2)
