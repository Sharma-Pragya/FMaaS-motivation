"""Orchestrator — coordinates deployment, inference, and runtime task addition.

Delegates to:
  - orchestrator.mqtt_client.MQTTManager  : MQTT connection + ACK tracking
  - orchestrator.planner                  : scheduler dispatch + state load/save
  - orchestrator.router                   : request routing
"""

import json
import os
import time
import threading
from collections import defaultdict

import paho.mqtt.client as mqtt
import ssl

from orchestrator.config import BROKER, PORT, TIMEOUT
from orchestrator.mqtt_client import MQTTManager


class Orchestrator:
    """Long-running orchestrator: manages deployment, inference, and runtime task addition."""

    def __init__(self, devices, tasks, scheduler_name, state_dir):
        self.devices = devices
        self.tasks = tasks
        self.scheduler_name = scheduler_name
        self.state_dir = os.path.abspath(state_dir)
        self.plan = None

        self._mqtt = MQTTManager()
        self._add_task_lock = threading.Lock()
        self._scheduler = None
        self._profile_data = None
        self._pipelines = None
        self._total_requests_generated = 0  # for unique req_id generation across add-task calls
        self._inference_client = None       # kept alive during continuous inference mode

    # ------------------------------------------------------------------ #
    # State helpers                                                        #
    # ------------------------------------------------------------------ #

    def _state_path(self):
        return os.path.join(self.state_dir, "deployment_plan.json")

    def _load_state(self):
        from orchestrator.planner import load_state
        return load_state(self._state_path())

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
    # Deployment                                                           #
    # ------------------------------------------------------------------ #

    def publish_deployments(self, client, plan, routed_trace, output_dir=None):
        """Publish deployment specs and request chunks to all site managers."""
        site_requests = defaultdict(list)
        for r in routed_trace:
            site_requests[r.site_manager].append(r.to_dict())

        total_count = len(routed_trace)
        self._total_requests_generated += total_count
        print(f"[Orchestrator] Sent {total_count} requests (total={self._total_requests_generated})")

        for site in plan["sites"]:
            site_id = site["id"]
            reqs = site_requests.get(site_id, [])

            # Deploy message first — triggers clear_state() on site manager
            deploy_msg = {"deployments": site["deployments"]}
            if output_dir:
                deploy_msg["output_dir"] = output_dir
            client.publish(f"fmaas/deploy/site/{site_id}", json.dumps(deploy_msg), qos=1)
            print(f"[MQTT] Sent deployment to {site_id} (output_dir={output_dir})")
            time.sleep(30)  # allow site manager to process deploy before requests arrive

            # Send requests in chunks
            chunk_length = 3000
            for i in range(0, len(reqs), chunk_length):
                client.publish(
                    f"fmaas/deploy/site/{site_id}/req",
                    json.dumps({"runtime_requests": reqs[i:i + chunk_length]}),
                    qos=1,
                )
                print(f"[MQTT] Sent request chunk [{i}:{i+chunk_length}] to {site_id}")
                time.sleep(5)
            print(f"[MQTT] All {len(reqs)} requests sent to {site_id}")
            time.sleep(0.1)

    def initial_deployment(self, plan, routed_trace, output_dir=None):
        """Deploy all sites and wait for ACKs before returning."""
        self._mqtt.reset_acks({site["id"] for site in plan["sites"]}, ack_type='deploytime')
        client = self._mqtt.connect("orchestrator-deploy")

        print("Publishing deployments + requests to all sites...")
        self.publish_deployments(client, plan, routed_trace, output_dir=output_dir)

        print(f"Waiting for deployment ACKs from {len(self._mqtt._expected_sites)} sites...")
        if not self._mqtt.wait_for_acks(timeout=TIMEOUT):
            print(f"WARNING: Timeout waiting for all ACKs. "
                  f"Received from: {list(self._mqtt._acks.keys())}")

        client.disconnect()
        client.loop_stop()
        print(f"Deployment phase complete. Results in {output_dir}")

    # ------------------------------------------------------------------ #
    # Runtime request delivery                                             #
    # ------------------------------------------------------------------ #

    def send_new_requests(self, routed_trace):
        """Send new runtime requests to site managers (for dynamically added tasks).

        Reuses _inference_client if available (avoids new TLS connection overhead).
        """
        site_requests = defaultdict(list)
        for r in routed_trace:
            site_requests[r.site_manager].append(r.to_dict())

        total_count = len(routed_trace)
        self._total_requests_generated += total_count
        print(f"[Orchestrator] Sending {total_count} new requests (total={self._total_requests_generated})")

        if self._inference_client:
            client = self._inference_client
            should_cleanup = False
            print("[MQTT] Reusing existing MQTT connection (fast path)")
        else:
            client = self._mqtt._make_client("orchestrator-add-requests")
            client.connect(BROKER, PORT, 60)
            client.loop_start()
            time.sleep(1)
            should_cleanup = True
            print("[MQTT] Created new MQTT connection")

        for site_id, reqs in site_requests.items():
            chunk_length = 3000
            for i in range(0, len(reqs), chunk_length):
                client.publish(
                    f"fmaas/deploy/site/{site_id}/req",
                    json.dumps({"runtime_requests": reqs[i:i + chunk_length]}),
                    qos=1,
                )
                print(f"[MQTT] Sent chunk [{i}:{i+chunk_length}] to {site_id}")

        if should_cleanup:
            client.loop_stop()
            client.disconnect()

        print("[MQTT] All new requests sent.")

    # ------------------------------------------------------------------ #
    # Inference lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def trigger_runtime_start(self, client, plan):
        for site in plan["sites"]:
            topic = f"fmaas/runtime/start/site/{site['id']}"
            client.publish(topic, json.dumps({"command": "start"}), qos=1)
            print(f"Runtime start published to {topic}")
            time.sleep(0.05)

    def run_inference_requests(self):
        """Trigger runtime start on site managers (non-blocking — continuous mode)."""
        self._mqtt.reset_acks({site["id"] for site in self.plan["sites"]}, ack_type='runtime')
        client = self._mqtt.connect("orchestrator-inference")

        self.trigger_runtime_start(client, self.plan)

        # Keep client alive to receive runtime ACKs throughout the experiment
        self._inference_client = client
        print("[Orchestrator] Runtime inference triggered (continuous mode)")

    def wait_for_inference_completion(self, timeout=TIMEOUT):
        """Block until all site managers send runtime ACKs."""
        self._mqtt.reset_acks(self._mqtt._expected_sites, ack_type='runtime')

        if not self._mqtt.wait_for_acks(timeout=timeout):
            print(f"WARNING: Timeout waiting for runtime ACKs. "
                  f"Received from: {list(self._mqtt._acks.keys())}")

        print("Runtime inference complete.")
        if self._inference_client:
            self._inference_client.disconnect()
            self._inference_client.loop_stop()
            self._inference_client = None

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def trigger_cleanup(self, client, plan):
        """Send cleanup signal to all site managers."""
        for site in plan["sites"]:
            topic = f"fmaas/cleanup/site/{site['id']}"
            client.publish(topic, json.dumps({"command": "cleanup"}), qos=1)
            print(f"Cleanup signal published to {topic}")
            time.sleep(0.05)

    def cleanup_only(self, plan):
        """Kill all Triton servers without running experiments."""
        self._mqtt.reset_acks({site["id"] for site in plan["sites"]}, ack_type='cleanup')
        client = self._mqtt.connect("orchestrator-cleanup")

        self.trigger_cleanup(client, plan)

        if not self._mqtt.wait_for_acks(timeout=120):
            print(f"WARNING: Timeout waiting for cleanup ACKs. "
                  f"Received from: {list(self._mqtt._acks.keys())}")

        client.disconnect()
        client.loop_stop()
        print("Cleanup complete. All device servers killed.")

    # ------------------------------------------------------------------ #
    # Runtime task addition                                                #
    # ------------------------------------------------------------------ #

    def _publish_diff(self, diffs):
        """Publish deployment diffs to site managers and wait for ACKs."""
        if not diffs:
            print("[Orchestrator] No diffs to publish.")
            return

        client = self._inference_client
        if client is None:
            raise RuntimeError("No active MQTT client. Call run_inference_requests() first.")

        self._mqtt.reset_acks({d.site_manager for d in diffs})

        for d in diffs:
            if d.action == "add_full" and d.full_deployment:
                payload = json.dumps({"deployments": [d.full_deployment]})
                client.publish(f"fmaas/deploy/site/{d.site_manager}/add", payload, qos=1)
                print(f"[MQTT] Published /add to {d.site_manager} (server={d.server_name})")
            elif d.action == "add_decoder":
                payload = json.dumps({"device": d.ip, "decoders": d.new_decoders})
                client.publish(f"fmaas/deploy/site/{d.site_manager}/update", payload, qos=1)
                print(f"[MQTT] Published /update to {d.site_manager} (server={d.server_name})")

        print(f"[Orchestrator] Waiting for ACKs from {len(self._mqtt._expected_sites)} site(s)...")
        if not self._mqtt.wait_for_acks(timeout=60):
            print(f"[WARN] Timeout waiting for ACKs. "
                  f"Received: {len(self._mqtt._acks)}/{len(self._mqtt._expected_sites)}")
        else:
            print("[Orchestrator] All ACKs received for diff operation.")

    def handle_add_task(self, task_name, task_spec, scheduler_mode):
        """Add a new task at runtime: plan → publish diff → save state.

        Args:
            task_name: Name of the new task.
            task_spec: Task specification dict.
            scheduler_mode: True/False (share_mode or accuracy_mode depending on scheduler).

        Returns:
            (success: bool, message: str, actions: list)
        """
        with self._add_task_lock:
            from planner.incremental import plan_new_task
            from orchestrator.planner import scheduler_kwargs

            state = self._load_state()
            if state is None:
                return False, "No deployment state; run initial deploy first.", []

            task_spec_dict = {
                "type": task_spec.get("type", "classification"),
                "peak_workload": task_spec.get("peak_workload", 0),
                "latency": task_spec.get("latency", float("inf")),
                "metric": task_spec.get("metric", "mae"),
                "value": task_spec.get("value", 0),
            }

            kwargs, mode_str = scheduler_kwargs(self.scheduler_name, scheduler_mode)

            state, diffs = plan_new_task(
                self._scheduler, state, task_name, task_spec_dict,
                self._profile_data, self._pipelines, **kwargs
            )
            print(f"[Orchestrator] Planned '{task_name}' with {mode_str}. Diffs: {diffs}")

            if not diffs:
                self._save_state(state)
                return True, "Task planned (no new deployments).", []

            self._publish_diff(diffs)
            self._save_state(state)

            actions = [
                {
                    "action": d.action,
                    "site_manager": d.site_manager,
                    "server_name": d.server_name,
                    "backbone": d.backbone,
                    "ip": d.ip,
                    "device_type": d.device_type,
                    "new_decoders": d.new_decoders,
                }
                for d in diffs
            ]
            return True, "Deployment diff applied.", actions

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def get_total_requests_generated(self):
        """Return total requests generated so far (for req_id offset in add-task)."""
        return self._total_requests_generated

    def plan_only(self, plan, routed_trace, output_dir=None):
        print(f"[Orchestrator] Plan-only mode. Deployment plan saved to {output_dir}")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "trace_summary.json"), "w") as f:
                json.dump({
                    "scheduler": self.scheduler_name,
                    "num_sites": len(plan.get("sites", [])),
                    "num_routed": len(routed_trace),
                }, f, indent=2)
