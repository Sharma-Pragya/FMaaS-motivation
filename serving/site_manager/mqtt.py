"""MQTTSiteManager — geo-distributed site manager transport via MQTT broker.

This is the original transport behavior extracted from orchestrator/main.py.
It publishes commands to site managers over MQTT and waits for ACKs.

The site manager processes (site_manager/main.py) remain unchanged and continue
to run as separate processes subscribed to MQTT topics.
"""

import json
import os
import time
import uuid
from collections import defaultdict

from orchestrator.config import BROKER, PORT, TIMEOUT
from orchestrator.mqtt_client import MQTTManager
from site_manager.base import BaseSiteManager

_MQTT_ID_SUFFIX = f"{os.getpid()}-{int(time.time())}"


class MQTTSiteManager(BaseSiteManager):
    """Communicates with remote site managers via MQTT broker.

    Wraps the original publish/ACK-wait logic that lived in Orchestrator.
    The orchestrator calls these methods; the MQTT layer handles transport.
    """

    def __init__(self):
        self._mqtt = MQTTManager()
        self._inference_client = None   # kept alive during inference phase
        self._runtime_start_epoch = None

    # ------------------------------------------------------------------ #
    # deploy                                                               #
    # ------------------------------------------------------------------ #

    def deploy(self, plan: dict, routed_trace: list, output_dir: str = None):
        """Publish deployment specs + request chunks to all site managers, wait for ACKs."""
        self._mqtt.reset_acks({site["id"] for site in plan["sites"]}, ack_type='deploytime')
        client = self._mqtt.connect(f"orchestrator-deploy-{_MQTT_ID_SUFFIX}")

        print("Publishing deployments + requests to all sites...")
        self._publish_deployments(client, plan, routed_trace, output_dir)

        print(f"Waiting for deployment ACKs from {len(self._mqtt._expected_sites)} sites...")
        if not self._mqtt.wait_for_acks(timeout=TIMEOUT):
            print(f"WARNING: Timeout waiting for all ACKs. "
                  f"Received from: {list(self._mqtt._acks.keys())}")

        client.disconnect()
        client.loop_stop()
        print(f"Deployment phase complete. Results in {output_dir}")

    def _publish_deployments(self, client, plan, routed_trace, output_dir=None):
        site_requests = defaultdict(list)
        for r in routed_trace:
            site_requests[r.site_manager].append(r.to_dict())

        for site in plan["sites"]:
            site_id = site["id"]
            reqs = site_requests.get(site_id, [])

            deploy_msg = {"deployments": site["deployments"]}
            if output_dir:
                deploy_msg["output_dir"] = output_dir
            client.publish(f"fmaas/deploy/site/{site_id}", json.dumps(deploy_msg), qos=1)
            print(f"[MQTT] Sent deployment to {site_id} (output_dir={output_dir})")
            time.sleep(30)

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

    # ------------------------------------------------------------------ #
    # run_inference / wait_for_completion                                  #
    # ------------------------------------------------------------------ #

    def run_inference(self):
        """Publish runtime start to all site managers (non-blocking)."""
        plan = self._plan
        self._mqtt.reset_acks({site["id"] for site in plan["sites"]}, ack_type='runtime')
        client = self._mqtt.connect(f"orchestrator-inference-{_MQTT_ID_SUFFIX}")

        for site in plan["sites"]:
            topic = f"fmaas/runtime/start/site/{site['id']}"
            client.publish(topic, json.dumps({"command": "start"}), qos=1)
            print(f"Runtime start published to {topic}")
            time.sleep(0.05)

        self._inference_client = client
        self._runtime_start_epoch = time.time()
        print("[MQTTSiteManager] Runtime inference triggered (continuous mode)")

    def wait_for_completion(self, timeout: float = TIMEOUT):
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
        self._runtime_start_epoch = None

    # ------------------------------------------------------------------ #
    # apply_diff                                                           #
    # ------------------------------------------------------------------ #

    def apply_diff(self, diff) -> None:
        """Publish one deployment diff and block until the site manager ACKs it."""
        client = self._inference_client
        if client is None:
            raise RuntimeError("No active MQTT client. Call run_inference() first.")

        ack_id = f"deploy-{uuid.uuid4().hex[:12]}"

        if diff.action == "add_full" and diff.full_deployment:
            payload = json.dumps({"deployments": [diff.full_deployment], "ack_id": ack_id})
            client.publish(f"fmaas/deploy/site/{diff.site_manager}/add", payload, qos=1)
            print(f"[MQTT] Published /add to {diff.site_manager} (server={diff.server_name})")
        elif diff.action == "add_decoder":
            payload = json.dumps({"device": diff.ip, "decoders": diff.new_decoders, "ack_id": ack_id})
            client.publish(f"fmaas/deploy/site/{diff.site_manager}/update", payload, qos=1)
            print(f"[MQTT] Published /update to {diff.site_manager} (server={diff.server_name})")
        elif diff.action == "migrate" and diff.full_deployment:
            payload = json.dumps({
                "deployments": [diff.full_deployment],
                "old_backbone": diff.old_backbone,
                "ack_id": ack_id,
            })
            client.publish(f"fmaas/deploy/site/{diff.site_manager}/migrate", payload, qos=1)
            print(f"[MQTT] Published /migrate to {diff.site_manager} "
                  f"(server={diff.server_name}, {diff.old_backbone} → {diff.backbone})")
        else:
            return

        # Wait for this specific ACK before returning
        received_id, payload_out = self._mqtt.wait_for_any_ack({ack_id}, timeout=120)
        if received_id is None:
            raise TimeoutError(f"Timed out waiting for ACK {ack_id} ({diff.action} on {diff.site_manager})")
        if payload_out.get("error"):
            raise RuntimeError(f"Deployment failed for {ack_id}: {payload_out['error']}")

    # ------------------------------------------------------------------ #
    # send_requests                                                        #
    # ------------------------------------------------------------------ #

    def send_requests(self, routed_trace: list):
        """Publish new request chunks to site managers."""
        site_requests = defaultdict(list)
        for r in routed_trace:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            site_requests[d["site_manager"]].append(d)

        if self._inference_client:
            client = self._inference_client
            should_cleanup = False
        else:
            client = self._mqtt._make_client("orchestrator-add-requests")
            client.connect(BROKER, PORT, 60)
            client.loop_start()
            time.sleep(1)
            should_cleanup = True

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

    # ------------------------------------------------------------------ #
    # cleanup                                                              #
    # ------------------------------------------------------------------ #

    def cleanup(self, plan: dict):
        """Publish cleanup signal to all site managers and wait for ACKs."""
        self._mqtt.reset_acks({site["id"] for site in plan["sites"]}, ack_type='cleanup')
        client = self._mqtt.connect(f"orchestrator-cleanup-{_MQTT_ID_SUFFIX}")

        for site in plan["sites"]:
            topic = f"fmaas/cleanup/site/{site['id']}"
            client.publish(topic, json.dumps({"command": "cleanup"}), qos=1)
            print(f"Cleanup signal published to {topic}")
            time.sleep(0.05)

        if not self._mqtt.wait_for_acks(timeout=120):
            print(f"WARNING: Timeout waiting for cleanup ACKs. "
                  f"Received from: {list(self._mqtt._acks.keys())}")

        client.disconnect()
        client.loop_stop()
        print("Cleanup complete. All device servers killed.")
        self._runtime_start_epoch = None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def set_plan(self, plan: dict):
        """Store plan reference (needed by run_inference / cleanup)."""
        self._plan = plan

    def current_runtime_elapsed(self, fallback: float = 0.0) -> float:
        """Return elapsed experiment time based on runtime start."""
        if self._runtime_start_epoch is None:
            return fallback
        return max(fallback, time.time() - self._runtime_start_epoch)

    @property
    def mqtt(self) -> MQTTManager:
        """Expose underlying MQTTManager for server.py ACK wait helpers."""
        return self._mqtt
