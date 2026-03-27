"""LocalSiteManager — single-process site manager, no MQTT.

Orchestrator and site managers run in the same process.
Deployment diffs are executed by direct function calls (blocking).
Requests are dispatched live by TraceRunner at scheduled times — no upfront queuing.
"""

import asyncio
import json
import os
import time

from site_manager.base import BaseSiteManager
from site_manager.deployment_handler import (
    _add_decoder_to_device,
    _swap_backbone_on_device,
    deploy_models,
    shutdown_devices,
)


class LocalSiteManager(BaseSiteManager):
    """Executes all site manager operations in-process, without MQTT.

    Key behaviour:
    - deploy()     : calls deploy_models() directly (blocking until hardware ready)
    - apply_diff() : calls deployment_handler functions directly (blocking until done)
    - cleanup()    : calls shutdown_devices() directly

    Request dispatch is handled entirely by TraceRunner (client/runner.py),
    which routes lazily against live_plan at each request's scheduled time.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.live_plan: dict = {}   # shared dict; updated by deploy() and apply_diff()
        self._runtime_start_epoch: float = None

    # ------------------------------------------------------------------ #
    # deploy                                                               #
    # ------------------------------------------------------------------ #

    def deploy(self, plan: dict, output_dir: str = None):
        """Deploy all devices directly (blocking). Requests are not queued here —
        TraceRunner dispatches them live at scheduled times."""
        self.live_plan.update(plan)
        if output_dir:
            self.output_dir = output_dir

        print("[LocalSiteManager] Deploying models...")
        for site in plan["sites"]:
            deployment_status = asyncio.run(deploy_models(site["deployments"]))
            _output_dir = output_dir or self.output_dir
            if _output_dir:
                os.makedirs(_output_dir, exist_ok=True)
                json_path = os.path.join(_output_dir, "model_deployment_results.json")
                with open(json_path, "w") as f:
                    json.dump(deployment_status, f, indent=4)
                print(f"[LocalSiteManager] Saved deployment results to {json_path}")
        print("[LocalSiteManager] Deployment complete.")

    # ------------------------------------------------------------------ #
    # apply_diff                                                           #
    # ------------------------------------------------------------------ #

    def apply_diff(self, diff) -> None:
        """Execute a deployment diff synchronously, then update live_plan.

        Because this blocks until the hardware operation completes, the
        live_plan update happens only after the new device is ready.
        The inference loop's lazy routing will then naturally send
        subsequent requests to the new device.
        """
        print(f"[LocalSiteManager] Applying diff: {diff.action} on {diff.site_manager}")

        if diff.action == "add_full" and diff.full_deployment:
            deployment_status = asyncio.run(deploy_models([diff.full_deployment]))
            _append_deployment_results(self.output_dir, deployment_status)

        elif diff.action == "add_decoder":
            result = asyncio.run(_add_decoder_to_device(diff.ip, diff.new_decoders))
            _append_deployment_results(self.output_dir, [result])

        elif diff.action == "migrate" and diff.full_deployment:
            spec = diff.full_deployment
            result = asyncio.run(
                _swap_backbone_on_device(spec["device"], spec["backbone"], spec.get("decoders", []))
            )
            _append_deployment_results(self.output_dir, [result])

        else:
            print(f"[LocalSiteManager] Unknown or incomplete diff action: {diff.action}, skipping.")
            return

        print(f"[LocalSiteManager] Diff {diff.action} complete.")

    # ------------------------------------------------------------------ #
    # cleanup                                                              #
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """Shut down all device servers directly."""
        specs = []
        for site in self.live_plan.get("sites", []):
            specs.extend(site.get("deployments", []))
        if specs:
            asyncio.run(shutdown_devices(specs))
        self.live_plan.clear()
        self._runtime_start_epoch = None
        print("[LocalSiteManager] Cleanup complete.")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def current_runtime_elapsed(self, fallback: float = 0.0) -> float:
        if self._runtime_start_epoch is None:
            return fallback
        return max(fallback, time.time() - self._runtime_start_epoch)


# ------------------------------------------------------------------ #
# Module-level helper                                                  #
# ------------------------------------------------------------------ #

def _append_deployment_results(output_dir: str, entries: list):
    """Append deployment status entries to model_deployment_results.json."""
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "model_deployment_results.json")
    existing = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)
    existing.extend(entries)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=4)
