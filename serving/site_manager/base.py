"""Abstract interface for site manager implementations.

Two concrete implementations:
  - MQTTSiteManager  (serving/site_manager/mqtt.py)  — geo-distributed, broker-mediated
  - LocalSiteManager (serving/site_manager/local.py)  — single-process, direct calls

Common interface: deploy, apply_diff, cleanup.
MQTT-only: run_inference, wait_for_completion, send_requests (not needed in local mode
since TraceRunner handles dispatch directly).
"""

from abc import ABC, abstractmethod


class BaseSiteManager(ABC):
    """Common interface for all site manager transport backends."""

    @abstractmethod
    def deploy(self, plan: dict, output_dir: str):
        """Initial deployment: load models on all devices.

        Blocks until all devices have confirmed they are ready.
        """
        ...

    @abstractmethod
    def apply_diff(self, diff) -> None:
        """Execute a single deployment diff and block until complete.

        diff is a DeploymentDiff namedtuple/object with fields:
          action         : "add_full" | "add_decoder" | "migrate"
          site_manager   : site id string
          full_deployment: deployment spec dict (for add_full / migrate)
          ip             : device URL (for add_decoder)
          new_decoders   : list of decoder specs (for add_decoder)
          old_backbone   : backbone name being replaced (for migrate)
          backbone       : new backbone name
        """
        ...

    @abstractmethod
    def cleanup(self):
        """Kill all device servers and reset state."""
        ...
