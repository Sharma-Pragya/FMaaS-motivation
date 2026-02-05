"""Data models for deployment scheduling.

This module defines dataclasses for all domain entities used in the
deployment scheduling algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Component:
    """Represents a model component (backbone, decoder, or task head).
    
    Attributes:
        name: Unique identifier for the component.
        mem: Memory footprint in MB.
        type: Component family type (e.g., 'chronos', 'moment', 'papagei').
    """
    name: str
    mem: float
    type: Optional[str] = None


@dataclass
class Pipeline:
    """Represents a complete inference pipeline.
    
    A pipeline consists of a backbone encoder, decoder, and task-specific head.
    
    Attributes:
        id: Unique pipeline identifier (e.g., 'p1', 'p2').
        backbone: Name of the backbone encoder component.
        decoder: Name of the decoder component.
        task: Name of the task this pipeline serves.
    """
    id: str
    backbone: str
    decoder: str
    task: str


@dataclass
class Server:
    """Represents a compute server/device.
    
    Attributes:
        name: Unique server identifier.
        type: GPU type (e.g., 'NVIDIA A100', 'A16').
        mem: Total memory capacity in MB.
        ip: IP address of the server.
        site_manager: Site manager ID this server belongs to.
        util: Current utilization (0.0 to 1.0).
    """
    name: str
    type: str
    mem: float
    ip: str
    site_manager: str
    util: float = 0.0


@dataclass
class TaskSpec:
    """Specification for a task to be deployed.
    
    Attributes:
        name: Task name (e.g., 'heartrate', 'ecgclass').
        type: Task type ('classification' or 'regression').
        peak_workload: Maximum requests per second required.
        latency: Maximum acceptable latency in ms.
        metric: Metric name ('accuracy' or 'mae').
        value: Required metric threshold.
    """
    name: str
    type: str  # 'classification' or 'regression'
    peak_workload: float
    latency: float
    metric: str  # 'accuracy' or 'mae'
    value: float


@dataclass
class TaskInfo:
    """Information about a deployed task instance.
    
    Attributes:
        type: Task type ('classification' or 'regression').
        total_requested_workload: Original workload requested.
        request_per_sec: Actual workload allocated to this deployment.
    """
    type: str  # 'classification' or 'regression'
    total_requested_workload: float
    request_per_sec: float


@dataclass
class Deployment:
    """Represents a deployment of a backbone and tasks on a server.
    
    Attributes:
        server_name: Name of the server hosting this deployment.
        backbone: Name of the backbone used.
        ip: Full endpoint address (ip:port).
        site_manager: Site manager ID.
        device_type: GPU type of the hosting device.
        mem: Memory capacity of the device.
        util: Current utilization of the device.
        components: Map of component names to their memory usage.
        task_info: Map of task names to their deployment info.
    """
    server_name: str
    backbone: str
    ip: str
    site_manager: str
    device_type: str
    mem: float
    util: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    task_info: Dict[str, TaskInfo] = field(default_factory=dict)

    def get_endpoint_key(self) -> tuple:
        """Returns the unique key for this deployment."""
        return (self.server_name, self.backbone)
