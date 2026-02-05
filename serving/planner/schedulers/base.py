"""Abstract base class for deployment schedulers.

This module defines the interface that all scheduler implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..models import Deployment, Server
from ..data_loader import ProfileData
from ..config import SchedulerConfig, DEFAULT_CONFIG


class BaseScheduler(ABC):
    """Abstract base class for deployment schedulers.
    
    All scheduler implementations should inherit from this class and
    implement the `schedule` method.
    
    Attributes:
        data: ProfileData instance with component/pipeline information.
        config: SchedulerConfig with scheduling parameters.
    """
    
    def __init__(
        self, 
        profile_data: ProfileData, 
        config: Optional[SchedulerConfig] = None
    ):
        """Initialize the scheduler.
        
        Args:
            profile_data: ProfileData instance with profiler information.
            config: Optional SchedulerConfig. Uses DEFAULT_CONFIG if not provided.
        """
        self.data = profile_data
        self.config = config or DEFAULT_CONFIG
    
    @abstractmethod
    def schedule(
        self, 
        devices: Dict[str, Dict], 
        tasks: Dict[str, Dict],
        **kwargs
    ) -> List[Deployment]:
        """Schedule tasks onto devices.
        
        Args:
            devices: Dictionary of device configurations.
                     Keys are device names, values contain:
                     - type: GPU type (e.g., 'A100', 'A16')
                     - mem: Memory capacity in MB
                     - ip: IP address
                     - site_manager: Site manager ID
            tasks: Dictionary of task specifications.
                   Keys are task names, values contain:
                   - type: 'classification' or 'regression'
                   - peak_workload: Maximum requests per second
                   - latency: Maximum acceptable latency (optional)
                   - metric: 'accuracy' or 'mae' (optional)
                   - value: Required metric threshold (optional)
            **kwargs: Additional scheduler-specific parameters.
            
        Returns:
            List of Deployment objects representing the deployment plan.
        """
        pass
    
    def _create_servers(self, devices: Dict[str, Dict]) -> List[Server]:
        """Convert device configuration to Server objects.
        
        Args:
            devices: Dictionary of device configurations.
            
        Returns:
            List of Server objects.
        """
        return [
            Server(
                name=name,
                type=d['type'],
                mem=d['mem'],
                ip=d['ip'],
                site_manager=d['site_manager'],
                util=0.0
            )
            for name, d in devices.items()
        ]
    
    @property
    def name(self) -> str:
        """Return the scheduler name for identification."""
        return self.__class__.__name__
