"""Configuration for deployment schedulers.

This module defines configuration classes and constants used by the
scheduling algorithms.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SchedulerConfig:
    """Configuration for deployment schedulers.
    
    Attributes:
        util_factor: Maximum utilization factor for devices (0.0-1.0).
                    Devices won't be loaded beyond this threshold.
        base_port: Starting port number for deployments.
        port_increment: Port increment between deployments on same device.
        demand_epsilon: Threshold for considering demand satisfied.
                       Demand below this value is considered fully satisfied.
        max_fit_attempts: Maximum number of backbone downsizing attempts
                         in the fit() algorithm.
    """
    util_factor: float = 0.8
    base_port: int = 8000
    port_increment: int = 10
    demand_epsilon: float = 1e-9
    max_fit_attempts: int = 1


@dataclass 
class TaskConstraints:
    """Constraints for task deployment.
    
    Attributes:
        metric_type: Type of metric ('accuracy' or 'mae').
        metric_threshold: Required metric value.
        max_latency: Maximum acceptable latency in ms.
    """
    metric_type: str  # 'accuracy' or 'mae'
    metric_threshold: float
    max_latency: float
    
    def satisfies_metric(self, value: float) -> bool:
        """Check if a metric value satisfies the constraint.
        
        Args:
            value: Metric value to check.
            
        Returns:
            True if constraint is satisfied.
        """
        if self.metric_type == 'accuracy':
            return value >= self.metric_threshold
        else:  # mae
            return value <= self.metric_threshold
    
    def satisfies_latency(self, latency: float) -> bool:
        """Check if a latency value satisfies the constraint.
        
        Args:
            latency: Latency value in ms.
            
        Returns:
            True if latency is acceptable.
        """
        return latency <= self.max_latency


# Default configuration instance
DEFAULT_CONFIG = SchedulerConfig()
