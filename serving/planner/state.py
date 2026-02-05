"""Deployment state management.

This module provides classes for managing the state of deployments
without relying on global variables.
"""

from typing import Dict, List, Optional, Tuple
from .models import Server, Deployment, TaskInfo


class DeploymentState:
    """Manages deployment state without global variables.
    
    This class maintains the state of all servers and deployments,
    providing methods for querying and modifying the deployment plan.
    
    Attributes:
        servers: Dictionary mapping server names to Server objects.
    """
    
    def __init__(self, servers: List[Server]):
        """Initialize deployment state with available servers.
        
        Args:
            servers: List of available Server objects.
        """
        self._servers: Dict[str, Server] = {s.name: s for s in servers}
        self._deployments: Dict[Tuple[str, str], Deployment] = {}
        self._port_allocations: Dict[str, int] = {}  # ip -> max_port
    
    # --- Server Access ---
    
    def get_server(self, name: str) -> Optional[Server]:
        """Get a server by name."""
        return self._servers.get(name)
    
    def get_all_servers(self) -> List[Server]:
        """Get list of all servers."""
        return list(self._servers.values())
    
    def get_available_servers(self, max_util: float = 1.0) -> List[Server]:
        """Get servers with utilization below threshold.
        
        Args:
            max_util: Maximum utilization threshold.
            
        Returns:
            List of servers with util < max_util.
        """
        return [s for s in self._servers.values() if s.util < max_util]
    
    def get_servers_with_memory(self, min_mem: float) -> List[Server]:
        """Get servers with at least the specified memory.
        
        Args:
            min_mem: Minimum memory requirement in MB.
            
        Returns:
            List of servers with mem >= min_mem.
        """
        return [s for s in self._servers.values() if s.mem >= min_mem]
    
    # --- Deployment Access ---
    
    def get_deployment(self, server_name: str, backbone: str) -> Optional[Deployment]:
        """Get a deployment by server and backbone.
        
        Args:
            server_name: Name of the server.
            backbone: Name of the backbone.
            
        Returns:
            Deployment object if found, None otherwise.
        """
        return self._deployments.get((server_name, backbone))
    
    def get_all_deployments(self) -> List[Deployment]:
        """Get list of all deployments."""
        return list(self._deployments.values())
    
    def find_active_deployments(self, backbone: str, max_util: float) -> List[Deployment]:
        """Find deployments using a specific backbone with capacity.
        
        Args:
            backbone: Name of the backbone.
            max_util: Maximum utilization threshold.
            
        Returns:
            List of deployments using the backbone with util < max_util.
        """
        return [
            d for d in self._deployments.values()
            if d.backbone == backbone and d.util < max_util
        ]
    
    def find_deployments_by_server(self, server_name: str) -> List[Deployment]:
        """Find all deployments on a specific server.
        
        Args:
            server_name: Name of the server.
            
        Returns:
            List of deployments on the server.
        """
        return [
            d for d in self._deployments.values()
            if d.server_name == server_name
        ]
    
    # --- Port Management ---
    
    def get_next_port(self, ip: str, base_port: int = 8000, increment: int = 10) -> int:
        """Get next available port for an IP address.
        
        Calculates max port from CURRENT deployments (like legacy), not a running max.
        This ensures deleted deployments' ports can be reused.
        
        Args:
            ip: IP address (without port).
            base_port: Starting port number.
            increment: Port increment value.
            
        Returns:
            Next available port number.
        """
        # Calculate max port from current deployments (legacy behavior)
        max_port = base_port - increment  # So first deployment gets base_port
        
        for deployment in self._deployments.values():
            if ':' in deployment.ip:
                dep_ip, dep_port = deployment.ip.rsplit(':', 1)
                if dep_ip == ip:
                    try:
                        max_port = max(max_port, int(dep_port))
                    except ValueError:
                        pass
        
        return max_port + increment
    
    def get_endpoint_for_backbone(
        self, 
        server_name: str, 
        backbone: str,
        base_port: int = 8000,
        increment: int = 10
    ) -> str:
        """Get or create endpoint for a server/backbone combination.
        
        Args:
            server_name: Name of the server.
            backbone: Name of the backbone.
            base_port: Starting port number.
            increment: Port increment value.
            
        Returns:
            Full endpoint string (ip:port).
        """
        # Check if deployment already exists
        existing = self.get_deployment(server_name, backbone)
        if existing:
            return existing.ip
        
        # Get server IP and allocate new port
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")
        
        port = self.get_next_port(server.ip, base_port, increment)
        return f"{server.ip}:{port}"
    
    # --- Deployment Modification ---
    
    def add_deployment(
        self, 
        deployment: Deployment,
        base_port: int = 8000,
        port_increment: int = 10
    ) -> None:
        """Add or update a deployment.
        
        Args:
            deployment: Deployment object to add.
            base_port: Base port for new deployments.
            port_increment: Port increment for new deployments.
        """
        key = (deployment.server_name, deployment.backbone)
        
        existing = self._deployments.get(key)
        if existing:
            # Merge components
            existing.components.update(deployment.components)
            # Merge task_info
            for task_name, task_info in deployment.task_info.items():
                if task_name in existing.task_info:
                    existing.task_info[task_name].request_per_sec += task_info.request_per_sec
                else:
                    existing.task_info[task_name] = task_info
            existing.util = deployment.util
        else:
            # Allocate port only when committing (not during temp plan creation)
            # This prevents consuming ports for rolled-back plans
            if ':' not in deployment.ip:
                # IP without port - allocate one now based on current deployments
                port = self.get_next_port(deployment.ip, base_port, port_increment)
                deployment.ip = f"{deployment.ip}:{port}"
            # Note: get_next_port calculates from current deployments, so no tracking needed
            
            self._deployments[key] = deployment
        
        # Sync utilization across all deployments on this server
        self._sync_server_utilization(deployment.server_name, deployment.util)
    
    def remove_deployment(self, server_name: str, backbone: str) -> Optional[Deployment]:
        """Remove a deployment.
        
        Args:
            server_name: Name of the server.
            backbone: Name of the backbone.
            
        Returns:
            Removed Deployment object, or None if not found.
        """
        key = (server_name, backbone)
        return self._deployments.pop(key, None)
    
    def update_deployment_backbone(
        self, 
        server_name: str, 
        old_backbone: str, 
        new_backbone: str,
        new_components: Dict[str, float],
        new_util: float
    ) -> Optional[Deployment]:
        """Update a deployment with a new backbone.
        
        Args:
            server_name: Name of the server.
            old_backbone: Current backbone name.
            new_backbone: New backbone name.
            new_components: New components dictionary.
            new_util: New utilization value.
            
        Returns:
            Updated Deployment object, or None if not found.
        """
        old_deployment = self.remove_deployment(server_name, old_backbone)
        if not old_deployment:
            return None
        
        # Get base IP without port - legacy allocates new port for new backbone
        base_ip = old_deployment.ip.split(':')[0] if ':' in old_deployment.ip else old_deployment.ip
        
        new_deployment = Deployment(
            server_name=server_name,
            backbone=new_backbone,
            ip=base_ip,  # Don't preserve port - let add_deployment allocate new one
            site_manager=old_deployment.site_manager,
            device_type=old_deployment.device_type,
            mem=old_deployment.mem,
            util=new_util,
            components=new_components,
            task_info=old_deployment.task_info
        )
        self.add_deployment(new_deployment)
        return new_deployment
    
    # --- Utilization Management ---
    
    def _sync_server_utilization(self, server_name: str, util: float) -> None:
        """Sync utilization across server and all its deployments.
        
        Args:
            server_name: Name of the server.
            util: New utilization value.
        """
        # Update server
        if server_name in self._servers:
            self._servers[server_name].util = util
        
        # Update all deployments on this server
        for deployment in self.find_deployments_by_server(server_name):
            deployment.util = util
    
    def get_server_utilization(self, server_name: str) -> float:
        """Get current utilization of a server.
        
        Args:
            server_name: Name of the server.
            
        Returns:
            Current utilization (0.0 to 1.0).
        """
        server = self._servers.get(server_name)
        return server.util if server else 0.0
    
    def calculate_device_utilization(self, server_name: str) -> float:
        """Calculate total utilization from all deployments on a server.
        
        Args:
            server_name: Name of the server.
            
        Returns:
            Calculated total utilization.
        """
        deployments = self.find_deployments_by_server(server_name)
        if not deployments:
            return 0.0
        # All deployments on same server should have same util
        return deployments[0].util if deployments else 0.0
    
    # --- Utility Methods ---
    
    def get_deployment_count(self) -> int:
        """Get total number of deployments."""
        return len(self._deployments)
    
    def get_deployment_with_min_tasks(self) -> Optional[Deployment]:
        """Get the deployment with minimum number of tasks.
        
        Returns:
            Deployment with fewest tasks, or None if no deployments.
        """
        if not self._deployments:
            return None
        return min(self._deployments.values(), key=lambda d: len(d.task_info))
    
    def to_legacy_format(self) -> List[Dict]:
        """Convert deployments to legacy list-of-dicts format.
        
        Returns:
            List of single-item dictionaries in legacy format.
        """
        result = []
        for (server_name, backbone), deployment in self._deployments.items():
            # Convert TaskInfo objects to dicts
            task_info_dict = {}
            for task_name, info in deployment.task_info.items():
                task_info_dict[task_name] = {
                    'type': info.type,
                    'total_requested_workload': info.total_requested_workload,
                    'request_per_sec': info.request_per_sec
                }
            
            result.append({
                (server_name, backbone): {
                    'name': deployment.server_name,
                    'backbone': deployment.backbone,
                    'site_manager': deployment.site_manager,
                    'type': deployment.device_type,
                    'mem': deployment.mem,
                    'components': deployment.components,
                    'task_info': task_info_dict,
                    'ip': deployment.ip,
                    'util': deployment.util
                }
            })
        return result
