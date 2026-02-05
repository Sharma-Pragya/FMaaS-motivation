"""Data access layer for profiler data.

This module provides a clean abstraction over the raw profiler data,
offering indexed lookups and type-safe access methods.
"""

from typing import Dict, List, Optional, Tuple
from .models import Component, Pipeline


class ProfileData:
    """Encapsulates all profiler data with clean access methods.
    
    This class provides an abstraction layer over the raw profiler dictionaries,
    offering indexed lookups and type-safe access to components, pipelines,
    latency, and metric data.
    
    Attributes:
        components: Mapping of component names to Component objects.
        pipelines: Mapping of pipeline IDs to Pipeline objects.
    """
    
    def __init__(
        self,
        components: Dict,
        pipelines: Dict,
        latency: Dict,
        metric: Dict
    ):
        """Initialize ProfileData with raw profiler dictionaries.
        
        Args:
            components: Raw components dictionary from profiler.
            pipelines: Raw pipelines dictionary from profiler.
            latency: Raw latency dictionary from profiler.
            metric: Raw metric dictionary from profiler.
        """
        self._raw_components = components
        self._raw_pipelines = pipelines
        self._latency = latency
        self._metric = metric
        
        # Build typed objects
        self.components = self._build_components()
        self.pipelines = self._build_pipelines()
        
        # Build indexes for fast lookups
        self._pipeline_by_task_backbone: Dict[Tuple[str, str], str] = {}
        self._pipelines_by_task: Dict[str, List[str]] = {}
        self._backbones_by_type: Dict[str, List[str]] = {}
        self._build_indexes()
    
    def _build_components(self) -> Dict[str, Component]:
        """Convert raw components dict to typed Component objects."""
        result = {}
        for name, data in self._raw_components.items():
            result[name] = Component(
                name=name,
                mem=data['mem'],
                type=data.get('type')
            )
        return result
    
    def _build_pipelines(self) -> Dict[str, Pipeline]:
        """Convert raw pipelines dict to typed Pipeline objects."""
        result = {}
        for pid, data in self._raw_pipelines.items():
            result[pid] = Pipeline(
                id=pid,
                backbone=data['backbone'],
                decoder=data['decoder'],
                task=data['task']
            )
        return result
    
    def _build_indexes(self) -> None:
        """Build lookup indexes for efficient queries."""
        for pid, pipeline in self.pipelines.items():
            # Index by (task, backbone)
            key = (pipeline.task, pipeline.backbone)
            self._pipeline_by_task_backbone[key] = pid
            
            # Index by task
            if pipeline.task not in self._pipelines_by_task:
                self._pipelines_by_task[pipeline.task] = []
            self._pipelines_by_task[pipeline.task].append(pid)
        
        # Index backbones by type
        for name, component in self.components.items():
            if component.type is not None:
                if component.type not in self._backbones_by_type:
                    self._backbones_by_type[component.type] = []
                self._backbones_by_type[component.type].append(name)
    
    # --- Component Access ---
    
    def get_component(self, name: str) -> Optional[Component]:
        """Get a component by name."""
        return self.components.get(name)
    
    def get_component_mem(self, name: str) -> float:
        """Get memory footprint of a component."""
        return self._raw_components[name]['mem']
    
    def get_component_type(self, name: str) -> Optional[str]:
        """Get the type of a component (for backbones)."""
        return self._raw_components[name].get('type')
    
    def is_backbone(self, name: str) -> bool:
        """Check if a component is a backbone."""
        return self.get_component_type(name) is not None
    
    # --- Pipeline Access ---
    
    def get_pipeline(self, pid: str) -> Optional[Pipeline]:
        """Get a pipeline by ID."""
        return self.pipelines.get(pid)
    
    def get_pipeline_latency(self, pid: str, device_type: str) -> Optional[float]:
        """Get latency for a pipeline on a device type.
        
        Args:
            pid: Pipeline ID.
            device_type: GPU type (e.g., 'NVIDIA A100').
            
        Returns:
            Latency in milliseconds, or None if not available.
        """
        return self._latency.get(pid, {}).get(device_type)
    
    def get_pipeline_metric(self, pid: str) -> float:
        """Get the accuracy/MAE metric for a pipeline."""
        return self._metric[pid]
    
    def find_pipeline_id(self, task: str, backbone: str) -> Optional[str]:
        """Find pipeline ID by task and backbone.
        
        Args:
            task: Task name.
            backbone: Backbone name.
            
        Returns:
            Pipeline ID if found, None otherwise.
        """
        return self._pipeline_by_task_backbone.get((task, backbone))
    
    def get_pipelines_for_task(self, task_name: str) -> Dict[str, Pipeline]:
        """Get all pipelines that can serve a task.
        
        Args:
            task_name: Name of the task.
            
        Returns:
            Dictionary mapping pipeline IDs to Pipeline objects.
        """
        result = {}
        for pid in self._pipelines_by_task.get(task_name, []):
            result[pid] = self.pipelines[pid]
        return result
    
    def get_backbones_for_task(self, task_name: str) -> Dict[str, str]:
        """Get all backbones that can serve a task.
        
        Args:
            task_name: Name of the task.
            
        Returns:
            Dictionary mapping pipeline IDs to backbone names.
        """
        result = {}
        for pid in self._pipelines_by_task.get(task_name, []):
            result[pid] = self.pipelines[pid].backbone
        return result
    
    # --- Component Memory Calculations ---
    
    def get_pipeline_components_mem(self, pipeline: Pipeline) -> Dict[str, float]:
        """Get memory footprint for all components in a pipeline.
        
        Args:
            pipeline: Pipeline object.
            
        Returns:
            Dictionary mapping component names to memory in MB.
        """
        backbone_name = pipeline.backbone
        decoder_name = f"{pipeline.decoder}_{pipeline.backbone}_{pipeline.task}"
        task_name = f"{pipeline.task}_{pipeline.backbone}_{pipeline.decoder}"
        
        return {
            backbone_name: self.get_component_mem(backbone_name),
            decoder_name: self.get_component_mem(decoder_name),
            task_name: self.get_component_mem(task_name)
        }
    
    def get_pipeline_components_mem_by_id(self, pid: str) -> Dict[str, float]:
        """Get memory footprint for all components in a pipeline by ID.
        
        Args:
            pid: Pipeline ID.
            
        Returns:
            Dictionary mapping component names to memory in MB.
        """
        pipeline = self.pipelines[pid]
        return self.get_pipeline_components_mem(pipeline)
    
    # --- Backbone Operations ---
    
    def find_smaller_backbone(self, backbone_name: str) -> Optional[str]:
        """Find a smaller backbone of the same type.
        
        Args:
            backbone_name: Current backbone name.
            
        Returns:
            Name of a smaller backbone of the same type, or None if none exists.
        """
        current_mem = self.get_component_mem(backbone_name)
        backbone_type = self.get_component_type(backbone_name)
        
        if backbone_type is None:
            return None
        
        # Find all backbones of the same type that are smaller
        candidates = []
        for name in self._backbones_by_type.get(backbone_type, []):
            if name != backbone_name:
                mem = self.get_component_mem(name)
                if mem < current_mem:
                    candidates.append((mem, name))
        
        if not candidates:
            return None
        
        # Return the largest one that's still smaller than current
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def get_all_backbones(self) -> List[str]:
        """Get list of all backbone names."""
        return [name for name, comp in self.components.items() if comp.type is not None]
