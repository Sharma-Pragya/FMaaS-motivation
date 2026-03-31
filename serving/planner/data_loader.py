"""Data access layer for profiler data.

This module provides a clean abstraction over the raw profiler data,
offering indexed lookups and type-safe access methods.
"""

import csv
import os
from typing import Dict, List, Optional, Tuple
from .models import Component, Pipeline


class BatchProfile:
    """Loads batch_profile.csv and provides lookup methods for batching.

    The profile stores per-backbone, per-device measurements across batch sizes:
    backbone_mean_ms, backbone_ms_per_sample, avg_latency_ms, etc.
    """

    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "parser", "batch_profile.csv")
        self._rows: List[Dict] = []
        # Indexed: (backbone, device) -> list of row dicts sorted by batch_size
        self._index: Dict[Tuple[str, str], List[Dict]] = {}
        self._load(csv_path)

    def _load(self, csv_path: str) -> None:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("backbone"):
                    continue
                parsed = {
                    "backbone": row["backbone"],
                    "task": row["task"],
                    "batch_size": int(row["batch_size"]),
                    "device": row["device"],
                    "backbone_mean_ms": float(row["backbone_mean_ms"]),
                    "backbone_ms_per_sample": float(row["backbone_ms_per_sample"]),
                    "avg_latency_ms": float(row["avg_latency_ms"]),
                    "decoder_mean_ms": float(row["decoder_mean_ms"]),
                    "throughput_rps": float(row["throughput_rps"]),
                }
                self._rows.append(parsed)
                key = (parsed["backbone"], parsed["device"])
                self._index.setdefault(key, []).append(parsed)

        # Sort each group by batch_size
        for key in self._index:
            self._index[key].sort(key=lambda r: r["batch_size"])

    def get_saturation_batch_size(
        self, backbone: str, device: str, threshold: float = 0.05
    ) -> Tuple[int, float]:
        """Find the batch size where backbone_ms_per_sample saturates.

        Saturation = the first batch size where the relative decrease in
        backbone_ms_per_sample drops below `threshold` (default 5%).

        Returns:
            (saturation_batch_size, backbone_mean_ms at that batch size)
        """
        rows = self._index.get((backbone, device), [])
        if not rows:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}'")

        prev = rows[0]
        for row in rows[1:]:
            relative_drop = (prev["backbone_ms_per_sample"] - row["backbone_ms_per_sample"]) / prev["backbone_ms_per_sample"]
            print(f"Checking batch_size={row['batch_size']}: prev_ms_per_sample={prev['backbone_ms_per_sample']:.4f}, current_ms_per_sample={row['backbone_ms_per_sample']:.4f}, relative_drop={relative_drop:.4f}")
            print(f"  Threshold={threshold:.4f} -> {'Saturated' if relative_drop < threshold else 'Not saturated'}" )
            if relative_drop < threshold:
                return row["batch_size"], row["backbone_mean_ms"]
            prev = row

        # Never saturated — return last entry
        last = rows[-1]
        return last["batch_size"], last["backbone_mean_ms"]

    def get_backbone_mean_ms(self, backbone: str, device: str, batch_size: int) -> float:
        """Get backbone_mean_ms for the closest available batch size (floor).

        If the requested batch_size is smaller than the smallest profiled size,
        returns the smallest profiled entry.
        """
        rows = self._index.get((backbone, device), [])
        if not rows:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}'")

        # Find largest profiled batch_size <= requested
        best = rows[0]
        for row in rows:
            if row["batch_size"] <= batch_size:
                best = row
            else:
                break
        return best["backbone_mean_ms"]

    def get_max_batch_size(self, backbone: str, device: str) -> int:
        """Return the largest profiled batch size for a backbone/device."""
        rows = self._index.get((backbone, device), [])
        if not rows:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}'")
        return rows[-1]["batch_size"]

    def get_available_batch_sizes(self, backbone: str, device: str) -> List[int]:
        """Return sorted list of profiled batch sizes."""
        rows = self._index.get((backbone, device), [])
        return [r["batch_size"] for r in rows]

    def snap_to_profile(self, backbone: str, device: str, target_bs: float) -> int:
        """Snap a continuous batch size to the nearest available profiled size (floor)."""
        sizes = self.get_available_batch_sizes(backbone, device)
        if not sizes:
            return 1
        best = sizes[0]
        for s in sizes:
            if s <= target_bs:
                best = s
            else:
                break
        return best


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

    def get_component_mem(self, name: str) -> float:
        """Get memory footprint of a component."""
        return self._raw_components[name]['mem']
    
    def get_component_type(self, name: str) -> Optional[str]:
        """Get the type of a component (for backbones)."""
        return self._raw_components[name].get('type')
    
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

        # VLM pipelines have no separate decoder — profiler stores inference mem
        # under "{task}_{backbone}" (no decoder suffix).
        if pipeline.decoder == 'none':
            task_name = f"{pipeline.task}_{pipeline.backbone}"
            return {
                backbone_name: self.get_component_mem(backbone_name),
                task_name: self.get_component_mem(task_name),
            }

        decoder_name = f"{pipeline.decoder}_{pipeline.backbone}_{pipeline.task}"
        task_name = f"{pipeline.task}_{pipeline.backbone}_{pipeline.decoder}"

        return {
            backbone_name: self.get_component_mem(backbone_name),
            decoder_name: self.get_component_mem(decoder_name),
            task_name: self.get_component_mem(task_name)
        }
    
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
