"""Data access layer for profiler data.

This module provides a clean abstraction over the raw profiler data,
offering indexed lookups and type-safe access methods.
"""

from typing import Dict, List, Optional, Tuple
from .models import Component, Pipeline


class BatchProfile:
    """Per-backbone/device/tpc batch-size profile sourced from `latency` dict.

    The `latency` dict (from parser.profiler) is keyed as:
        latency[pid][device][tpc][batch_size] = {
            'avg_latency_ms', 'backbone_mean_ms',
            'backbone_ms_per_sample', 'throughput_rps'
        }

    BatchProfile flattens this by (backbone, device, tpc) for batch-size lookups
    used by batching-aware schedulers.
    """

    def __init__(
        self,
        latency: Optional[Dict] = None,
        pipelines: Optional[Dict] = None,
    ):
        # (backbone, device, tpc) -> list of (batch_size, row_dict) sorted by batch_size
        self._index: Dict[Tuple[str, str, int], List[Tuple[int, Dict]]] = {}
        if latency is not None and pipelines is not None:
            self._load(latency, pipelines)

    def _load(self, latency: Dict, pipelines: Dict) -> None:
        # Aggregate across pipelines: first-seen wins per (backbone, device, tpc, bs).
        # All pipelines sharing a backbone see the same backbone_mean_ms at a given bs.
        for pid, dev_map in latency.items():
            pipe = pipelines.get(pid)
            if not pipe:
                continue
            backbone = pipe['backbone']
            for device, tpc_map in dev_map.items():
                for tpc, bs_map in tpc_map.items():
                    key = (backbone, device, int(tpc))
                    existing_bs = {bs for bs, _ in self._index.get(key, [])}
                    for bs, row in bs_map.items():
                        if bs in existing_bs:
                            continue
                        self._index.setdefault(key, []).append((int(bs), row))
                        existing_bs.add(bs)

        for key in self._index:
            self._index[key].sort(key=lambda x: x[0])

    def _max_tpc(self, backbone: str, device: str) -> Optional[int]:
        """Largest tpc profiled for (backbone, device)."""
        tpcs = [t for (b, d, t) in self._index.keys() if b == backbone and d == device]
        return max(tpcs) if tpcs else None

    def _resolve_tpc(self, backbone: str, device: str, tpc: Optional[int]) -> int:
        if tpc is not None:
            return int(tpc)
        resolved = self._max_tpc(backbone, device)
        if resolved is None:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}'")
        return resolved

    def get_saturation_batch_size(
        self,
        backbone: str,
        device: str,
        tpc: Optional[int] = None,
        threshold: float = 0.05,
    ) -> Tuple[int, float]:
        """Find the batch size where backbone_ms_per_sample saturates.

        Saturation = first batch size where the relative decrease in
        backbone_ms_per_sample drops below `threshold` (default 5%).

        Returns:
            (saturation_batch_size, backbone_mean_ms at that batch size)
        """
        tpc = self._resolve_tpc(backbone, device, tpc)
        rows = self._index.get((backbone, device, tpc), [])
        if not rows:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}', tpc={tpc}")

        prev_bs, prev_row = rows[0]
        for bs, row in rows[1:]:
            rel_drop = (prev_row['backbone_ms_per_sample'] - row['backbone_ms_per_sample']) / prev_row['backbone_ms_per_sample']
            print(f"Checking batch_size={bs}: prev_ms_per_sample={prev_row['backbone_ms_per_sample']:.4f}, current_ms_per_sample={row['backbone_ms_per_sample']:.4f}, relative_drop={rel_drop:.4f}")
            print(f"  Threshold={threshold:.4f} -> {'Saturated' if rel_drop < threshold else 'Not saturated'}")
            if rel_drop < threshold:
                return bs, row['backbone_mean_ms']
            prev_bs, prev_row = bs, row

        last_bs, last_row = rows[-1]
        return last_bs, last_row['backbone_mean_ms']

    def get_backbone_mean_ms(
        self, backbone: str, device: str, batch_size: int, tpc: Optional[int] = None,
    ) -> float:
        """Get backbone_mean_ms for the largest profiled batch_size <= requested."""
        tpc = self._resolve_tpc(backbone, device, tpc)
        rows = self._index.get((backbone, device, tpc), [])
        if not rows:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}', tpc={tpc}")

        best = rows[0]
        for bs, row in rows:
            if bs <= batch_size:
                best = (bs, row)
            else:
                break
        return best[1]['backbone_mean_ms']

    def get_max_batch_size(
        self, backbone: str, device: str, tpc: Optional[int] = None,
    ) -> int:
        tpc = self._resolve_tpc(backbone, device, tpc)
        rows = self._index.get((backbone, device, tpc), [])
        if not rows:
            raise KeyError(f"No batch profile for backbone='{backbone}', device='{device}', tpc={tpc}")
        return rows[-1][0]

    def get_available_batch_sizes(
        self, backbone: str, device: str, tpc: Optional[int] = None,
    ) -> List[int]:
        try:
            tpc = self._resolve_tpc(backbone, device, tpc)
        except KeyError:
            return []
        return [bs for bs, _ in self._index.get((backbone, device, tpc), [])]

    def snap_to_profile(
        self, backbone: str, device: str, target_bs: float, tpc: Optional[int] = None,
    ) -> int:
        sizes = self.get_available_batch_sizes(backbone, device, tpc)
        if not sizes:
            return 1
        best = sizes[0]
        for s in sizes:
            if s <= target_bs:
                best = s
            else:
                break
        return best

    def snap_ceil_to_profile(
        self, backbone: str, device: str, target_bs: float, tpc: Optional[int] = None,
    ) -> int:
        sizes = self.get_available_batch_sizes(backbone, device, tpc)
        if not sizes:
            return 1
        for s in sizes:
            if s >= target_bs:
                return s
        return sizes[-1]


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
    
    def get_pipeline_latency(
        self,
        pid: str,
        device_type: str,
        tpc: Optional[int] = None,
        batch_size: int = 1,
    ) -> Optional[float]:
        """Get avg_latency_ms for a pipeline on a device at a given tpc/batch_size.

        Args:
            pid: Pipeline ID.
            device_type: GPU type (e.g., 'NVIDIA A100').
            tpc: TPC count. If None, uses the largest profiled tpc.
            batch_size: Batch size to look up. Snaps down to the largest
                profiled batch_size <= requested.

        Returns:
            avg_latency_ms, or None if not available.
        """
        dev_map = self._latency.get(pid, {}).get(device_type)
        if not dev_map:
            return None

        if tpc is None:
            tpc = max(dev_map.keys())
        bs_map = dev_map.get(int(tpc))
        if not bs_map:
            return None

        # Snap down to largest profiled bs <= requested
        available = sorted(bs_map.keys())
        chosen = available[0]
        for bs in available:
            if bs <= batch_size:
                chosen = bs
            else:
                break
        row = bs_map[chosen]
        return row['avg_latency_ms'] if isinstance(row, dict) else float(row)
    
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
