"""
VLLMRuntime — continuous-batching LLM runtime for the FMaaS device server.

Loads a fmtk vLLM backbone (e.g. Phi3VLLMModel) and delegates inference to
its async_forward() method, which uses AsyncLLMEngine internally for true
continuous batching across concurrent gRPC requests.

Supported backbones:
    phi3-mini, phi3-small, phi3-medium
    qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b
"""

from fmtk.components.backbones.phi3_vllm import Phi3VLLMModel
from fmtk.components.backbones.qwen_vllm import QwenVLLMModel


# Model weight sizes in GB (from vLLM logs: "model weights took X GB").
# These are weight-only sizes, before KV cache pre-allocation.
_WEIGHT_GB = {
    "phi3-mini":    3.8,
    "phi3-small":   7.7,
    "phi3-medium":  14.0,
    "qwen2.5-0.5b": 0.928,
    "qwen2.5-1.5b": 2.77,
    "qwen2.5-3b":   5.54,
    "qwen2.5-7b":   14.2,
}

# Maps backbone name → (fmtk class, model_name string)
# Add entries here as fmtk gains more *_vllm backbone files.
_BACKBONE_REGISTRY = {
    "phi3-mini":     (Phi3VLLMModel, "phi3-mini"),
    "phi3-small":    (Phi3VLLMModel, "phi3-small"),
    "phi3-medium":   (Phi3VLLMModel, "phi3-medium"),
    "qwen2.5-0.5b":  (QwenVLLMModel, "qwen2.5-0.5b"),
    "qwen2.5-1.5b":  (QwenVLLMModel, "qwen2.5-1.5b"),
    "qwen2.5-3b":    (QwenVLLMModel, "qwen2.5-3b"),
    "qwen2.5-7b":    (QwenVLLMModel, "qwen2.5-7b"),
}


class VLLMRuntime:
    """
    Thin wrapper around fmtk vLLM backbones that exposes the same
    load() / async infer() interface as SharedModelRuntime.

    Continuous batching is handled inside fmtk's Phi3VLLMModel.async_forward()
    via AsyncLLMEngine — no batching logic needed here.
    """

    def __init__(self):
        self.model = None
        self.backbone: str | None = None
        self.memory_stats: dict | None = None

    def load(self, backbone: str, decoders: list, device: str = "cuda:0",
             model_config: dict | None = None):
        """
        Instantiate the fmtk vLLM backbone.

        decoders is accepted for interface compatibility but unused — LLMs
        have no task-specific decoder heads.
        """
        entry = _BACKBONE_REGISTRY.get(backbone)
        if entry is None:
            raise ValueError(
                f"[VLLMRuntime] Unknown backbone '{backbone}'. "
                f"Supported: {list(_BACKBONE_REGISTRY)}"
            )
        cls, model_name = entry
        print(f"[VLLMRuntime] Loading backbone='{backbone}' on {device}")
        kwargs = dict(device=device, model_name=model_name, model_config=model_config or {})
        if cls is QwenVLLMModel:
            kwargs["async_only"] = True
        self.model = cls(**kwargs)
        self.backbone = backbone

        import torch
        gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
        total_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
        reserved_gb = torch.cuda.memory_reserved(gpu_idx) / 1024**3
        weight_gb = _WEIGHT_GB.get(backbone, 0.0)
        self.memory_stats = {
            "total_gpu_gb": round(total_gb, 3),
            "model_memory_gb": round(weight_gb, 3),
            "reserved_gb": round(reserved_gb, 3),
            "gpu_memory_utilization": round(reserved_gb / total_gb, 4),
        }
        print(f"[VLLMRuntime] Ready. weights={weight_gb:.2f}GB reserved={reserved_gb:.2f}GB "
              f"total={total_gb:.2f}GB util={reserved_gb/total_gb:.3f}")

    async def infer(self, req_id: int, prompt: str) -> dict:
        """
        Run one prompt through async_forward().

        Multiple concurrent calls are batched at the iteration level by
        vLLM's AsyncLLMEngine inside fmtk — true continuous batching.
        """
        if self.model is None:
            raise RuntimeError("vllm_model_not_loaded")
        return await self.model.async_forward(prompt)
