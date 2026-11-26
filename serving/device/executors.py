"""executors.py — what the backbone process asks of its model (backbone_proc.py).

The backbone process owns lanes, pools, messages, the policy and control;
none of that depends on the model. The model is behind an executor:

    kind            "batch": the backbone loop gathers a batch of slots and
                    calls forward() once (vision, time series).
                    "continuous": the backbone loop is device/llm_loop.py; the
                    executor's runtime is stepped (LLMs on vLLM).
    load / swap     build the backbone (returns peak MB allocated)
    output_spec     shape and dtype of one request's output slot (feat[j])
    model_category  reported to the gRPC process at load

PyTorchExecutor is the code backbone_proc.py used inline before (moved, not
changed). VLLMExecutor wraps runtime.VLLMRuntime with the step driver, the
same object shared mode uses, so both isolation modes run one engine setup.
"""
from __future__ import annotations

import gc


class PyTorchExecutor:
    kind = "batch"

    def __init__(self, device: str, logger, stream_ctx, sync):
        self.device = device
        self.logger = logger
        self._stream_ctx = stream_ctx     # backbone stream context (backbone_proc)
        self._sync = sync                 # waits for that stream
        self.pipeline = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def model_category(self) -> str:
        return self.pipeline.model_instance.model_category

    def load(self, backbone: str, payload: dict) -> None:
        from device.model_loader import _build_pipeline
        self.pipeline = _build_pipeline(backbone, self.device, self.logger,
                                        model_config=payload.get("model_config"))

    def swap(self, backbone: str, payload: dict) -> None:
        import torch
        self.pipeline = None
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self.load(backbone, payload)

    def forward(self, bx, bmask):
        return self.pipeline.model_instance.forward(bx, bmask)

    def output_spec(self, x_shape, x_dtype, m_shape, m_dtype):
        """(shape, dtype) of one request's features, from a dry-run forward."""
        import torch
        with self._stream_ctx(), torch.no_grad():
            x = torch.zeros((1, *x_shape), dtype=x_dtype, device=self.device)
            m = (torch.ones((1, *m_shape), dtype=m_dtype, device=self.device)
                 if m_shape is not None else None)
            f = self.forward(x, m)
        self._sync()
        return tuple(f.shape[1:]), f.dtype


class VLLMExecutor:
    """An LLM on vLLM's LLMEngine, stepped by device/llm_loop.py.

    Slot layout (task_proc.py writes x/mask, this process writes feat):
        x[j]     prompt token ids, int32, padded to llm_max_prompt_len
        mask[j]  [prompt length], int32
        feat[j]  [n_generated, id_0, id_1, ...], int32, max_new_tokens + 1 wide
    Every task's LoRA is registered on the engine at load (payload["decoders"],
    the same adapter specs shared mode loads), so the engine is built with
    LoRA enabled and max_loras sized for all of them.
    """
    kind = "continuous"
    model_category = "llm"

    def __init__(self, config, cuda_stream=None):
        from device.runtime import VLLMRuntime
        self.config = config
        self.runtime = VLLMRuntime(cuda_stream=cuda_stream, driver="step")

    @property
    def loaded(self) -> bool:
        return self.runtime.pipeline is not None

    def load(self, backbone: str, payload: dict) -> None:
        from device.server import EdgeRuntimeApplication
        model_config = EdgeRuntimeApplication._vllm_model_config(self.config)
        model_config.update(payload.get("model_config") or {})
        self.runtime.load(backbone, payload.get("decoders", []), model_config=model_config)

    def swap(self, backbone: str, payload: dict) -> None:
        raise RuntimeError("swap_backbone is not supported for vllm in process mode")

    def output_spec(self, x_shape, x_dtype, m_shape, m_dtype):
        import torch
        return (self.runtime.max_new_tokens + 1,), torch.int32

    def tokenizer_name(self) -> str:
        return self.runtime.tokenizer_name()


def make_executor(config, device: str, logger, stream_ctx, sync, bb_stream):
    if config.runtime_type == "vllm":
        if config.vllm_driver != "step":
            raise ValueError("isolation_mode=process with vllm needs --vllm-driver step")
        # TPC masking applies to vLLM only when configured (_setup_tpc made a
        # masked stream); otherwise vLLM keeps its own streams, as in shared mode.
        masked = bb_stream if config.tpc_mode != "none" else None
        return VLLMExecutor(config, cuda_stream=masked)
    return PyTorchExecutor(device, logger, stream_ctx, sync)
