# device/model_loader.py
import gc
import torch

from fmtk.pipeline import Pipeline
from fmtk.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
from fmtk.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP
from fmtk.components.decoders.classification.linear import LinearDecoder as ClassificationLinear
from fmtk.components.decoders.forecasting.mlp import MLPDecoder as ForecastingMLP
from fmtk.logger import Logger
from device.config import DEVICE, DECODERS, ADAPTERS
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.backbones.dinov2 import DinoV2Model
from fmtk.components.backbones.swin import SwinModel
from fmtk.components.backbones.mae import MAEModel
from fmtk.components.backbones.resnet import ResNetVisionModel
from fmtk.components.backbones.papagei import PapageiModel
from fmtk.components.backbones.phi import PhiModel


def _build_pipeline(backbone: str, device, logger: Logger | None, model_config: dict | None = None) -> Pipeline:
    """Instantiate and return a Pipeline. Logger is passed in so Pipeline.add_decoder,
    forward, etc. all record into the same Logger automatically."""
    if backbone in ("momentlarge", "momentbase", "momentsmall"):
        return Pipeline(MomentModel(device, backbone.replace("moment", "")), logger=logger)
    elif backbone in ("chronostiny", "chronosmini", "chronossmall", "chronosbase", "chronoslarge"):
        return Pipeline(ChronosModel(device, backbone.replace("chronos", "")), logger=logger)
    elif backbone == "papageis":
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512, "n_experts": 3}
        return Pipeline(PapageiModel(device, "papagei_s", model_config=cfg), logger=logger)
    elif backbone == "papageip":
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512}
        return Pipeline(PapageiModel(device, "papagei_p", model_config=cfg), logger=logger)
    elif backbone == "papageissvri":
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512}
        return Pipeline(PapageiModel(device, "papagei_s_svri", model_config=cfg), logger=logger)
    elif backbone in ("dinosmall", "dinobase", "dinolarge", "dinogiant"):
        size = backbone.replace("dino", "")  # small, base, large, giant
        return Pipeline(DinoV2Model(device, size, model_config=model_config or {}), logger=logger)
    elif backbone in ("swintiny", "swinsmall", "swinbase", "swinlarge"):
        size = backbone.replace("swin", "")  # tiny, small, base, large
        return Pipeline(SwinModel(device, size, model_config=model_config or {}), logger=logger)
    elif backbone in ("maebase", "maelarge", "maehuge"):
        size = backbone.replace("mae", "")  # base, large, huge
        return Pipeline(MAEModel(device, size, model_config=model_config or {}), logger=logger)
    elif backbone in ("vgg11", "vgg13", "vgg16", "vgg19"):
        from fmtk.components.backbones.vgg import VGGModel
        return Pipeline(VGGModel(device, backbone, model_config=model_config or {}), logger=logger)
    elif backbone in ("resnet18", "resnet34", "resnet50", "resnet101"):
        return Pipeline(ResNetVisionModel(device, backbone, model_config=model_config or {}), logger=logger)
    elif backbone in ("qwen-2B", "qwen-3B", "qwen-7B"):
        from fmtk.components.backbones.qwen import QwenModel
        return Pipeline(QwenModel(device, backbone, model_config=model_config), logger=logger)
    elif backbone in ("phi3-mini", "phi3-small", "phi3-medium"):
        from fmtk.components.backbones.phi3_vllm import Phi3VLLMModel
        return Pipeline(Phi3VLLMModel(device, backbone, model_config=model_config), logger=logger)
    elif backbone in ("phi"):
        return Pipeline(PhiModel(device, backbone, model_config=model_config), logger=logger)
    elif backbone in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"):
        from fmtk.components.backbones.qwen_vllm import QwenVLLMModel
        return Pipeline(QwenVLLMModel(device, backbone, model_config=model_config, async_only=True), logger=logger)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def _build_adapter_cfg(adapter_type: str):
    """Return the peft LoraConfig instance for a given adapter type."""
    from peft import LoraConfig
    cfg = ADAPTERS[adapter_type]['adapter_config']
    return LoraConfig(**cfg)


def _build_decoder(backbone: str, task: str, dtype: str, device):
    if dtype == "regression":
        cfg = DECODERS[f"mlp_{backbone}_regression"]["decoder_config"]["cfg"]
        return RegressionMLP(device=device, cfg=cfg)
    elif dtype == "classification":
        cfg = DECODERS[f"mlp_{backbone}_{task}"]["decoder_config"]["cfg"]
        return ClassificationMLP(device=device, cfg=cfg)
    elif dtype == "linear_classification":
        cfg = DECODERS[f"linear_{backbone}_{task}"]["decoder_config"]["cfg"]
        return ClassificationLinear(device=device, cfg=cfg)
    elif dtype == "forecasting":
        cfg = DECODERS[f"mlp_{backbone}_forecasting"]["decoder_config"]["cfg"]
        return ForecastingMLP(device=device, cfg=cfg)
    else:
        raise ValueError(f"Unknown decoder type: {dtype}")


class ModelLoader:
    """
    Loads backbone + decoders onto a device. Accepts a Logger in __init__ so
    all operations (load_backbone, add_decoder, load_decoder) record into the
    same Logger automatically via the Pipeline. No manual measure() wrappers
    needed here — Pipeline.add_decoder already handles them.
    """

    def __init__(self, device=None, logger: Logger | None = None):
        self.device        = device or DEVICE
        self.logger        = logger   # shared Logger; None = no measurement
        self.pipeline: Pipeline | None = None
        self.decoders: dict[str, str] = {}
        self.adapters: dict[str, str] = {}  # task -> adapter_name (or None if no adapter)
        self.backbone_name: str | None = None
        self._loaded       = False

    def _op_logger(self) -> Logger:
        """Create a fresh Logger for one operation. Pipeline will record into it,
        and we copy records into self.logger (the runtime's shared logger) after."""
        return Logger(self.device, "deployment")

    def _merge(self, op_logger: Logger):
        """Copy op_logger records into the shared runtime logger (if set)."""
        if self.logger is not None:
            self.logger.records.extend(op_logger.records)

    def load_models(self, backbone: str, task_specs: list, model_config: dict | None = None) -> Logger:
        """Load backbone + task components. Each spec may have decoder and/or adapter fields.

        Decoder-only spec:  {"task": "ecgclass", "type": "classification", "path": "..."}
        Adapter-only spec:  {"task": "ecgclass", "adapter": "lora", "path": "..."}
        Both:               {"task": "ecgclass", "type": "classification", "path": "...", "adapter": "lora"}
        """
        op_log = self._op_logger()
        print(f"[ModelLoader] Loading backbone: {backbone}")
        with op_log.measure("load_backbone", device=self.device):
            self.pipeline = _build_pipeline(backbone, self.device, op_log, model_config=model_config)
        self.backbone_name = backbone
        self.decoders = {}
        self.adapters = {}
        self.pipeline.logger = op_log
        for spec in task_specs:
            task, path = spec["task"], spec["path"]
            dtype = spec.get("type")
            adapter_type = spec.get("adapter")
            print(f"[ModelLoader] Loading task: {task} (decoder={dtype}, adapter={adapter_type}) from {path}")
            if dtype:
                decoder_obj = _build_decoder(backbone, task, dtype, self.device)
                self.decoders[task] = self.pipeline.add_decoder(decoder_obj, load=True, train=False, path=path)
            if adapter_type:
                peft_cfg = _build_adapter_cfg(adapter_type)
                self.adapters[task] = self.pipeline.add_adapter(peft_cfg, load=True, train=False, path=path)
        self.pipeline.logger = self.logger
        self._loaded = True
        print(f"[ModelLoader] Loaded {self.pipeline.model_instance.__class__.__name__} "
              f"with {len(self.decoders)} decoder(s), {sum(v is not None for v in self.adapters.values())} adapter(s).")
        self._merge(op_log)
        return op_log

    def add_decoder(self, decoder_specs: list) -> Logger:
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("No backbone loaded. Call load_models() first.")
        op_log = self._op_logger()
        self.pipeline.logger = op_log
        for dec in decoder_specs:
            task, dtype, path = dec["task"], dec["type"], dec["path"]
            print(f"[ModelLoader] Hot-adding decoder: {task} ({dtype}) from {path}")
            decoder_obj = _build_decoder(self.backbone_name, task, dtype, self.device)
            self.decoders[task] = self.pipeline.add_decoder(decoder_obj, load=True, train=False, path=path)
        self.pipeline.logger = self.logger
        print(f"[ModelLoader] Hot-added {len(decoder_specs)} decoder(s). Total: {len(self.decoders)}")
        self._merge(op_log)
        return op_log

    def add_adapter(self, adapter_specs: list) -> Logger:
        """Hot-add adapters (LoRA) to the loaded backbone.

        Each spec: {"task": "ecgclass", "adapter": "lora", "path": "ecgclass_momentbase_mlp_lora"}
        """
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("No backbone loaded. Call load_models() first.")
        op_log = self._op_logger()
        self.pipeline.logger = op_log
        for spec in adapter_specs:
            task, adapter_type, path = spec["task"], spec["adapter"], spec["path"]
            print(f"[ModelLoader] Hot-adding adapter: {task} ({adapter_type}) from {path}")
            peft_cfg = _build_adapter_cfg(adapter_type)
            self.adapters[task] = self.pipeline.add_adapter(peft_cfg, load=True, train=False, path=path)
        self.pipeline.logger = self.logger
        print(f"[ModelLoader] Hot-added {len(adapter_specs)} adapter(s). Total adapters: "
              f"{sum(v is not None for v in self.adapters.values())}")
        self._merge(op_log)
        return op_log

    def swap_backbone(self, backbone: str, task_specs: list) -> Logger:
        if self.pipeline is not None:
            print(f"[ModelLoader] Releasing '{self.backbone_name}' from GPU memory...")
            del self.pipeline
            self.pipeline = None
        self.decoders = {}
        self.adapters = {}
        self._loaded = False
        self.backbone_name = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[ModelLoader] GPU memory released. Loading new backbone...")
        return self.load_models(backbone, task_specs)
