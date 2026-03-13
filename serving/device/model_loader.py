# device/model_loader.py
import gc
import torch

from fmtk.pipeline import Pipeline
from fmtk.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
from fmtk.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP
from fmtk.components.decoders.forecasting.mlp import MLPDecoder as ForecastingMLP
from fmtk.logger import Logger
from device.config import DEVICE, DECODERS


def _build_pipeline(backbone: str, device, logger: Logger | None, model_config: dict | None = None) -> Pipeline:
    """Instantiate and return a Pipeline. Logger is passed in so Pipeline.add_decoder,
    forward, etc. all record into the same Logger automatically."""
    if backbone in ("momentlarge", "momentbase", "momentsmall"):
        from fmtk.components.backbones.moment import MomentModel
        return Pipeline(MomentModel(device, backbone.replace("moment", "")), logger=logger)
    elif backbone in ("chronostiny", "chronosmini", "chronossmall", "chronosbase", "chronoslarge"):
        from fmtk.components.backbones.chronos import ChronosModel
        return Pipeline(ChronosModel(device, backbone.replace("chronos", "")), logger=logger)
    elif backbone == "papageis":
        from fmtk.components.backbones.papagei import PapageiModel
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512, "n_experts": 3}
        return Pipeline(PapageiModel(device, "papagei_s", model_config=cfg), logger=logger)
    elif backbone == "papageip":
        from fmtk.components.backbones.papagei import PapageiModel
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512}
        return Pipeline(PapageiModel(device, "papagei_p", model_config=cfg), logger=logger)
    elif backbone == "papageissvri":
        from fmtk.components.backbones.papagei import PapageiModel
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512}
        return Pipeline(PapageiModel(device, "papagei_s_svri", model_config=cfg), logger=logger)
    elif backbone in ("phi3-mini", "phi3-small", "phi3-medium"):
        from fmtk.components.backbones.phi3_vllm import Phi3VLLMModel
        return Pipeline(Phi3VLLMModel(device, backbone, model_config=model_config), logger=logger)
    elif backbone in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"):
        from fmtk.components.backbones.qwen_vllm import QwenVLLMModel
        return Pipeline(QwenVLLMModel(device, backbone, model_config=model_config, async_only=True), logger=logger)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def _build_decoder(backbone: str, task: str, dtype: str, device):
    if dtype == "regression":
        cfg = DECODERS[f"mlp_{backbone}_regression"]["decoder_config"]["cfg"]
        return RegressionMLP(device=device, cfg=cfg)
    elif dtype == "classification":
        cfg = DECODERS[f"mlp_{backbone}_{task}"]["decoder_config"]["cfg"]
        return ClassificationMLP(device=device, cfg=cfg)
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

    def load_models(self, backbone: str, decoder_specs: list, model_config: dict | None = None) -> Logger:
        op_log = self._op_logger()
        print(f"[ModelLoader] Loading backbone: {backbone}")
        with op_log.measure("load_backbone", device=self.device):
            self.pipeline = _build_pipeline(backbone, self.device, op_log, model_config=model_config)
        self.backbone_name = backbone
        self.decoders = {}
        for dec in decoder_specs:
            task, dtype, path = dec["task"], dec["type"], dec["path"]
            print(f"[ModelLoader] Loading decoder: {task} ({dtype}) from {path}")
            decoder_obj = _build_decoder(backbone, task, dtype, self.device)
            self.decoders[task] = self.pipeline.add_decoder(decoder_obj, load=True, train=False, path=path)
        self._loaded = True
        print(f"[ModelLoader] Loaded {self.pipeline.model_instance.__class__.__name__} "
              f"with {len(self.decoders)} decoders.")
        self._merge(op_log)
        return op_log

    def add_decoder(self, decoder_specs: list) -> Logger:
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("No backbone loaded. Call load_models() first.")
        op_log = self._op_logger()
        # Temporarily swap pipeline logger so add_decoder records into op_log
        self.pipeline.logger = op_log
        for dec in decoder_specs:
            task, dtype, path = dec["task"], dec["type"], dec["path"]
            print(f"[ModelLoader] Hot-adding decoder: {task} ({dtype}) from {path}")
            decoder_obj = _build_decoder(self.backbone_name, task, dtype, self.device)
            self.decoders[task] = self.pipeline.add_decoder(decoder_obj, load=True, train=False, path=path)
        self.pipeline.logger = self.logger  # restore
        print(f"[ModelLoader] Hot-added {len(decoder_specs)} decoder(s). Total: {len(self.decoders)}")
        self._merge(op_log)
        return op_log

    def swap_backbone(self, backbone: str, decoder_specs: list) -> Logger:
        if self.pipeline is not None:
            print(f"[ModelLoader] Releasing '{self.backbone_name}' from GPU memory...")
            del self.pipeline
            self.pipeline = None
        self.decoders = {}
        self._loaded = False
        self.backbone_name = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[ModelLoader] GPU memory released. Loading new backbone...")
        return self.load_models(backbone, decoder_specs)
