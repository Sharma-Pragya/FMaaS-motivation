# device/model_loader.py
import torch
from fmtk.pipeline import Pipeline
from fmtk.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
from fmtk.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP
from fmtk.components.decoders.forecasting.mlp import MLPDecoder as ForecastingMLP
from fmtk.logger import Logger
from device.config import DEVICE, DECODERS
from contextlib import nullcontext


_pipeline = None
_decoders = {}
_loaded = False
current_task = None


def load_models(backbone: str, decoders: list):
    """
    Load a specific backbone and a set of decoders.
    backbone: str (e.g., "moment_large")
    decoders: list of {"task": str, "type": str, "path": str}
    """
    global _pipeline, _decoders, _loaded, current_task
    logger=Logger(DEVICE,'deploymentlogger')
    print(f"[ModelLoader] Loading backbone: {backbone}")
    with (logger.measure("load_backbone", device=DEVICE) if logger else nullcontext()):
        if backbone == "momentlarge":
            from fmtk.components.backbones.moment import MomentModel
            _pipeline = Pipeline(MomentModel(DEVICE, "large"),logger=logger)
        elif backbone == "momentsmall":
            from fmtk.components.backbones.moment import MomentModel
            _pipeline = Pipeline(MomentModel(DEVICE, "small"),logger=logger)
        elif backbone == "momentbase":
            from fmtk.components.backbones.moment import MomentModel
            _pipeline = Pipeline(MomentModel(DEVICE, "base"),logger=logger)
        elif backbone == "chronostiny":
            from fmtk.components.backbones.chronos import ChronosModel
            _pipeline = Pipeline(ChronosModel(DEVICE, "tiny"),logger=logger)
        elif backbone == "chronosbase":
            from fmtk.components.backbones.chronos import ChronosModel
            _pipeline = Pipeline(ChronosModel(DEVICE, "base"),logger=logger)
        elif backbone == "chronoslarge":
            from fmtk.components.backbones.chronos import ChronosModel
            _pipeline = Pipeline(ChronosModel(DEVICE, "large"),logger=logger)
        elif backbone == "chronosmini": 
            from fmtk.components.backbones.chronos import ChronosModel
            _pipeline = Pipeline(ChronosModel(DEVICE, "mini"),logger=logger)
        elif backbone == "papageis":
            from fmtk.components.backbones.papagei import PapageiModel
            _pipeline = Pipeline(PapageiModel(DEVICE, "papagei_s"),logger=logger)
        elif backbone == "papageip":
            from fmtk.components.backbones.papagei import PapageiModel
            _pipeline = Pipeline(PapageiModel(DEVICE, "papagei_p"))
        elif backbone == "papageissvri":
            from fmtk.components.backbones.papagei import PapageiModel
            _pipeline = Pipeline(PapageiModel(DEVICE, "papagei_s_svri"),logger=logger)
        elif backbone == "llava":
            from fmtk.components.backbones.llava import LlavaModel
            _pipeline = Pipeline(LlavaModel(DEVICE, "llava-1.5-7b-hf"),logger=logger)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}")
    
    for dec in decoders:
        task, dtype, path = dec["task"], dec["type"], dec["path"]
        print(f"[ModelLoader] Loading decoder: {task} ({dtype}) from {path}")
        if dtype == "regression":
            decoder_config=DECODERS[f'mlp_{backbone}_{dtype}']['decoder_config']
            decoder = RegressionMLP(device=decoder_config['DEVICE'],cfg=decoder_config['cfg'])
        elif dtype == "classification":
            decoder_config=DECODERS[f'mlp_{backbone}_{task}']['decoder_config']
            decoder = ClassificationMLP(device=decoder_config['DEVICE'],cfg=decoder_config['cfg'])
        elif dtype == "forecasting":
            decoder_config=DECODERS[f'mlp_{backbone}_{dtype}']['decoder_config']
            decoder = ForecastingMLP(device=decoder_config['DEVICE'],cfg=decoder_config['cfg'])
        else:
            raise ValueError(f"Unknown decoder type: {dtype} or mlp_{backbone}_{dtype} or mlp_{backbone}_{task}")

        _decoders[task] = _pipeline.add_decoder(decoder, load=True, train=False, path=path)
        current_task=task
    _loaded = True
    print(f"[ModelLoader] Loaded {_pipeline.model_instance.__class__.__name__} with {len(_decoders)} decoders.")
    return logger


def get_loaded_pipeline():
    """Return the current loaded pipeline and decoders."""
    print(f"[ModelLoader] Retrieving loaded models. Loaded status: {_loaded}")
    if not _loaded:
        raise RuntimeError("No models are loaded. Please deploy first via /load_model.")
    return _pipeline, _decoders, current_task
