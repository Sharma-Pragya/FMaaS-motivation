# device/model_loader.py
import torch
from timeseries.pipeline import Pipeline
from timeseries.components.backbones.moment import MomentModel
# from timeseries.components.backbones.llava import LlavaModel
from timeseries.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
from timeseries.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP
from timeseries.components.decoders.forecasting.mlp import MLPDecoder as ForecastingMLP
from device.config import DEVICE

_pipeline = None
_decoders = {}
_loaded = False


def unload_models():
    """Unload all currently loaded models to free GPU memory."""
    global _pipeline, _decoders, _loaded
    if _loaded:
        del _pipeline
        _decoders.clear()
        torch.cuda.empty_cache()
        _loaded = False
        print("[ModelLoader] Unloaded all models from memory.")


def load_models(backbone: str, decoders: list):
    """
    Load a specific backbone and a set of decoders.
    backbone: str (e.g., "moment_large")
    decoders: list of {"task": str, "type": str, "path": str}
    """
    global _pipeline, _decoders, _loaded

    unload_models()

    print(f"[ModelLoader] Loading backbone: {backbone}")
    if backbone == "moment_large":
        _pipeline = Pipeline(MomentModel(DEVICE, "large"))
    elif backbone == "llava":
        _pipeline = Pipeline(LlavaModel(DEVICE, "llava-1.5-7b-hf"))
    else:
        raise ValueError(f"Unsupported backbone type: {backbone}")
    
    for dec in decoders:
        task, dtype, path = dec["task"], dec["type"], dec["path"]
        print(f"[ModelLoader] Loading decoder: {task} ({dtype}) from {path}")

        if dtype == "regression":
            decoder = RegressionMLP(device=DEVICE, cfg={"input_dim": 1024, "output_dim": 1, "hidden_dim": 128})
        elif dtype == "classification":
            output_dim = 5 if "ecg" in task else 10
            decoder = ClassificationMLP(device=DEVICE, cfg={"input_dim": 1024, "output_dim": output_dim, "hidden_dim": 128})
        elif dtype == "forecasting":
            decoder = ForecastingMLP(device=DEVICE, cfg={"input_dim": 64 * 1024, "output_dim": 192, "dropout": 0.1})
        else:
            raise ValueError(f"Unknown decoder type: {dtype}")

        _decoders[task] = _pipeline.add_decoder(decoder, load=True, trained=True, path=path)

    _loaded = True
    print(f"[ModelLoader] Loaded {_pipeline.model_instance.__class__.__name__} with {len(_decoders)} decoders.")


def get_loaded_pipeline():
    """Return the current loaded pipeline and decoders."""
    if not _loaded:
        raise RuntimeError("No models are loaded. Please deploy first via /load_model.")
    return _pipeline, _decoders
