# device/model_loader.py
import torch
from timeseries.pipeline import Pipeline
from timeseries.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
from timeseries.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP
from timeseries.components.decoders.forecasting.mlp import MLPDecoder as ForecastingMLP
from device.config import DEVICE, DECODERS

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

    # unload_models()

    print(f"[ModelLoader] Loading backbone: {backbone}")
    if backbone == "momentlarge":
        from timeseries.components.backbones.moment import MomentModel
        _pipeline = Pipeline(MomentModel(DEVICE, "large"))
    elif backbone == "momentsmall":
        from timeseries.components.backbones.moment import MomentModel
        _pipeline = Pipeline(MomentModel(DEVICE, "small"))
    elif backbone == "momentbase":
        from timeseries.components.backbones.moment import MomentModel
        _pipeline = Pipeline(MomentModel(DEVICE, "base"))
    elif backbone == "chronostiny":
        from timeseries.components.backbones.chronos import ChronosModel
        _pipeline = Pipeline(ChronosModel(DEVICE, "tiny"))
    elif backbone == "chronosbase":
        from timeseries.components.backbones.chronos import ChronosModel
        _pipeline = Pipeline(ChronosModel(DEVICE, "base"))
    elif backbone == "chronoslarge":
        from timeseries.components.backbones.chronos import ChronosModel
        _pipeline = Pipeline(ChronosModel(DEVICE, "large"))
    elif backbone == "chronosmini": 
        from timeseries.components.backbones.chronos import ChronosModel
        _pipeline = Pipeline(ChronosModel(DEVICE, "mini"))
    elif backbone == "papageis":
        from timeseries.components.backbones.papagei import PapageiModel
        _pipeline = Pipeline(PapageiModel(DEVICE, "papagei_s"))
    elif backbone == "papageip":
        from timeseries.components.backbones.papagei import PapageiModel
        _pipeline = Pipeline(PapageiModel(DEVICE, "papagei_p"))
    elif backbone == "papageissvri":
        from timeseries.components.backbones.papagei import PapageiModel
        _pipeline = Pipeline(PapageiModel(DEVICE, "papagei_s_svri"))
    elif backbone == "llava":
        from timeseries.components.backbones.llava import LlavaModel
        _pipeline = Pipeline(LlavaModel(DEVICE, "llava-1.5-7b-hf"))
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

        _decoders[task] = _pipeline.add_decoder(decoder, load=True, trained=True, path=path)

    _loaded = True
    print(f"[ModelLoader] Loaded {_pipeline.model_instance.__class__.__name__} with {len(_decoders)} decoders.")


def get_loaded_pipeline():
    """Return the current loaded pipeline and decoders."""
    if not _loaded:
        raise RuntimeError("No models are loaded. Please deploy first via /load_model.")
    return _pipeline, _decoders
