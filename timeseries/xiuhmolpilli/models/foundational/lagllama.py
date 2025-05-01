from gluonts.torch.model.predictor import PyTorchPredictor
from lag_llama.gluon.estimator import LagLlamaEstimator

from ..utils.gluonts_forecaster import GluonTSForecaster
from ..utils.log_collector import *
from time import perf_counter

class LagLlama(GluonTSForecaster):
    def __init__(
        self,
        repo_id: str = "time-series-foundation-models/Lag-Llama",
        filename: str = "lag-llama.ckpt",
        batch_size: int = 32,
        alias: str = "LagLlama",
    ):
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
        )
        self.handle = init_nvml()


    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        ckpt = self.load()
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        # this context length is reported in the paper
        context_length = 32
        start = perf_counter()
        self.load_memory_before = get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]
        estimator = LagLlamaEstimator(
            ckpt_path=self.checkpoint_path,
            prediction_length=prediction_length,
            context_length=context_length,
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
        self.load_duration = perf_counter() - start
        self.load_memory= get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]-self.load_memory_before

        return predictor
