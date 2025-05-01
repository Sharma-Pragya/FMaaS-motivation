from gluonts.torch.model.predictor import PyTorchPredictor
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from ..utils.gluonts_forecaster import GluonTSForecaster


class Moirai(GluonTSForecaster):
    def __init__(
        self,
        repo_id: str = "Salesforce/moirai-1.0-R-small",
        filename: str = "model.ckpt",
        batch_size: int = 32,
        alias: str = "Moirai",
    ):
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
        )
        self.handle = init_nvml()

    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        start = perf_counter()
        self.load_memory_before = get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self.repo_id),
            prediction_length=prediction_length,
            context_length=200,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=self.batch_size)
        self.load_duration = perf_counter() - start
        self.load_memory= get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]-self.load_memory_before

        return predictor
