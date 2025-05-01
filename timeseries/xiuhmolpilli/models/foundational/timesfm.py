import pandas as pd
import timesfm
from time import perf_counter
import torch
from paxml import checkpoints

from ..utils.forecaster import Forecaster


class TimesFM(Forecaster):
    def __init__(
        self,
        repo_id: str = "google/timesfm-1.0-200m",
        context_length: int = 512,
        batch_size: int = 64,
        alias: str = "TimesFM",
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.handle = init_nvml()

    def get_predictor(
        self,
        prediction_length: int,
    ) -> timesfm.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        start = perf_counter()
        self.load_memory_before = get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]
        tfm = timesfm.TimesFm(
            context_len=self.context_length,
            horizon_len=prediction_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=backend,
            per_core_batch_size=self.batch_size,
        )
        tfm.load_from_checkpoint(repo_id=self.repo_id)
        self.load_duration = perf_counter() - start
        self.load_memory= get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]-self.load_memory_before
        return tfm

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        predictor = self.get_predictor(prediction_length=h)
        inference_times=[]
        total_gpu_util = 0
        total_gpu_mem = 0
        tracemalloc.start()
        handle = init_nvml()
        start = perf_counter()
        gpu_before = get_gpu_memory_and_util(handle)
        fcst_df = predictor.forecast_on_df(
            inputs=df,
            freq=freq,
            value_name="y",
            model_name=self.alias,
            num_jobs=1,
        )
        inference_times.append(perf_counter() - start)
        gpu_after = get_gpu_memory_and_util(handle)
        gpu_mem_delta = gpu_after["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
        print(f"GPU util: {gpu_after['gpu_util_percent']}%, Mem used: Δ{gpu_mem_delta:.2f} MB")
        total_gpu_util += gpu_after["gpu_util_percent"]
        total_gpu_mem += gpu_after["gpu_mem_used_mb"]

        total_inference_time = sum(inference_times)
        average_batch_time = total_inference_time / len(inference_times)
        avg_gpu_util = total_gpu_util / len(inference_times)
        avg_gpu_mem = total_gpu_mem / len(inference_times)

        print(f"GPU util: {avg_gpu_util}%, Mem used: Δ{avg_gpu_mem} MB")
        print(f"Total inference time: {total_inference_time}s, Avg per batch: {average_batch_time}s")

        fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df,average_batch_time,total_inference_time,avg_gpu_util,avg_gpu_mem, peak / 1024**2