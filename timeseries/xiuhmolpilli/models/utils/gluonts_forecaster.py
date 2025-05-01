from typing import Iterable, List, Any
from time import perf_counter
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .forecaster import Forecaster


def fix_freq(freq: str) -> str:
    # see https://github.com/awslabs/gluonts/pull/2462/files
    if len(freq) > 1 and freq.endswith("S"):
        return freq[:-1]
    return freq


def maybe_convert_col_to_float32(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if df[col_name].dtype != "float32":
        df = df.copy()
        df[col_name] = df[col_name].astype("float32")
    return df


class GluonTSForecaster(Forecaster):
    def __init__(self, repo_id: str, filename: str, alias: str):
        self.repo_id = repo_id
        self.filename = filename
        self.alias = alias

    @property
    def checkpoint_path(self) -> str:
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )

    @property
    def map_location(self) -> str:
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        return map_location

    def load(self) -> Any:
        return torch.load(
            self.checkpoint_path,
            map_location=self.map_location,
        )

    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        raise NotImplementedError

    def gluonts_instance_fcst_to_df(
        self,
        fcst: Forecast,
        freq: str,
        model_name: str,
    ) -> pd.DataFrame:
        point_forecast = fcst.mean
        h = len(point_forecast)
        dates = pd.date_range(
            fcst.start_date.to_timestamp(),
            freq=freq,
            periods=h,
        )
        fcst_df = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                model_name: point_forecast,
            }
        )
        return fcst_df

    def gluonts_fcsts_to_df(
        self,
        fcsts: Iterable[Forecast],
        freq: str,
        model_name: str,
    ) -> pd.DataFrame:
        df = []
        for fcst in tqdm(fcsts):
            fcst_df = self.gluonts_instance_fcst_to_df(
                fcst=fcst,
                freq=freq,
                model_name=model_name,
            )
            df.append(fcst_df)
        return pd.concat(df).reset_index(drop=True)

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        df = maybe_convert_col_to_float32(df, "y")
        gluonts_dataset = PandasDataset.from_long_dataframe(
            df,
            target="y",
            item_id="unique_id",
            timestamp="ds",
            freq=fix_freq(freq),
        )
        inference_times=[]
        total_gpu_util = 0
        total_gpu_mem = 0
        total_cpu_util=0
        total_cpu_mem=0

        predictor = self.get_predictor(prediction_length=h)

        gpu_before = get_gpu_memory_and_util(self.handle)
        cpu_before = get_cpu_memory_and_util()
        start = perf_counter()

        fcsts = predictor.predict(gluonts_dataset, num_samples=100)
        inference_times.append(perf_counter() - start)
        gpu_after = get_gpu_memory_and_util(self.handle)
        cpu_after = get_cpu_memory_and_util()
        
        fcst_df = self.gluonts_fcsts_to_df(
            fcsts,
            freq=freq,
            model_name=self.alias,
        )
        gpu_mem_delta = gpu_after["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
        gpu_util_delta = gpu_after["gpu_util_percent"] - gpu_before["gpu_util_percent"]
        cpu_mem_delta = cpu_after["cpu_mem_used_mb"] - cpu_before["cpu_mem_used_mb"]
        cpu_util_delta = cpu_after["cpu_util_percent"] - cpu_before["cpu_util_percent"]
        print(f"CPU util: {cpu_after['cpu_util_percent']}%, Mem used: Δ{cpu_mem_delta:.2f} MB")
        print(f"GPU util: {gpu_after['gpu_util_percent']}%, Mem used: Δ{gpu_mem_delta:.2f} MB")
        total_gpu_util += gpu_after["gpu_util_percent"]
        total_gpu_mem += gpu_mem_delta
        total_cpu_util += cpu_after["cpu_util_percent"]
        total_cpu_mem += cpu_mem_delta

        inference_time = sum(inference_times)

        average_batch_time = inference_time / len(inference_times)
        avg_gpu_util = total_gpu_util / len(inference_times)
        avg_gpu_mem = total_gpu_mem / len(inference_times)
        avg_cpu_util = total_cpu_util / len(inference_times)
        avg_cpu_mem = total_cpu_mem / len(inference_times)

        print(f"GPU util: {avg_gpu_util}%, Mem used: Δ{avg_gpu_mem} MB")
        print(f"Avg per batch: {average_batch_time}s")

        return fcst_df,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem