from typing import Iterable, List
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from tqdm import tqdm
from utilsforecast.processing import make_future_dataframe

from ..utils.forecaster import Forecaster
from ..utils.log_collector import *

import tracemalloc
import psutil


class TimeSeriesDataset:
    def __init__(
        self,
        data: torch.Tensor,
        uids: Iterable,
        last_times: Iterable,
        batch_size: int,
    ):
        self.data = data
        self.uids = uids
        self.last_times = last_times
        self.batch_size = batch_size
        self.n_batches = len(data) // self.batch_size + (
            0 if len(data) % self.batch_size == 0 else 1
        )
        self.current_batch = 0

    @classmethod
    def from_df(cls, df: pd.DataFrame, batch_size: int):
        num_unique_ids = df["unique_id"].nunique()
        max_series_length = df["unique_id"].value_counts().max()
        padded_tensor = torch.full(
            size=(num_unique_ids, max_series_length),
            fill_value=torch.nan,
            dtype=torch.bfloat16,
        )  # type: ignore
        df_sorted = df.sort_values(by=["unique_id", "ds"])
        for idx, (_, group) in enumerate(df_sorted.groupby("unique_id")):
            series_length = len(group)
            padded_tensor[idx, -series_length:] = torch.tensor(
                group["y"].values,
                dtype=torch.bfloat16,
            )
        uids = df_sorted["unique_id"].unique()
        last_times = df_sorted.groupby("unique_id")["ds"].tail(1)
        return cls(padded_tensor, uids, last_times, batch_size)

    def __len__(self):
        return self.n_batches

    def make_future_dataframe(self, h: int, freq: str) -> pd.DataFrame:
        return make_future_dataframe(
            uids=self.uids,
            last_times=pd.to_datetime(self.last_times),
            h=h,
            freq=freq,
        )  # type: ignore

    def __iter__(self):
        self.current_batch = 0  # Reset for new iteration
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_batch += 1
            return self.data[start_idx:end_idx]
        else:
            raise StopIteration


class Chronos(Forecaster):
    def __init__(
        self,
        repo_id: str = "amazon/chronos-t5-tiny",
        batch_size: int = 16,
        alias: str = "Chronos",
    ):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        self.handle = init_nvml()
        start = perf_counter()
        self.load_memory_before = get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]
        self.model = ChronosPipeline.from_pretrained(
            repo_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )
        self.load_duration = perf_counter() - start
        self.load_memory = get_gpu_memory_and_util(self.handle)["gpu_mem_used_mb"]-self.load_memory_before

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        inference_times=[]
        fcsts=[]
        total_gpu_util = 0
        total_gpu_mem = 0
        total_cpu_util=0
        total_cpu_mem=0
        gpu_before = get_gpu_memory_and_util(self.handle)
        cpu_before = get_cpu_memory_and_util()
        for batch in tqdm(dataset):
            start = perf_counter()

            pred = self.model.predict(batch, prediction_length=h)
            inference_times.append(perf_counter() - start)
            gpu_after = get_gpu_memory_and_util(self.handle)
            cpu_after = get_cpu_memory_and_util()
            
            fcsts.append(pred)
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

        fcst = torch.cat(fcsts)
        fcst = fcst.numpy()
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        fcst_df[self.alias] = np.mean(fcst, axis=1).reshape(-1, 1)

        inference_time = sum(inference_times)

        average_batch_time = inference_time / len(inference_times)
        avg_gpu_util = total_gpu_util / len(inference_times)
        avg_gpu_mem = total_gpu_mem / len(inference_times)
        avg_cpu_util = total_cpu_util / len(inference_times)
        avg_cpu_mem = total_cpu_mem / len(inference_times)

        print(f"GPU util: {avg_gpu_util}%, Mem used: Δ{avg_gpu_mem} MB")
        print(f"Avg per batch: {average_batch_time}s")

        return fcst_df,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem