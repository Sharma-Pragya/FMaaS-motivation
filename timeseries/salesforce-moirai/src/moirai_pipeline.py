import time
from time import perf_counter
from typing import Iterable, List, Tuple

import fire
import pandas as pd
import torch
from gluonts.dataset import Dataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from uni2ts.model.moirai import MoiraiForecast,MoiraiModule

from src.utils import ExperimentHandler
from src.utils import *

from collections.abc import Iterator

def get_morai_predictor(
    model_name: str,
    prediction_length: int,
    target_dim: int,
    batch_size: int,
) -> PyTorchPredictor:
    start = perf_counter()
    load_memory_before = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"{model_name}"),
        prediction_length=prediction_length,
        context_length=200,
        patch_size="auto",
        num_samples=100,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    predictor = model.create_predictor(batch_size)
    load_duration = perf_counter() - start
    load_memory= get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]-load_memory_before
    return predictor,load_duration,load_memory


def gluonts_instance_fcst_to_df(
    fcst: Forecast,
    quantiles: List[float],
    model_name: str,
) -> pd.DataFrame:
    point_forecast = fcst.mean
    h = len(point_forecast)
    dates = pd.date_range(
        fcst.start_date.to_timestamp(),
        freq=fcst.freq,
        periods=h,
    )
    fcst_df = pd.DataFrame(
        {
            "ds": dates,
            "unique_id": fcst.item_id,
            model_name: point_forecast,
        }
    )
    for q in quantiles:
        fcst_df[f"{model_name}-q-{q}"] = fcst.quantile(q)
    return fcst_df


def gluonts_fcsts_to_df(
    fcsts: Iterable[Forecast],
    quantiles: List[float],
    model_name: str,
) -> pd.DataFrame:
    df = []
    for fcst in tqdm(fcsts):
        fcst_df = gluonts_instance_fcst_to_df(fcst, quantiles, model_name)
        df.append(fcst_df)
    return pd.concat(df).reset_index(drop=True)


def run_moirai(
    gluonts_dataset: Dataset,
    model_name: str,
    horizon: int,
    target_dim: int,
    batch_size: int,
    quantiles: List[float],
) -> Tuple[pd.DataFrame, float, str]:
    init_time = time.time()
    predictor,load_duration,load_memory = get_morai_predictor(model_name, horizon, target_dim, batch_size)

    fcsts_iter = predictor.predict(gluonts_dataset)
    fcsts_iter = iter(fcsts_iter)  # ensure it's an iterator
    inference_times=[]
    fcsts=[]
    total_gpu_util = 0
    total_gpu_mem = 0
    total_cpu_util=0
    total_cpu_mem=0
    gpu_before = get_gpu_memory_and_util(handle)
    cpu_before = get_cpu_memory_and_util()

    while True:
        try:
            start = perf_counter()
            pred = next(fcsts_iter)

            inference_times.append(perf_counter()- start)
            gpu_after = get_gpu_memory_and_util(handle)
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
        except StopIteration:
            break
        
    fcsts_df = gluonts_fcsts_to_df(
        fcsts,
        quantiles=quantiles,
        model_name=model_name,
    )
    inference_time = sum(inference_times)
    average_batch_time = inference_time / len(inference_times)
    avg_gpu_util = total_gpu_util / len(inference_times)
    avg_gpu_mem = total_gpu_mem / len(inference_times)
    avg_cpu_util = total_cpu_util / len(inference_times)
    avg_cpu_mem = total_cpu_mem / len(inference_times)
    print(f"GPU util: {avg_gpu_util}%, Mem used: Δ{avg_gpu_mem} MB")
    print(f"Avg per batch: {average_batch_time}s")
    
    total_time = time.time() - init_time
    return fcsts_df,total_time,load_memory,load_duration, average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem



def main(dataset: str, model_name: str):
    exp = ExperimentHandler(dataset)
    fcst_df, total_time,load_memory,load_duration,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem = run_moirai(
        gluonts_dataset=exp.gluonts_train_dataset,
        model_name=model_name,
        horizon=exp.horizon,
        target_dim=1,
        batch_size=1,
        quantiles=exp.quantiles,
    )
    exp.save_results(fcst_df,model_name,total_time,load_memory,load_duration,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem)


if __name__ == "__main__":
    handle = init_nvml()
    fire.Fire(main)
