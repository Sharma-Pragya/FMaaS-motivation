import os
from time import time
from typing import List, Tuple

import fire
import pandas as pd


from ..utils import ExperimentHandler
from .forecaster import AmazonChronos


def run_amazon_chronos(
    train_df: pd.DataFrame,
    model_name: str,
    horizon: int,
    freq: str,
    quantiles: List[float],
) -> Tuple[pd.DataFrame, float, str]:
    ac = AmazonChronos(model_name)
    init_time = time()
    fcsts_df,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem = ac.forecast(
        df=train_df,
        h=horizon,
        freq=freq,
        batch_size=1,
        quantiles=quantiles,
        # parameters as in https://github.com/amazon-science/chronos-forecasting/blob/73be25042f5f587823d46106d372ba133152fb00/README.md?plain=1#L62-L65
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    total_time = time() - init_time
    return fcsts_df,total_time,ac.load_memory,ac.load_duration, average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem


def main(dataset: str, model_name: str):
    exp = ExperimentHandler(dataset)
    fcst_df, total_time,load_memory,load_duration,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem = run_amazon_chronos(
        train_df=exp.train_df,
        model_name=model_name,
        horizon=exp.horizon,
        freq=exp.freq,
        quantiles=exp.quantiles,
    )
    exp.save_results(fcst_df,model_name,total_time,load_memory,load_duration,average_batch_time,avg_gpu_util,avg_gpu_mem,avg_cpu_util,avg_cpu_mem)


if __name__ == "__main__":
    fire.Fire(main)
