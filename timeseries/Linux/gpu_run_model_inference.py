import sys
import json
import torch
import gc
import pandas as pd
from pathlib import Path
from time import perf_counter
from models.utils.forecaster import Forecaster
from utils.experiment_handler import ExperimentDataset, ForecastDataset
from utils.logger_config import setup_logger
from utils.logger_config import setup_logger
from models.foundational import Chronos, LagLlama, Moirai, TimesFM

main_logger = setup_logger(__name__)

MODEL_REGISTRY = {
    "Chronos": Chronos,
    "LagLlama": LagLlama,
    "Moirai": Moirai,
    "TimesFM": TimesFM,
}


def time_to_df(total_time: float, average_batch_time:float, average_gpu_mem:float, average_gpu_util:float, average_cpu_util:float, average_cpu_mem:float, load_duration:float, load_memory:float, model_name: str) -> pd.DataFrame:
    return pd.DataFrame([{"metric": "time", model_name: total_time},
            {"metric": "AVG batch time", model_name: average_batch_time},
            {"metric": "AVG GPU mem", model_name: average_gpu_mem},
            {"metric": "AVG GPU util", model_name: average_gpu_util},
            {"metric": "AVG CPU mem", model_name: average_cpu_mem},
            {"metric": "AVG CPU util", model_name: average_cpu_util},
            {"metric": "Load time", model_name: load_duration},
            {"metric": "Load memory", model_name: load_memory}
            ])


def run_model_and_evaluate(model_config: dict, dataset_path: str, dataset_results_path: str, device: str, output_csv_path: str):
    dataset = ExperimentDataset.from_parquet(parquet_path=Path(dataset_path))
    model_type = model_config["type"]
    model_class = MODEL_REGISTRY[model_type]

    main_logger.info(f"Running model: {model_config['alias']} on dataset {dataset_path}")
    model = model_class(**{k: v for k, v in model_config.items() if k != "type"})
    start= perf_counter()
    forecast_df, average_batch_time, average_gpu_util, average_gpu_mem, average_cpu_util, average_cpu_mem = model.cross_validation(
        df=dataset.df,
        h=dataset.horizon,
        freq=dataset.pandas_frequency,
    )

    total_time = perf_counter() - start
    time_df = time_to_df(
        total_time,
        average_batch_time,
        average_gpu_mem,
        average_gpu_util,
        average_cpu_util,
        average_cpu_mem,
        model.load_duration,
        model.load_memory,
        model.alias,
    )

    model_results_path = Path(dataset_results_path) / model.alias
    fcst_dataset = ForecastDataset(forecast_df=forecast_df, time_df=time_df)
    fcst_dataset.save_to_dir(dir=model_results_path)

    eval_df = dataset.evaluate_forecast_df(forecast_df, models=[model.alias])
    eval_df = eval_df.groupby(["metric"], as_index=False).mean(numeric_only=True)
    print(eval_df)
    mae = eval_df.loc[eval_df["metric"] == "mae", model.alias].values[0]
    rmse = eval_df.loc[eval_df["metric"] == "rmse", model.alias].values[0]
    mase = eval_df.loc[eval_df["metric"] == "mase", model.alias].values[0]
    flat_metrics = {
        "model_name": model.alias,
        "dataset_name": Path(dataset_path).stem,
        "device": device,
        "num_samples": forecast_df["unique_id"].nunique(),
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "total_inference_time_sec": total_time,
        "average_latency_ms": average_batch_time,
        "model_load_duration_sec": model.load_duration,
        "gpu_load_memory_mb": model.load_memory,
        "avg_cpu_usage_percent": average_cpu_util,
        "avg_cpu_memory_usage_mb": average_cpu_mem,
        "avg_gpu_usage_percent": average_gpu_util,
        "avg_gpu_memory_usage_mb": average_gpu_mem,
    }

    flat_df = pd.DataFrame([flat_metrics])

    # Append or create file
    output_path = Path(output_csv_path)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        updated_df = pd.concat([existing_df, flat_df], ignore_index=True)
    else:
        updated_df = flat_df

    updated_df.to_csv(output_path, index=False)
    main_logger.info(f"Metrics for {model.alias} written to {output_csv_path}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    model_config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    dataset_results_path = sys.argv[3]
    device = sys.argv[4]
    output_csv_path = sys.argv[5]

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    run_model_and_evaluate(
        model_config=model_config,
        dataset_path=dataset_path,
        dataset_results_path=dataset_results_path,
        device=device,
        output_csv_path=output_csv_path,
    )
