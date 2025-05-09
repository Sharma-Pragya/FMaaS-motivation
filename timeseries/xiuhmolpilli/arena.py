from pathlib import Path
from time import perf_counter
from typing import List

import pandas as pd
from rich.console import Console
from rich.table import Table

from .models.utils.forecaster import Forecaster
from .utils.experiment_handler import ExperimentDataset, ForecastDataset
from .utils.logger_config import setup_logger

import sys
import yaml
main_logger = setup_logger(__name__)


def print_df_rich(df: pd.DataFrame):
    console = Console()
    table = Table()
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")
    for col in df.columns:
        table.add_column(col)
    for row in df.itertuples(index=False):
        table.add_row(*row)
    console.print(table)


def time_to_df(total_time: float, average_batch_time:float,average_gpu_mem:float,average_gpu_util:float,average_cpu_util:float,average_cpu_mem:float,load_duration:float,load_memory:float,model_name: str) -> pd.DataFrame:
    return pd.DataFrame([{"metric": "time", model_name: total_time},
            {"metric": "AVG batch time", model_name: average_batch_time},
            {"metric": "AVG GPU mem", model_name: average_gpu_mem},
            {"metric": "AVG GPU util", model_name: average_gpu_util},
            {"metric": "AVG CPU mem", model_name: average_cpu_mem},
            {"metric": "AVG CPU util", model_name: average_cpu_util},
            {"metric": "Load time", model_name: load_duration},
            {"metric": "Load memory", model_name: load_memory}
            ])


class FoundationalTimeSeriesArena:
    def __init__(
        self,
        models: List[Forecaster],
        parquet_data_paths: List[str],
        results_dir: str = "./nixtla-foundational-time-series/results/",
    ):
        self.models = models
        self.parquet_data_paths: List[Path] = [
            Path(path) for path in parquet_data_paths
        ]
        self.results_dir = Path(results_dir)
        self.evaluation_path = self.results_dir / "evaluation.csv"

    def get_model_results_path(self, model_alias: str):
        return Path(self.results_dir) / model_alias

    def compete(self, overwrite: bool = False):
        complete_eval = []
        for parquet_path in self.parquet_data_paths:
            main_logger.info(f"Running on {parquet_path}")
            dataset = ExperimentDataset.from_parquet(parquet_path=parquet_path)
            dataset_results_path = self.results_dir / parquet_path.stem
            models_fcsts = None
            models_times = None
            for model in self.models:
                main_logger.info(f"Evaluating {model.alias}")
                model_results_path = dataset_results_path / model.alias
                is_forecast_ready = ForecastDataset.is_forecast_ready(
                    model_results_path
                )
                if not is_forecast_ready or overwrite:
                    main_logger.info(f"Forecasting {model.alias}")
                    start = perf_counter()
                    forecast_df,average_batch_time,average_gpu_util,average_gpu_mem,average_cpu_util,average_cpu_mem = model.cross_validation(
                        df=dataset.df,
                        h=dataset.horizon,
                        freq=dataset.pandas_frequency,
                    )
                    total_time = perf_counter() - start
                    fcst_dataset = ForecastDataset(
                        forecast_df=forecast_df,
                        time_df=time_to_df(total_time,average_batch_time,average_gpu_mem,average_gpu_util,average_cpu_util,average_cpu_mem,model.load_duration,model.load_memory,model.alias),
                    )
                    print(time_to_df(total_time, average_batch_time,average_gpu_mem,average_gpu_util,average_cpu_util,average_cpu_mem,model.load_duration,model.load_memory,model.alias))
                    fcst_dataset.save_to_dir(dir=model_results_path)
                else:
                    main_logger.info(f"Loading {model.alias} forecast")
                    fcst_dataset = ForecastDataset.from_dir(model_results_path)
                if models_fcsts is None:
                    models_fcsts = fcst_dataset.forecast_df
                    models_times = fcst_dataset.time_df
                else:
                    models_fcsts = models_fcsts.merge(
                        fcst_dataset.forecast_df.drop(columns="y"),
                        how="left",
                        on=["unique_id", "cutoff", "ds"],
                    )
                    models_times = models_times.merge(
                        fcst_dataset.time_df,
                        how="left",
                        on=["metric"],
                    )
            main_logger.info("Evaluating forecasts")
            eval_df = dataset.evaluate_forecast_df(
                forecast_df=models_fcsts,
                models=[model.alias for model in self.models],
            )
            eval_df = eval_df.groupby(["metric"], as_index=False).mean(
                numeric_only=True
            )
            eval_df = pd.concat([models_times, eval_df])
            eval_df.insert(0, "dataset", parquet_path.stem)
            complete_eval.append(eval_df)
        complete_eval = pd.concat(complete_eval)
        complete_eval.to_csv(
            self.evaluation_path,
            index=False,
        )
        print_df_rich(complete_eval)


if __name__ == "__main__":
    from .models.foundational import Chronos
    # LagLlama, Moirai, TimesFM
    # Model class lookup
    MODEL_REGISTRY = {
        "Chronos": Chronos,
        # "LagLlama": LagLlama,
        # "Moirai": Moirai,
        # "TimesFM": TimesFM,
    }

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    frequencies = config.get("frequencies", [])
    files = [f"./nixtla-foundational-time-series/data/{freq}.parquet" for freq in frequencies]

    models = []
    for mcfg in config["models"]:
        model_type = mcfg.pop("type")
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = MODEL_REGISTRY[model_type]
        print(f"Instantiating model: {model_type} with config: {mcfg}")
        models.append(model_class(**mcfg))

    arena = FoundationalTimeSeriesArena(
        models=models,
        parquet_data_paths=files,
    )

    arena.compete()
