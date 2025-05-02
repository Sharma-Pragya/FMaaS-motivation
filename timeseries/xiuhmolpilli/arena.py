from pathlib import Path
from time import perf_counter
from typing import List, Dict

import pandas as pd
from rich.console import Console
from rich.table import Table

from .models.utils.forecaster import Forecaster
from .utils.experiment_handler import ExperimentDataset, ForecastDataset
from .utils.logger_config import setup_logger

import sys
import yaml
import torch
import gc
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


class FoundationalTimeSeriesArena:
    def __init__(
        self,
        model_configs: List[Dict],
        model_registry: Dict,
        parquet_data_paths: List[str],
        results_dir: str = "./nixtla-foundational-time-series/results/",
        device: str = "cuda",
    ):
        self.model_configs = model_configs
        self.model_registry = model_registry
        self.parquet_data_paths: List[Path] = [
            Path(path) for path in parquet_data_paths
        ]
        self.results_dir = Path(results_dir)
        self.evaluation_path = self.results_dir / "evaluation.csv"
        self.device = device

    def get_model_results_path(self, model_alias: str):
        return Path(self.results_dir) / model_alias

    def instantiate_model(self, model_config: Dict):
        model_config_copy = model_config.copy()
        model_type = model_config_copy.pop("type")
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = self.model_registry[model_type]
        main_logger.info(f"Instantiating model: {model_type} with config: {model_config_copy}")
        return model_class(**model_config_copy)

    def process_single_model(self, model, dataset, dataset_results_path, overwrite=False):
        """Process a single model and return its forecast and time data"""
        try:
            main_logger.info(f"Evaluating {model.alias}")
            model_results_path = dataset_results_path / model.alias
            is_forecast_ready = ForecastDataset.is_forecast_ready(model_results_path)
            
            if not is_forecast_ready or overwrite:
                main_logger.info(f"Forecasting {model.alias}")
                start = perf_counter()
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
                    model.alias
                )
                print(time_df)
                
                fcst_dataset = ForecastDataset(
                    forecast_df=forecast_df,
                    time_df=time_df,
                )
                fcst_dataset.save_to_dir(dir=model_results_path)
            else:
                main_logger.info(f"Loading {model.alias} forecast")
                fcst_dataset = ForecastDataset.from_dir(model_results_path)
                
        except Exception as e:
            main_logger.error(f"Error in {model.alias}: {e}")
            return None, None
        finally:
            # Clean up resources
            del model
            torch.cuda.empty_cache()
            gc.collect()
        return fcst_dataset.forecast_df, fcst_dataset.time_df

    def compete(self, overwrite: bool = False):
        complete_eval = []
        
        for parquet_path in self.parquet_data_paths:
            main_logger.info(f"Running on {parquet_path}")
            dataset = ExperimentDataset.from_parquet(parquet_path=parquet_path)
            dataset_results_path = self.results_dir / parquet_path.stem
            
            # Storage for combined results
            all_forecasts = {}
            all_times = {}
            model_aliases = []
            
            # Process one model at a time
            for model_config in self.model_configs:
                # Instantiate model
                model = self.instantiate_model(model_config)
                model_aliases.append(model.alias)
                
                # Process the model
                forecast_df, time_df = self.process_single_model(
                    model=model,
                    dataset=dataset,
                    dataset_results_path=dataset_results_path,
                    overwrite=overwrite
                )
                
                # Store results if successful
                if forecast_df is not None and time_df is not None:
                    all_forecasts[model.alias] = forecast_df
                    all_times[model.alias] = time_df
            
            try:
                # Combine forecasts
                if all_forecasts:
                    # Start with the first model's forecast (which includes 'y')
                    first_model = model_aliases[0]
                    combined_forecast = all_forecasts[first_model].copy()
                    combined_time = all_times[first_model].copy()
                    
                    # Add other models
                    for model_alias in model_aliases[1:]:
                        if model_alias in all_forecasts:
                            # Merge forecasts, dropping the 'y' column to avoid duplicates
                            combined_forecast = combined_forecast.merge(
                                all_forecasts[model_alias].drop(columns="y"),
                                how="left",
                                on=["unique_id", "cutoff", "ds"],
                            )
                            # Merge time metrics
                            combined_time = combined_time.merge(
                                all_times[model_alias],
                                how="left",
                                on=["metric"],
                            )
                    
                    # Evaluate combined forecasts
                    main_logger.info("Evaluating forecasts")
                    eval_df = dataset.evaluate_forecast_df(
                        forecast_df=combined_forecast,
                        models=model_aliases,
                    )
                    eval_df = eval_df.groupby(["metric"], as_index=False).mean(
                        numeric_only=True
                    )
                    eval_df = pd.concat([combined_time, eval_df])
                    eval_df.insert(0, "dataset", parquet_path.stem)
                    complete_eval.append(eval_df)
                    
            except Exception as e:
                main_logger.error(f"Error in evaluation: {e}")
                continue
        
        try:
            if complete_eval:
                complete_eval = pd.concat(complete_eval)
                complete_eval.to_csv(
                    self.evaluation_path,
                    index=False,
                )
                print_df_rich(complete_eval)
            else:
                main_logger.error("No evaluations were completed successfully")
        except Exception as e:
            main_logger.error(f"Error in final evaluation: {e}")
            raise e


if __name__ == "__main__":
    from .models.foundational import Chronos, LagLlama, Moirai, TimesFM
    # Model class lookup
    MODEL_REGISTRY = {
        "Chronos": Chronos,
        "LagLlama": LagLlama,
        "Moirai": Moirai,
        "TimesFM": TimesFM,
    }

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    frequencies = config.get("frequencies", [])
    device = config.get("device", "cuda")
    files = [f"./nixtla-foundational-time-series/data/{freq}.parquet" for freq in frequencies]

    arena = FoundationalTimeSeriesArena(
        model_configs=config["models"],
        model_registry=MODEL_REGISTRY,
        parquet_data_paths=files,
        device=device
    )

    arena.compete()