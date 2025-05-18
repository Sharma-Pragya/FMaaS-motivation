from pathlib import Path
from time import perf_counter
from typing import List, Dict

import pandas as pd
from rich.console import Console
from rich.table import Table

from models.utils.forecaster import Forecaster
from utils.experiment_handler import ExperimentDataset, ForecastDataset
from utils.logger_config import setup_logger

import sys
import yaml
import torch
import gc
import subprocess
import json
import tempfile
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
        self.device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

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

    def compete(self, overwrite: bool = False):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        all_eval_paths = []

        for parquet_path in self.parquet_data_paths:
            main_logger.info(f"Running on {parquet_path}")
            dataset_results_path = self.results_dir / parquet_path.stem
            dataset_results_path.mkdir(parents=True, exist_ok=True)

            for model_config in self.model_configs:
                try:
                    # Write model config to a temp file
                    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json") as tmp_config_file:
                        json.dump(model_config, tmp_config_file)
                        tmp_config_path = tmp_config_file.name

                    # Build the subprocess command
                    command = [
                        sys.executable,
                        "gpu_run_model_inference.py",
                        tmp_config_path,
                        str(parquet_path),
                        str(dataset_results_path),
                        self.device,
                        str(self.evaluation_path),
                    ]

                    main_logger.info(f"Starting subprocess for model: {model_config['alias']}")
                    subprocess.run(command, check=True)
                    main_logger.info(f"Finished subprocess for model: {model_config['alias']}")

                except subprocess.CalledProcessError as e:
                    main_logger.error(f"Subprocess for model {model_config['alias']} failed: {e}")

        # Final evaluation display
        try:
            if self.evaluation_path.exists():
                final_eval_df = pd.read_csv(self.evaluation_path)
                # print_df_rich(final_eval_df)
            else:
                main_logger.error("No evaluations were completed successfully")
        except Exception as e:
            main_logger.error(f"Error reading final evaluation file: {e}")
            raise e


if __name__ == "__main__":
    from models.foundational import Chronos, LagLlama, Moirai, TimesFM
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