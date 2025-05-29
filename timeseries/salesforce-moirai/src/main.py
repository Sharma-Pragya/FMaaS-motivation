import logging
import subprocess
from typing import Literal

import fire
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

datasets = [
    # "m1_yearly",
    # "m1_quarterly",
    # "m1_monthly",
    # "m3_yearly",
    # "m3_quarterly",
    # "m3_monthly",
    # "m3_other",
    # "m4_yearly",
    # "m4_quarterly",
    # "m4_monthly",
    # "m4_weekly",
    # "m4_daily",
    # "m4_hourly",
    "tourism_yearly",
    "tourism_quarterly",
    "tourism_monthly",
]

moirai_models=[
    "Salesforce/moirai-1.0-R-small",
    "Salesforce/moirai-1.0-R-base",
    "Salesforce/moirai-1.0-R-large"
    ]

def main(mode: Literal["fcst_statsforecast", "fcst_moirai`", "evaluation"]):
    prefix_process = ["python", "-m"]

    if mode in ["fcst_statsforecast", "fcst_moirai"]:
        for dataset in datasets:
            logger.info(f"Forecasting {dataset}...")
            suffix_process = ["--dataset", dataset]

            def process(middle_process):
                return prefix_process + middle_process + suffix_process

            if mode == "fcst_statsforecast":
                logger.info("Running StatisticalEnsemble")
                subprocess.run(process(["src.statsforecast_pipeline"]))
            elif mode == "fcst_moirai":
                for model in moirai_models:
                    logger.info(f"Running SalesforceMoirai {model}")
                    moirai_process = process(["src.moirai_pipeline"])
                    moirai_process.extend(["--model_name", model])
                    subprocess.run(moirai_process)

    elif mode == "evaluation":
        from src.utils import ExperimentHandler

        eval_df = []
        for dataset in datasets:
            logger.info(f"Evaluating {dataset}...")
            exp = ExperimentHandler(dataset)
            try:
                eval_dataset_df = exp.evaluate_models(
                    [
                        "SalesforceMoirai",
                        # "StatisticalEnsemble",
                        # "SeasonalNaive",
                    ]
                )
                print(eval_dataset_df)
                eval_df.append(eval_dataset_df)
            except Exception as e:
                logger.error(e)
        eval_df = pd.concat(eval_df).reset_index(drop=True)
        exp.save_dataframe(eval_df, "complete-results.csv")
    else:
        raise ValueError(f"mode {mode} not found")


if __name__ == "__main__":
    fire.Fire(main)
