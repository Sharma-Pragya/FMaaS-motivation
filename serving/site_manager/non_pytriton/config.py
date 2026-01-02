import os

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
DEFAULT_BATCH_SIZE = 1
PORT = 8084
BROKER = "broker.emqx.io"
SITE_ID = "site2"

pythonpath = (
    "/project/pi_shenoy_umass_edu/hshastri/FMTK:"
    "/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation:$PYTHONPATH"
)

# remote_cmd = (
#     f"bash -lc 'cd /project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation/serving && "
#     f"module load conda/latest && "
#     f"export PYTHONPATH={pythonpath} && "
#     f"conda activate {conda_env} && "
#     f"nohup {cmd}> {log_path} 2>&1 &'"
# )
activate_env="conda activate"
vlm_env='benchmark-foundation-vqa'
timeseries_env='fmtk'

cmds=f"cd /project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation/serving && module load conda/latest && export PYTHONPATH={pythonpath}"

