import os

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
DEFAULT_BATCH_SIZE = 1
PORT = 8084
BROKER = "broker.emqx.io"
SITE_ID = "site2"

pythonpath = (
    "/project/pi_shenoy_umass_edu/hshastri/FMTK/src:"
    "/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation:$PYTHONPATH"
)

activate_env="conda activate"
vlm_env='fmtk_vlm'
timeseries_env='fmtk_vlm'

cmds=f"cd /project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation/serving && module load conda/latest && export PYTHONPATH={pythonpath}"
username="hshastri_umass_edu"

# pythonpath = (
#     "/nfs/obelix/users3/hshastri/FMTK/src:"
#     "/nfs/obelix/users3/hshastri/FMaaS-motivation:$PYTHONPATH"
# )

# activate_env="conda activate"
# vlm_env='fmtk'
# timeseries_env='fmtk'
# username="hshastri"

# cmds=f"cd /nfs/obelix/users3/hshastri/FMaaS-motivation/serving && export PYTHONPATH={pythonpath}"


