import os

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "FMTK", "dataset"))
DEFAULT_BATCH_SIZE = 1
PORT = 8084
BROKER = "broker.emqx.io"
SITE_ID = "site2"

##Unity setup
# pythonpath = (
#     "/project/pi_shenoy_umass_edu/hshastri/FMTK/src:"
#     "/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation:$PYTHONPATH"
# )

# activate_env="conda activate"
# vlm_env='fmtk_vllm'
# timeseries_env='fmtk_vllm'
# ssh_key=None  # use SSH agent (SSH_AUTH_SOCK)

# cmds=f"cd /project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation/serving && module load conda/latest && export PYTHONPATH={pythonpath}"
# username="hshastri_umass_edu"

# ##Obelix setup
# pythonpath = (
#     "/nfs/obelix/users3/hshastri/FMTK/src:"
#     "/nfs/obelix/users3/hshastri/FMaaS-motivation:$PYTHONPATH"
# )

# activate_env="conda activate"
# vlm_env='fmtk'
# timeseries_env='fmtk'
# username="hshastri"
# ssh_key=None  # use SSH agent (SSH_AUTH_SOCK)

# cmds=f"cd /nfs/obelix/users3/hshastri/FMaaS-motivation/serving && export PYTHONPATH={pythonpath}"

#AWS setup
pythonpath = (
    "/NFS/FMTK/src:"
    "/NFS/FMaaS-motivation:$PYTHONPATH"
)

activate_env="conda activate"
vlm_env='fmtk'
timeseries_env='fmtk'
username="ubuntu"
ssh_key="~/.ssh/hetvi-ohio.pem"

cmds=f"cd /NFS/FMaaS-motivation/serving && export PYTHONPATH={pythonpath} && source /home/ubuntu/anaconda3/etc/profile.d/conda.sh"


