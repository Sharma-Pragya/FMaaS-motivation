import os

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
DEFAULT_BATCH_SIZE = 1
PORT = 8084
BROKER = "broker.emqx.io"
SITE_ID = "site1"

custom_pythonpath = (
    "/work/pi_shenoy_umass_edu/hshastri/FMaaS-motivation:"
    "/project/pi_shenoy_umass_edu/hshastri/foundation-model-zoo:"
    "/work/pi_shenoy_umass_edu/hshastri/FMaaS-motivation:"
    "/project/pi_shenoy_umass_edu/hshastri/foundation-model-zoo:$PYTHONPATH"
)