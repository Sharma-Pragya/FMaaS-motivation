# device/config.py
import torch
import os

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model directory
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Optional: limit GPU memory growth if needed
torch.backends.cudnn.benchmark = True
