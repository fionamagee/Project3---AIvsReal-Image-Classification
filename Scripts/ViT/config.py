import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "save_path":       os.path.join(PROJECT_ROOT, "data", "defactify_filtered"),
    "checkpoint_path": os.path.join(PROJECT_ROOT, "checkpoints", "vit_binary.pt"),
    "batch_size":      32,
    "epochs":          20,        # higher since early stopping will cut it off
    "lr":              1e-5,      # small lr for unfrozen backbone
    "num_classes":     2,
    "freeze_backbone": False,     # unfrozen this time
    "patience":        4,
    "num_workers":     2,
    "weight_decay":    0.01,      # L2 regularization
}

LABEL_NAMES = {0: "Real", 3: "SD3", 4: "DALLE3"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")