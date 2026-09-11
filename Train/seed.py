# trainer/seed.py
import random
import numpy as np
import torch
import dgl

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Note: use_deterministic_algorithms may raise on some ops; keep it if your env supports it
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # If not supported in your torch version, ignore
        pass
