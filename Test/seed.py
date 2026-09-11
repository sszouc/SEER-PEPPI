# Trainer/seed.py
import random
import numpy as np
import torch
import dgl

class SeedSetter:
    @staticmethod
    def set_seed(seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
