# trainer/config.py
class Config:
    def __init__(self):
        self.seed = 913
        self.device = "cuda:0"
        self.base_path = "../Results"
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 512
        self.epochs = 800
        self.patience = 10

        # Dual-view loss parameters
        self.loss_eta = 0.3  # η: weight for BCE loss (0-1)
        self.loss_auc_margin = 1  # m: margin for AUC loss

        # Auxiliary masked edge prediction
        self.aux_weight = 0.2
        self.aux_edge_sample = 2048
        self.aux_neg_ratio = 1.0

        # Contrastive learning
        self.contrastive_weight = 0.1
        self.contrastive_drop_rate = 0.2
        self.contrastive_temperature = 0.2

