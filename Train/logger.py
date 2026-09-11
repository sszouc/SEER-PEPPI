# trainer/logger.py
import os

class CSVLogger:
    def __init__(self, folder, prefix):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.train_path = os.path.join(folder, f"{prefix}_train.csv")
        self.val_path = os.path.join(folder, f"{prefix}_val.csv")

    def write_headers(self):
        header = "epoch,loss,precision,recall,f1,accuracy,auc,aupr,mcc,TP,TN,FP,FN,current_lr,aux_loss\n"
        # 覆盖写入header
        with open(self.train_path, 'w') as f:
            f.write(header)
        with open(self.val_path, 'w') as f:
            f.write(header)

    def append_train(self, row_str):
        with open(self.train_path, 'a') as f:
            f.write(row_str + "\n")

    def append_val(self, row_str):
        with open(self.val_path, 'a') as f:
            f.write(row_str + "\n")
