# Trainer/summarizer.py
import os
import pandas as pd

class FiveFoldSummarizer:
    @staticmethod
    def summarize(base_path: str):
        all_results = []
        for fold in range(1, 6):
            val_csv = os.path.join(base_path, f"val_results_fold_{fold}.csv")
            if not os.path.exists(val_csv):
                continue
            df = pd.read_csv(val_csv)
            if df.empty:
                continue
            all_results.append(df.iloc[-1])
        if not all_results:
            print("No fold results found.")
            return
        summary_df = pd.DataFrame(all_results)
        mean = summary_df.mean(numeric_only=True)
        std = summary_df.std(numeric_only=True)
        print("\n========== 5-Fold Validation Summary ==========")
        for col in ['precision','recall','f1','accuracy','auc','aupr','mcc']:
            print(f"{col.upper():<9}: {mean[col]:.4f} ± {std[col]:.4f}")
