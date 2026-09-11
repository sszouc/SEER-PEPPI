# Test/test.py
import os
import torch
from Test.hgnn_model import Test
from Test.metrics import MetricUtils
from Test.seed import SeedSetter

class ModelEvaluator:
    def __init__(self, base_path, device='cuda:0'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def test(self, graph, target_etype, model_paths):
        SeedSetter.set_seed(913)
        for idx, model_path in enumerate(model_paths):
            print(f"Validating with model {idx+1}...")
            model = Test().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=False ))

            graph = graph.to(self.device)

            csv = os.path.join(self.base_path, f"val_results_fold_{idx+1}.csv")
            with open(csv, 'w') as f:
                f.write("precision,recall,f1,accuracy,auc,aupr,mcc,TP,TN,FP,FN,current_lr\n")

            model.eval()
            with torch.no_grad():
                logits = model(graph, target_etype)
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = graph.edges[target_etype].data['label'].cpu().numpy()
                threshold = MetricUtils.find_best_threshold_by_mcc(labels, probs)
                metrics = MetricUtils.calculate_metrics(labels, probs, threshold)

            with open(csv, 'a') as f:
                f.write(f"{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1']:.4f},"
                        f"{metrics['accuracy']:.4f},{metrics['auc']:.4f},{metrics['aupr']:.4f},"
                        f"{metrics['mcc']:.4f},{metrics['TP']},{metrics['TN']},"
                        f"{metrics['FP']},{metrics['FN']}\n")
            print(f"Test Results for model {idx}: {metrics}")
