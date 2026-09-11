# Trainer/metrics.py
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_recall_curve, confusion_matrix
)

class MetricUtils:
    @staticmethod
    def best_threshold_shift(y_true, y_pred):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = np.zeros_like(precision)
        nonzero = (precision + recall) > 0
        f1_scores[nonzero] = 2 * (precision[nonzero] * recall[nonzero]) / (precision[nonzero] + recall[nonzero])
        best_index = np.argmax(f1_scores)
        if best_index == len(thresholds):
            best_index -= 1
        return thresholds[best_index]

    @staticmethod
    def find_best_threshold_by_mcc(y_true, y_pred_probs, thresholds=None):
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 500)
        best_threshold = 0.5
        best_score = -1
        for threshold in thresholds:
            y_pred = (y_pred_probs >= threshold).astype(int)
            if len(np.unique(y_pred)) < 2:
                continue
            score = matthews_corrcoef(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold

    @staticmethod
    def calculate_metrics(y_true, y_prob, threshold):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'auc': auc, 'aupr': aupr, 'mcc': mcc,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        }

    @staticmethod
    def should_stop_early(current, best, counter, patience, epoch, min_epoch=100, larger_is_better=True):
        improved = (current > best) if larger_is_better else (current < best)
        if improved:
            return True, False, 0, current
        else:
            if epoch >= min_epoch:
                counter += 1
                if counter >= patience:
                    return False, True, counter, best
            return False, False, counter, best
