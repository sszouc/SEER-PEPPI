# main.py
"""
Main entry point to train the ImprovedProteinGNN model on peptide-receptor interaction graphs.
This script:
  1. (Optionally) builds the heterogeneous graph if fold files are missing.
  2. Loads pre-saved training/validation graph folds.
  3. Trains the GNN model for each fold using the Trainer class.
  4. Saves metrics and best checkpoints per fold.

Assumes that graph folds (e.g., train_fold_0.dgl, val_fold_0.dgl) have been pre-generated
and stored in a known directory.
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import dgl
import torch
from sklearn.model_selection import StratifiedKFold
from Train.config import Config
from Train.trainer import Trainer


def build_kfold_graphs(graph_path, target_etype, seed, num_folds=5):
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    g = dgl.load_graphs(graph_path)[0][0]
    if target_etype not in g.canonical_etypes:
        raise ValueError(f"Target etype {target_etype} not found in graph.")

    labels = g.edges[target_etype].data['label'].cpu().numpy()
    num_edges = g.number_of_edges(target_etype)
    edge_indices = list(range(num_edges))

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(edge_indices, labels)):
        non_target_masks = {}
        for etype in g.canonical_etypes:
            if etype != target_etype:
                non_target_masks[etype] = torch.ones(g.number_of_edges(etype), dtype=torch.bool)

        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask[val_idx] = True

        train_edge_masks = non_target_masks.copy()
        train_edge_masks[target_etype] = train_mask
        val_edge_masks = non_target_masks.copy()
        val_edge_masks[target_etype] = val_mask

        train_graph = dgl.edge_subgraph(g, train_edge_masks, relabel_nodes=False)
        val_graph = dgl.edge_subgraph(g, val_edge_masks, relabel_nodes=False)

        yield fold, train_graph, val_graph



def main():
    """
    Main training pipeline:
      - Load configuration.
      - Train and report validation AUC for each fold.
    """
    config = Config()
    trainer = Trainer(config)

    target_etype = ('receptor', 'binds', 'peptide')
    graph_path = os.path.join(r"../Graph/output/graph.dgl")


    fold_iter = build_kfold_graphs(
        graph_path=graph_path,
        target_etype=target_etype,
        seed=config.seed,
        num_folds=5)

    fold_aucs = []
    for fold, train_graph, val_graph in fold_iter:
        print(f"Starting Training for Fold {fold}")
        best_val_auc = trainer.train_fold(
            fold=fold,
            train_graph=train_graph,
            val_graph=val_graph,
            target_etype=target_etype,
            epochs=config.epochs,
            batch_size=config.batch_size,
        )
        fold_aucs.append(best_val_auc)
        print(f"Fold {fold} completed. Best validation AUC: {best_val_auc:.4f}")


    if fold_aucs:
        avg_auc = sum(fold_aucs) / len(fold_aucs)
        print(f"5-fold training completed. Avg validation AUC: {avg_auc:.4f}")


if __name__ == "__main__":
    """
    Entry point of the script.
    Ensures the main() function is only called when the script is run directly,
    not when imported as a module.
    """
    main()
