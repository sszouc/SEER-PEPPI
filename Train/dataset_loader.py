# trainer/dataset_loader.py
import dgl
from .feature_processing import process_node_features

class DatasetLoader:
    """
    Simple loader to read train/val fold graphs (as in your original main).
    Assumes dataset files follow pattern: train_fold_{i}.dgl and val_fold_{i}.dgl
    """
    def __init__(self, base_dir=r"D:\pyprogram\HGT-PepPI\Train\fold"):
        self.base_dir = base_dir

    def load_fold_graphs(self, fold_idx):
        train_path = f"{self.base_dir}/train_fold_{fold_idx}.dgl"
        val_path = f"{self.base_dir}/val_fold_{fold_idx}.dgl"
        train_graph = dgl.load_graphs(train_path)[0][0]
        val_graph = dgl.load_graphs(val_path)[0][0]
        # prepare pre_feat from pretrained_feat if present
        for g in (train_graph, val_graph):
            for ntype in g.ntypes:
                if 'pretrained_feat' in g.nodes[ntype].data:
                    process_node_features(g, ntype)
        return train_graph, val_graph
