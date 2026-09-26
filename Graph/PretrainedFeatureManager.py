import os
import json
import pickle
import torch
import numpy as np
import dgl
from pathlib import Path

class PretrainedFeatureManager:
    def __init__(self, G, peptide_to_id, receptor_to_id,
                 peptide_feature_file, receptor_feature_file,
                 save_dir, graph_file_name='graph_new.dgl'):
        self.G = G
        self.peptide_to_id = peptide_to_id
        self.receptor_to_id = receptor_to_id
        self.id_to_peptide = {v: k for k, v in peptide_to_id.items()}
        self.id_to_receptor = {v: k for k, v in receptor_to_id.items()}

        self.peptide_feature_file = Path(peptide_feature_file)
        self.receptor_feature_file = Path(receptor_feature_file)

        self.peptides_missing_files = []
        self.receptors_missing_files = []

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.graph_file_name = graph_file_name

    def set_pretrained_features(self, max_length_peptide=50, max_length_receptor=800, feature_dim=1024):
        with open(self.peptide_feature_file, 'rb') as f:
            peptide_features = pickle.load(f)
        with open(self.receptor_feature_file, 'rb') as f:
            receptor_features = pickle.load(f)

        num_peptide_nodes = self.G.num_nodes('peptide')
        num_receptor_nodes = self.G.num_nodes('receptor')

        peptide_feats_tensor = torch.zeros((num_peptide_nodes, max_length_peptide, feature_dim), dtype=torch.float32)
        for nid, peptide in self.id_to_peptide.items():
            if peptide in peptide_features and nid < num_peptide_nodes:
                feats = np.array(peptide_features[peptide])
                feats = self._pad_or_truncate(feats, max_length_peptide, feature_dim)
                peptide_feats_tensor[nid] = torch.tensor(feats, dtype=torch.float32)
        self.G.nodes['peptide'].data['pretrained_feat'] = peptide_feats_tensor

        receptor_feats_tensor = torch.zeros((num_receptor_nodes, max_length_receptor, feature_dim), dtype=torch.float32)
        for nid, receptor in self.id_to_receptor.items():
            if receptor in receptor_features and nid < num_receptor_nodes:
                feats = np.array(receptor_features[receptor])
                feats = self._pad_or_truncate(feats, max_length_receptor, feature_dim)
                receptor_feats_tensor[nid] = torch.tensor(feats, dtype=torch.float32)
        self.G.nodes['receptor'].data['pretrained_feat'] = receptor_feats_tensor


    def _pad_or_truncate(self, features, max_length, feature_dim):
        if features.shape[0] < max_length:
            padding = np.zeros((max_length - features.shape[0], feature_dim))
            features = np.vstack((features, padding))
        elif features.shape[0] > max_length:
            features = features[:max_length, :]
        return features

    def remove_nodes_without_features(self):
        peptides_missing = set().union(*self.peptides_missing_files)
        receptors_missing = set().union(*self.receptors_missing_files)

        peptide_nodes_to_remove = [nid for nid, pep in self.id_to_peptide.items() if pep in peptides_missing]
        receptor_nodes_to_remove = [nid for nid, rec in self.id_to_receptor.items() if rec in receptors_missing]

        peptide_nodes_to_remove = torch.tensor(peptide_nodes_to_remove, dtype=self.G.idtype, device=self.G.device)
        receptor_nodes_to_remove = torch.tensor(receptor_nodes_to_remove, dtype=self.G.idtype, device=self.G.device)

        if peptide_nodes_to_remove.numel() > 0:
            self.G = dgl.remove_nodes(self.G, peptide_nodes_to_remove, ntype='peptide')
            print(f"After removing peptide nodes: {self.G}")
        if receptor_nodes_to_remove.numel() > 0:
            self.G = dgl.remove_nodes(self.G, receptor_nodes_to_remove, ntype='receptor')
            print(f"After removing receptor nodes: {self.G}")

        remaining_peptide_ids = set(range(len(self.peptide_to_id))) - set(peptide_nodes_to_remove.tolist())
        remaining_receptor_ids = set(range(len(self.receptor_to_id))) - set(receptor_nodes_to_remove.tolist())

        id_to_peptide = {v: k for k, v in self.peptide_to_id.items()}
        id_to_receptor = {v: k for k, v in self.receptor_to_id.items()}

        self.peptide_to_id = {id_to_peptide[old_id]: new_id for new_id, old_id in enumerate(sorted(remaining_peptide_ids))}
        self.receptor_to_id = {id_to_receptor[old_id]: new_id for new_id, old_id in enumerate(sorted(remaining_receptor_ids))}

        with open(self.save_dir / 'peptide_to_id.json', 'w') as f:
            json.dump(self.peptide_to_id, f, indent=2)
        with open(self.save_dir / 'receptor_to_id.json', 'w') as f:
            json.dump(self.receptor_to_id, f, indent=2)

    def save_graph(self):
        os.makedirs(self.save_dir / 'graph', exist_ok=True)
        save_path = self.save_dir / 'graph' / self.graph_file_name
        dgl.save_graphs(str(save_path), [self.G])
        print(f"Graph saved to {save_path}")


