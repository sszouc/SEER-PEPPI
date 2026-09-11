# Trainer/feature_processing.py
import torch

class FeatureProcessor:
    @staticmethod
    def process_node_features(G, node_type):
        pretrained_feat = G.nodes[node_type].data['pretrained_feat']
        pretrained_feat = torch.mean(pretrained_feat, dim=1)
        mean = torch.mean(pretrained_feat, dim=0, keepdim=True)
        std = torch.std(pretrained_feat, dim=0, keepdim=True)
        std[std == 0] = 1.0
        normalized_feat = (pretrained_feat - mean) / std
        G.nodes[node_type].data['pre_feat'] = normalized_feat
        print(f"{node_type} pretrained_feat shape: {G.nodes[node_type].data['pretrained_feat'].shape}")
        print(f"{node_type} pre_feat shape: {G.nodes[node_type].data['pre_feat'].shape}")
