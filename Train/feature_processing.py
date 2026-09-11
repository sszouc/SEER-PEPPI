# trainer/feature_processing.py
import torch

def process_node_features(G, node_type):
    """
    Original script used pretrained_feat with shape (N, seq_len, feat_dim),
    then averaged across seq_len and standardized to make 'pre_feat'.
    """
    if 'pretrained_feat' not in G.nodes[node_type].data:
        raise KeyError(f"Node type {node_type} missing 'pretrained_feat'")

    pretrained_feat = G.nodes[node_type].data['pretrained_feat']  # (N, L, D)

    # average across sequence length
    pretrained_feat = torch.mean(pretrained_feat, dim=1)

    mean = torch.mean(pretrained_feat, dim=0, keepdim=True)
    std = torch.std(pretrained_feat, dim=0, keepdim=True)
    std[std == 0] = 1.0

    normalized_feat = (pretrained_feat - mean) / std
    G.nodes[node_type].data['pre_feat'] = normalized_feat

    # # debug prints (kept from original)
    # print(f"{node_type} pretrained_feat shape: {G.nodes[node_type].data['pretrained_feat'].shape}")
    # print(f"{node_type} pre_feat shape: {G.nodes[node_type].data['pre_feat'].shape}")
