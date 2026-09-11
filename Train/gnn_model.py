# trainer/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


class ImprovedProteinGNN(nn.Module):
    """
    The GNN model you provided: projects pretrained features -> two-layer hetero GNN -> edge predictor.
    Assumes node types 'peptide' and 'receptor' with node data 'pre_feat' present.
    """

    def __init__(self):
        super().__init__()


        self.peptide_proj = nn.Linear(1024, 512)
        self.receptor_proj = nn.Linear(1024, 512)

        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)

        self.conv1 = dglnn.HeteroGraphConv({
            'pp_interacts': dglnn.GraphConv(512, 256, norm='both', weight=True, bias=True),
            'rr_interacts': dglnn.GraphConv(512, 256, norm='both', weight=True, bias=True),
            'pp_e_interact': dglnn.GraphConv(512, 256, norm='both', weight=True, bias=True),
            'rr_e_interact': dglnn.GraphConv(512, 256, norm='both', weight=True, bias=True),
        }, aggregate='mean')

        self.conv2 = dglnn.HeteroGraphConv({
            'pp_interacts': dglnn.GraphConv(256, 128, norm='both', weight=True, bias=True),
            'rr_interacts': dglnn.GraphConv(256, 128, norm='both', weight=True, bias=True),
            'pp_e_interact': dglnn.GraphConv(256, 128, norm='both', weight=True, bias=True),
            'rr_e_interact': dglnn.GraphConv(256, 128, norm='both', weight=True, bias=True),
        }, aggregate='mean')

        self.res_proj1 = nn.Linear(512, 256)
        self.res_proj2 = nn.Linear(256, 128)

        # edge predictor expects concat([src_h(128), dst_h(128)]) -> scalar
        self.edge_predictor = nn.Sequential(
            nn.Linear(128 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.aux_edge_predictor = nn.Sequential(
            nn.Linear(128 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.simclr_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, g, etype):
        return self.edge_logits(g, etype, head="main")

    def _compute_node_embeddings_from_feats(self, g, peptide_feat, receptor_feat, exclude_etype=None):
        g = g.to(next(self.parameters()).device)
        supported_rel_names = set(self.conv1.mods.keys())
        msg_etypes = [
            rel for rel in g.canonical_etypes
            if rel[1] in supported_rel_names and rel != exclude_etype
        ]
        g_msg = dgl.edge_type_subgraph(g, msg_etypes) if msg_etypes else g

        peptide_feat = self.peptide_proj(peptide_feat)
        receptor_feat = self.receptor_proj(receptor_feat)

        peptide_feat = self.ln1(peptide_feat)
        receptor_feat = self.ln1(receptor_feat)

        h = {'peptide': peptide_feat, 'receptor': receptor_feat}

        h1 = self.conv1(g_msg, h)
        h1 = {
            'peptide': self.bn2(F.relu(h1['peptide'] + self.res_proj1(h['peptide']))),
            'receptor': self.bn2(F.relu(h1['receptor'] + self.res_proj1(h['receptor'])))
        }

        h2 = self.conv2(g_msg, h1)
        h2 = {
            'peptide': self.bn3(F.relu(h2['peptide'] + self.res_proj2(h1['peptide']))),
            'receptor': self.bn3(F.relu(h2['receptor'] + self.res_proj2(h1['receptor'])))
        }

        return h2

    def node_embeddings(self, g, feat_dropout=0.0, exclude_etype=None):
        peptide_feat = g.nodes['peptide'].data['pre_feat']
        receptor_feat = g.nodes['receptor'].data['pre_feat']
        if feat_dropout and feat_dropout > 0:
            peptide_feat = F.dropout(peptide_feat, p=feat_dropout, training=self.training)
            receptor_feat = F.dropout(receptor_feat, p=feat_dropout, training=self.training)
        return self._compute_node_embeddings_from_feats(g, peptide_feat, receptor_feat, exclude_etype=exclude_etype)

    def project_embeddings(self, h):
        return self.simclr_projector(h)

    @staticmethod
    def _edge_logits_from_embeddings(g, etype, h2, predictor):
        src_type, _, dst_type = etype
        with g.local_scope():
            g.nodes[src_type].data['h'] = h2[src_type]
            g.nodes[dst_type].data['h'] = h2[dst_type]

            def edge_fn_forward(edges):
                src_h = edges.src['h']
                dst_h = edges.dst['h']
                concat = torch.cat([src_h, dst_h], dim=1)
                return {'concat_feat': concat}

            g.apply_edges(edge_fn_forward, etype=etype)
            feats = g.edges[etype].data['concat_feat']
            logits = predictor(feats).squeeze()
        return logits

    def edge_logits(self, g, etype, head="main"):
        h2 = self.node_embeddings(g, feat_dropout=0.0, exclude_etype=etype)
        predictor = self.edge_predictor if head == "main" else self.aux_edge_predictor
        return self._edge_logits_from_embeddings(g, etype, h2, predictor)

    def edge_logits_from_pairs(self, h2, src_type, dst_type, src_ids, dst_ids, head="aux"):
        src_h = h2[src_type][src_ids]
        dst_h = h2[dst_type][dst_ids]
        feats = torch.cat([src_h, dst_h], dim=1)
        predictor = self.edge_predictor if head == "main" else self.aux_edge_predictor
        return predictor(feats).squeeze()
