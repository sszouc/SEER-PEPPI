import json
import pickle
import torch
import dgl
from pathlib import Path
import os


class TestGraphExtractor:
    def __init__(self, graph_file, receptor_json, peptide_json, test_pkl_file, save_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.graph_file = Path(graph_file)
        self.receptor_json = Path(receptor_json)
        self.peptide_json = Path(peptide_json)
        self.test_pkl_file = Path(test_pkl_file)
        self.save_path = Path(save_path)


        self.G = dgl.load_graphs(str(self.graph_file))[0][0]

        with open(self.receptor_json, 'r') as f:
            self.receptor_to_id = json.load(f)
        with open(self.peptide_json, 'r') as f:
            self.peptide_to_id = json.load(f)

        self.id_to_receptor = {v: k for k, v in self.receptor_to_id.items()}
        self.id_to_peptide = {v: k for k, v in self.peptide_to_id.items()}

        with open(self.test_pkl_file, 'rb') as f:
            test_data = pickle.load(f)
        self.test_receptor_peptide_pairs = set((rec, pep) for rec, _, pep, _, _ in test_data)

    def _select_binds_edges(self):
        src, dst = self.G.edges(etype=('receptor', 'binds', 'peptide'))
        src_names = [self.id_to_receptor.get(i, None) for i in src.tolist()]
        dst_names = [self.id_to_peptide.get(i, None) for i in dst.tolist()]

        selected_edge_mask = torch.tensor([
            (s is not None and d is not None and (s, d) in self.test_receptor_peptide_pairs)
            for s, d in zip(src_names, dst_names)
        ])

        return selected_edge_mask, src, dst

    def _collect_edges_to_keep(self, selected_receptors, selected_peptides):
        edges_to_keep = {}

        src, dst = self.G.edges(etype=('receptor', 'binds', 'peptide'))
        binds_mask = torch.isin(src, selected_receptors) | torch.isin(dst, selected_peptides)
        edges_to_keep[('receptor', 'binds', 'peptide')] = binds_mask.nonzero(as_tuple=True)[0]

        src_rec, dst_rec = self.G.edges(etype=('receptor', 'interacts', 'receptor'))
        rec_mask = torch.isin(src_rec, selected_receptors) | torch.isin(dst_rec, selected_receptors)
        edges_to_keep[('receptor', 'interacts', 'receptor')] = rec_mask.nonzero(as_tuple=True)[0]

        src_pep, dst_pep = self.G.edges(etype=('peptide', 'interacts', 'peptide'))
        pep_mask = torch.isin(src_pep, selected_peptides) | torch.isin(dst_pep, selected_peptides)
        edges_to_keep[('peptide', 'interacts', 'peptide')] = pep_mask.nonzero(as_tuple=True)[0]

        src_pep_e, dst_pep_e = self.G.edges(etype=('peptide', 'e_interact', 'peptide'))
        pep_e_mask = torch.isin(src_pep_e, selected_peptides) | torch.isin(dst_pep_e, selected_peptides)
        edges_to_keep[('peptide', 'e_interact', 'peptide')] = pep_e_mask.nonzero(as_tuple=True)[0]

        src_rec_e, dst_rec_e = self.G.edges(etype=('receptor', 'e_interact', 'receptor'))
        rec_e_mask = torch.isin(src_rec_e, selected_receptors) | torch.isin(dst_rec_e, selected_receptors)
        edges_to_keep[('receptor', 'e_interact', 'receptor')] = rec_e_mask.nonzero(as_tuple=True)[0]

        return edges_to_keep

    def extract_subgraph(self):

        selected_edge_mask, src, dst = self._select_binds_edges()

        selected_receptors = src[selected_edge_mask].unique()
        selected_peptides = dst[selected_edge_mask].unique()
        print(f"涉及 {len(selected_receptors)} 个受体，{len(selected_peptides)} 个肽。")

        edges_to_keep = self._collect_edges_to_keep(selected_receptors, selected_peptides)

        subgraph = self.G.edge_subgraph(edges_to_keep)
        print(f"子图信息: {subgraph}")

        save_dir = Path(self.save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        if subgraph.num_nodes() == 0:
            return

        dgl.save_graphs(str(self.save_path), [subgraph])