import os
import pickle
import torch
import dgl


class EdgeBuilder:

    def __init__(self, device='0'):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        # 初始化容器
        self.receptor_to_peptides_positive = {}
        self.receptor_to_peptides_negative = {}
        self.peptide_to_id = {}
        self.receptor_to_id = {}
        self.all_peptides = []
        self.all_receptors = []
        self.G = None

    # 读取正负样本数据
    def load_samples(self, positive_file, negative_file):
        with open(positive_file, 'r') as f:
            for line in f:
                peptide, receptor = line.strip().split()
                self.receptor_to_peptides_positive.setdefault(receptor, []).append(peptide)

        with open(negative_file, 'r') as f:
            for line in f:
                peptide, receptor = line.strip().split()
                self.receptor_to_peptides_negative.setdefault(receptor, []).append(peptide)

        self.all_peptides = sorted(set(
            peptide for peptides in self.receptor_to_peptides_positive.values() for peptide in peptides
        ).union(
            peptide for peptides in self.receptor_to_peptides_negative.values() for peptide in peptides
        ))
        self.all_receptors = sorted(set(
            self.receptor_to_peptides_positive.keys()
        ).union(self.receptor_to_peptides_negative.keys()))

        self.peptide_to_id = {p: idx for idx, p in enumerate(self.all_peptides)}
        self.receptor_to_id = {r: idx for idx, r in enumerate(self.all_receptors)}

    @staticmethod
    def add_new_node(node, node_to_id, current_max_id):
        if node not in node_to_id:
            node_to_id[node] = current_max_id
            current_max_id += 1
        return node_to_id[node], current_max_id

    # 创建受体-肽段边
    def build_receptor_peptide_edges(self):
        edges_positive = [
            (self.receptor_to_id[r], self.peptide_to_id[p])
            for r, peptides in self.receptor_to_peptides_positive.items()
            for p in peptides
        ]
        edges_negative = [
            (self.receptor_to_id[r], self.peptide_to_id[p])
            for r, peptides in self.receptor_to_peptides_negative.items()
            for p in peptides
        ]

        edges = edges_positive + edges_negative
        labels = [1] * len(edges_positive) + [0] * len(edges_negative)  # 正样本标签为1，负样本标签为0
        return edges, labels

    @staticmethod
    def build_similarity_edges(similarity_csv, node_to_id, threshold):
        src_nodes, dst_nodes, scores = EdgeBuilder.build_extra_edges(similarity_csv, node_to_id)
        edges, features = [], []
        for src, dst, score in zip(src_nodes, dst_nodes, scores):
            if score > threshold:
                edges.append((src, dst))
                features.append(score)
        return edges, features

    @staticmethod
    def build_extra_edges(file_path, node_to_id):
        src_nodes, dst_nodes, scores = [], [], []
        with open(file_path, 'r') as f:
            for line in f:
                node1, node2, score = line.strip().split(',')
                node1 = node1.strip()
                node2 = node2.strip()
                if node1 in node_to_id and node2 in node_to_id:
                    src_nodes.append(node_to_id[node1])
                    dst_nodes.append(node_to_id[node2])
                    scores.append(float(score))
        return src_nodes, dst_nodes, scores

    @staticmethod
    def build_peptide_cosine_edges(cosine_file, node_to_id, threshold_low,threshold_high):
        edges, features = [], []
        with open(cosine_file, 'rb') as f:
            similarity_dict = pickle.load(f)
        for node1, sims in similarity_dict.items():
            id1 = node_to_id.get(node1)
            if id1 is None:
                continue
            for node2, sim in sims.items():
                id2 = node_to_id.get(node2)
                if id2 is None or sim <= threshold_low or sim >= threshold_high:
                    continue
                edges.append((id1, id2))
                features.append(sim)
        return edges, features

    def construct_graph(self, pep_edges_file=None, pro_edges_file=None,
                        peptide_cos_file=None, protein_cos_file=None,
                        pep_threshold_low=0.62, pep_threshold_high=1, pro_threshold_low=0.41, pro_threshold_high=1):

        edges_rp, labels_rp = self.build_receptor_peptide_edges()

        edges_pp, feat_pp = self.build_peptide_cosine_edges(peptide_cos_file, self.peptide_to_id, pep_threshold_low,
                                                            pep_threshold_high)
        edges_rr, feat_rr = self.build_peptide_cosine_edges(protein_cos_file, self.receptor_to_id, pro_threshold_low,
                                                            pro_threshold_high)

        pep_src, pep_dst, pep_feat_e = self.build_extra_edges(pep_edges_file, self.peptide_to_id)
        pro_src, pro_dst, pro_feat_e = self.build_extra_edges(pro_edges_file, self.receptor_to_id)

        self.G = dgl.heterograph({
            ('receptor', 'binds', 'peptide'): edges_rp,
            ('peptide', 'pp_interacts', 'peptide'): edges_pp,
            ('receptor', 'rr_interacts', 'receptor'): edges_rr,
            ('peptide', 'pp_e_interact', 'peptide'): (pep_src, pep_dst),
            ('receptor', 'rr_e_interact', 'receptor'): (pro_src, pro_dst),
        })

        if labels_rp:
            self.G.edges[('receptor', 'binds', 'peptide')].data['label'] = torch.tensor(labels_rp)
        if feat_pp:
            self.G.edges[('peptide', 'pp_interacts', 'peptide')].data['similarity'] = torch.tensor(feat_pp)
        if feat_rr:
            self.G.edges[('receptor', 'rr_interacts', 'receptor')].data['similarity'] = torch.tensor(feat_rr)
        if pep_feat_e:
            self.G.edges[('peptide', 'pp_e_interact', 'peptide')].data['score'] = torch.tensor(pep_feat_e,dtype=torch.float32)
        if pro_feat_e:
            self.G.edges[('receptor', 'rr_e_interact', 'receptor')].data['score'] = torch.tensor(pro_feat_e,dtype=torch.float32)

        print(self.G)
