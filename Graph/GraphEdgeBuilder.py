import os
import pickle
import torch
import dgl


class EdgeBuilder:
    """
    负责：
    1. 读取正负样本的受体-肽段对
    2. 创建受体-肽段相互作用的初始边
    3. 加载相似性数据以及额外的肽段-肽段或受体-受体边
    4. 构建DGL异构图
    5. 分配边特征，如标签、相似性分数和e_interact分数
    """

    def __init__(self, device='0'):
        # 配置GPU设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        print(f"GPU可用性检查：{torch.cuda.is_available()}")

        # 初始化容器
        self.receptor_to_peptides_positive = {}  # 受体到正样本肽段的映射
        self.receptor_to_peptides_negative = {}  # 受体到负样本肽段的映射
        self.peptide_to_id = {}  # 肽段到ID的映射
        self.receptor_to_id = {}  # 受体到ID的映射
        self.all_peptides = []  # 所有肽段列表
        self.all_receptors = []  # 所有受体列表
        self.G = None  # DGL图对象

    # 读取正负样本数据
    def load_samples(self, positive_file, negative_file):
        """加载正负样本的受体-肽段对。"""
        # 读取正样本
        with open(positive_file, 'r') as f:
            for line in f:
                peptide, receptor = line.strip().split()
                self.receptor_to_peptides_positive.setdefault(receptor, []).append(peptide)

        # 读取负样本
        with open(negative_file, 'r') as f:
            for line in f:
                peptide, receptor = line.strip().split()
                self.receptor_to_peptides_negative.setdefault(receptor, []).append(peptide)

        # 获取所有肽段和受体
        self.all_peptides = sorted(set(
            peptide for peptides in self.receptor_to_peptides_positive.values() for peptide in peptides
        ).union(
            peptide for peptides in self.receptor_to_peptides_negative.values() for peptide in peptides
        ))
        self.all_receptors = sorted(set(
            self.receptor_to_peptides_positive.keys()
        ).union(self.receptor_to_peptides_negative.keys()))

        # 创建初始映射
        self.peptide_to_id = {p: idx for idx, p in enumerate(self.all_peptides)}
        self.receptor_to_id = {r: idx for idx, r in enumerate(self.all_receptors)}

    # 辅助函数：动态添加节点
    @staticmethod
    def add_new_node(node, node_to_id, current_max_id):
        """如果节点不存在，则将其添加到映射中。"""
        if node not in node_to_id:
            node_to_id[node] = current_max_id
            current_max_id += 1
        return node_to_id[node], current_max_id

    # 创建受体-肽段边
    def build_receptor_peptide_edges(self):
        """创建从受体到肽段的边，包括正负样本，并附带标签。"""
        # 正样本边
        edges_positive = [
            (self.receptor_to_id[r], self.peptide_to_id[p])
            for r, peptides in self.receptor_to_peptides_positive.items()
            for p in peptides
        ]
        # 负样本边
        edges_negative = [
            (self.receptor_to_id[r], self.peptide_to_id[p])
            for r, peptides in self.receptor_to_peptides_negative.items()
            for p in peptides
        ]

        # 合并边
        edges = edges_positive + edges_negative
        labels = [1] * len(edges_positive) + [0] * len(edges_negative)  # 正样本标签为1，负样本标签为0
        return edges, labels

    # 加载相似性文件并构建额外的边（CSV）
    @staticmethod
    def build_similarity_edges(similarity_csv, node_to_id, threshold):
        """
        基于相似性CSV构建边（格式：node1,node2,score）。
        返回边列表和相似度特征。
        """
        src_nodes, dst_nodes, scores = EdgeBuilder.build_extra_edges(similarity_csv, node_to_id)
        edges, features = [], []
        for src, dst, score in zip(src_nodes, dst_nodes, scores):
            if score > threshold:
                edges.append((src, dst))
                features.append(score)
        return edges, features

    # 读取额外的边（肽段/受体的e_interact）
    @staticmethod
    def build_extra_edges(file_path, node_to_id):
        """
        从CSV文件中读取额外的边，格式：node1,node2,score
        返回源节点列表、目标节点列表和分数列表。
        """
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

    # 加载肽段余弦相似度文件并构建边
    @staticmethod
    def build_peptide_cosine_edges(cosine_file, node_to_id, threshold_low,threshold_high):
        """
        基于余弦相似度pickle构建肽段-肽段边。
        返回边列表和相似度特征。
        """
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

    # 构建DGL图
    def construct_graph(self, pep_edges_file=None, pro_edges_file=None,
                        peptide_cos_file=None, protein_cos_file=None,
                        pep_threshold_low=0.62, pep_threshold_high=1, pro_threshold_low=0.41, pro_threshold_high=1):
        """构建带有边特征的DGL异构图。"""

        # 1. 受体-肽段边
        edges_rp, labels_rp = self.build_receptor_peptide_edges()

        # 2. 肽段-肽段相似性边（余弦pickle）
        edges_pp, feat_pp = self.build_peptide_cosine_edges(peptide_cos_file, self.peptide_to_id, pep_threshold_low,
                                                            pep_threshold_high)
        # 3. 受体-受体相似性边（CSV）
        edges_rr, feat_rr = self.build_peptide_cosine_edges(protein_cos_file, self.receptor_to_id, pro_threshold_low,
                                                            pro_threshold_high)

        # 4. 额外的e_interact边
        pep_src, pep_dst, pep_feat_e = self.build_extra_edges(pep_edges_file, self.peptide_to_id)
        pro_src, pro_dst, pro_feat_e = self.build_extra_edges(pro_edges_file, self.receptor_to_id)

        # 5. 创建DGL异构图
        self.G = dgl.heterograph({
            ('receptor', 'binds', 'peptide'): edges_rp,  # 受体结合肽段
            ('peptide', 'pp_interacts', 'peptide'): edges_pp,  # 肽段-肽段相互作用
            ('receptor', 'rr_interacts', 'receptor'): edges_rr,  # 受体-受体相互作用
            ('peptide', 'pp_e_interact', 'peptide'): (pep_src, pep_dst),  # 肽段e_interact边
            ('receptor', 'rr_e_interact', 'receptor'): (pro_src, pro_dst),  # 受体e_interact边
        })

        # 6. 分配边特征
        if labels_rp:  # 受体-肽段标签
            self.G.edges[('receptor', 'binds', 'peptide')].data['label'] = torch.tensor(labels_rp)
        if feat_pp:
            self.G.edges[('peptide', 'pp_interacts', 'peptide')].data['similarity'] = torch.tensor(feat_pp)
        if feat_rr:
            self.G.edges[('receptor', 'rr_interacts', 'receptor')].data['similarity'] = torch.tensor(feat_rr)
        if pep_feat_e:
            self.G.edges[('peptide', 'pp_e_interact', 'peptide')].data['score'] = torch.tensor(pep_feat_e,dtype=torch.float32)
        if pro_feat_e:
            self.G.edges[('receptor', 'rr_e_interact', 'receptor')].data['score'] = torch.tensor(pro_feat_e,dtype=torch.float32)

        # 7. 验证图结构
        print(self.G)
        print("受体-肽段标签是否存在:", 'label' in self.G.edges[('receptor', 'binds', 'peptide')].data)
        print("肽段-肽段相似性是否存在:", 'similarity' in self.G.edges[('peptide', 'pp_interacts', 'peptide')].data)
        print("受体-受体相似性是否存在:", 'similarity' in self.G.edges[('receptor', 'rr_interacts', 'receptor')].data)
        print("肽段-肽段e_interact分数是否存在:", 'score' in self.G.edges[('peptide', 'pp_e_interact', 'peptide')].data)
        print("受体-受体e_interact分数是否存在:", 'score' in self.G.edges[('receptor', 'rr_e_interact', 'receptor')].data)
