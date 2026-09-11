import json
import pickle
import torch
import dgl
from pathlib import Path
import os


class TestGraphExtractor:
    """
    负责：
    makeblastdb.bat. 加载完整的DGL图以及肽/受体ID映射。
    2. 选择对应测试肽-受体对的边。
    3. 提取所选边涉及的所有节点。
    4. 收集所有相关边（包括交互边和e_interact边）。
    5. 使用所选节点和边构建子图。
    6. 将生成的子图保存为DGL文件。
    """

    def __init__(self, graph_file, receptor_json, peptide_json, test_pkl_file, save_path):
        """
        初始化文件路径并加载映射关系。
        :param graph_file: DGL图文件路径
        :param receptor_json: receptor_to_id JSON文件路径
        :param peptide_json: peptide_to_id JSON文件路径
        :param test_pkl_file: 测试数据pickle文件路径
        :param save_path: 保存子图的路径
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置GPU设备
        self.graph_file = Path(graph_file)
        self.receptor_json = Path(receptor_json)
        self.peptide_json = Path(peptide_json)
        self.test_pkl_file = Path(test_pkl_file)
        self.save_path = Path(save_path)

        # 加载图和映射关系
        self.G = dgl.load_graphs(str(self.graph_file))[0][0]

        with open(self.receptor_json, 'r') as f:
            self.receptor_to_id = json.load(f)
        with open(self.peptide_json, 'r') as f:
            self.peptide_to_id = json.load(f)

        self.id_to_receptor = {v: k for k, v in self.receptor_to_id.items()}
        self.id_to_peptide = {v: k for k, v in self.peptide_to_id.items()}

        # 加载测试数据对
        with open(self.test_pkl_file, 'rb') as f:
            test_data = pickle.load(f)
        self.test_receptor_peptide_pairs = set((rec, pep) for rec, _, pep, _, _ in test_data)

    # 查找符合测试条件的受体-肽结合边
    def _select_binds_edges(self):
        """选择对应测试肽-受体对的边。"""
        src, dst = self.G.edges(etype=('receptor', 'binds', 'peptide'))
        src_names = [self.id_to_receptor.get(i, None) for i in src.tolist()]
        dst_names = [self.id_to_peptide.get(i, None) for i in dst.tolist()]

        selected_edge_mask = torch.tensor([
            (s is not None and d is not None and (s, d) in self.test_receptor_peptide_pairs)
            for s, d in zip(src_names, dst_names)
        ])

        print(f"找到 {selected_edge_mask.sum().item()} 条需要保留的'binds'边。")
        return selected_edge_mask, src, dst

    # 收集子图的相关边
    def _collect_edges_to_keep(self, selected_receptors, selected_peptides):
        """收集涉及所选受体和肽的所有边。"""
        edges_to_keep = {}

        # 受体-结合-肽边
        src, dst = self.G.edges(etype=('receptor', 'binds', 'peptide'))
        binds_mask = torch.isin(src, selected_receptors) | torch.isin(dst, selected_peptides)
        edges_to_keep[('receptor', 'binds', 'peptide')] = binds_mask.nonzero(as_tuple=True)[0]

        # 受体-交互-受体边
        src_rec, dst_rec = self.G.edges(etype=('receptor', 'interacts', 'receptor'))
        rec_mask = torch.isin(src_rec, selected_receptors) | torch.isin(dst_rec, selected_receptors)
        edges_to_keep[('receptor', 'interacts', 'receptor')] = rec_mask.nonzero(as_tuple=True)[0]

        # 肽-交互-肽边
        src_pep, dst_pep = self.G.edges(etype=('peptide', 'interacts', 'peptide'))
        pep_mask = torch.isin(src_pep, selected_peptides) | torch.isin(dst_pep, selected_peptides)
        edges_to_keep[('peptide', 'interacts', 'peptide')] = pep_mask.nonzero(as_tuple=True)[0]

        # 肽-e交互-肽边
        src_pep_e, dst_pep_e = self.G.edges(etype=('peptide', 'e_interact', 'peptide'))
        pep_e_mask = torch.isin(src_pep_e, selected_peptides) | torch.isin(dst_pep_e, selected_peptides)
        edges_to_keep[('peptide', 'e_interact', 'peptide')] = pep_e_mask.nonzero(as_tuple=True)[0]

        # 受体-e交互-受体边
        src_rec_e, dst_rec_e = self.G.edges(etype=('receptor', 'e_interact', 'receptor'))
        rec_e_mask = torch.isin(src_rec_e, selected_receptors) | torch.isin(dst_rec_e, selected_receptors)
        edges_to_keep[('receptor', 'e_interact', 'receptor')] = rec_e_mask.nonzero(as_tuple=True)[0]

        print("边收集完成。")
        return edges_to_keep

    # 生成并保存子图
    def extract_subgraph(self):

        """提取包含所有选定节点和边的子图，并保存。"""
        # makeblastdb.bat. 找到需要保留的binds边。
        selected_edge_mask, src, dst = self._select_binds_edges()

        # 2. 提取涉及的受体和肽
        selected_receptors = src[selected_edge_mask].unique()
        selected_peptides = dst[selected_edge_mask].unique()
        print(f"涉及 {len(selected_receptors)} 个受体，{len(selected_peptides)} 个肽。")

        # 3. 收集所有相关边
        edges_to_keep = self._collect_edges_to_keep(selected_receptors, selected_peptides)

        # 4. 构建子图
        subgraph = self.G.edge_subgraph(edges_to_keep)
        print(f"子图信息: {subgraph}")

        # # 5. 保存子图
        # dgl.save_graphs(str(self.save_path), [subgraph])
        # print(f"子图已成功保存至 {self.save_path}")

        try:
            # 确保目录存在
            save_dir = Path(self.save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # 检查图是否有效
            if subgraph.num_nodes() == 0:
                print("错误: 子图没有节点")
                return

            # 尝试保存
            dgl.save_graphs(str(self.save_path), [subgraph])
            print(f"成功保存子图到: {self.save_path}")

        except Exception as e:
            print(f"保存失败: {e}")
            # 备用方案：使用 pickle
            import pickle
            with open(str(self.save_path).replace('.bin', '.pkl'), 'wb') as f:
                pickle.dump(subgraph, f)
            print(f"使用 pickle 保存到备用文件")