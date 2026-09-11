import os
import json
import re


class BlastResultParser:
    """
    解析 extracted_*.txt 文件，提取字典（序列名称 → e_value），并保存为 JSON 格式。
    同时支持将 ID 替换为实际序列。
    """

    def __init__(self, extracted_dir, fasta_mapping, output_json):
        """
        参数：
            extracted_dir: 提取的 round_k 文件所在目录
            fasta_mapping: {sequence_id: actual_sequence} 字典
            output_json: 输出 JSON 文件的路径
        """
        self.extracted_dir = extracted_dir
        self.fasta_mapping = fasta_mapping
        self.output_json = output_json

    def parse_file(self, file_path):
        """
        解析单个提取文件中的查询序列和显著命中结果。
        跳过非命中部分，如标题或模型使用的序列。
        返回 (query_name, {hit: e_value})
        """
        sequences = {}
        query_name = None
        start_extracting = False
        in_significant_section = False

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 检测查询序列行
                if line.startswith("Query="):
                    query_name = line.split("=", 1)[1].strip()
                    continue

                # 轮次部分开始
                if re.search(r"Results from round \d+", line):
                    start_extracting = True
                    in_significant_section = False
                    continue

                if not start_extracting:
                    continue

                # 节标题 —— 命中表开始
                if "Sequences producing significant alignments:" in line:
                    in_significant_section = True
                    continue

                # 子标题（仍属于命中表，不应停止）
                if (
                    "Sequences used in model and found again:" in line or
                    "Sequences not found previously or not previously below threshold:" in line
                ):
                    in_significant_section = True
                    continue

                # 命中表结束：进入对齐细节或统计段
                if line.startswith(">") or line.startswith("Lambda") or line.startswith("Effective search space"):
                    in_significant_section = False
                    break

                # 仅在显著命中节中解析命中行
                if in_significant_section:
                    parts = line.split()
                    if len(parts) >= 2:
                        seq_id = parts[0]
                        if not seq_id.startswith("sequence_"):
                            continue
                        try:
                            e_val = float(parts[-1])
                            sequences[seq_id] = e_val
                        except ValueError:
                            continue  # 跳过最后一列不是数字的行

        return query_name, sequences

    def load_extracted(self):
        """
        解析所有提取文件，返回 {query: {hit: e_value}}
        """
        all_results = {}
        for fname in os.listdir(self.extracted_dir):
            if fname.startswith("extracted_") and fname.endswith(".txt"):
                print("准备提取"+fname)
                file_path = os.path.join(self.extracted_dir, fname)
                q, seqs = self.parse_file(file_path)
                if q and seqs:  # 仅当查询序列和命中结果都存在时包含
                    all_results[q] = seqs
        return all_results

    def replace_ids(self, data):
        """
        将查询序列和命中结果的 ID 替换为实际序列。
        """
        replaced = {}
        for q, hits in data.items():
            q_seq = self.fasta_mapping.get(q, q)
            new_hits = {self.fasta_mapping.get(h, h): v for h, v in hits.items()}
            replaced[q_seq] = new_hits
        return replaced

    def run(self):
        raw = self.load_extracted()
        replaced = self.replace_ids(raw)

        with open(self.output_json, "w") as f:
            json.dump(replaced, f, indent=4)

        print(f"[BlastResultParser] JSON 已保存至 {self.output_json}")