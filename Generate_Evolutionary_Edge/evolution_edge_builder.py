import os
import re
import math
from tqdm import tqdm


class EvolutionEdgeBuilder:
    """
    将 E-value 文件转换为最终模型中可用的边文件。
    示例格式：
        query_seq	hit_seq	score

    支持的分数转换：
      - 'none': 原始 E-value
      - 'neg_log10': -log10(E + 1e-180)  [推荐用于显著性加权]
      - 'exp_neg': exp(-E)  [不推荐，仅为兼容性保留]
    """

    def __init__(self, input_dir, seq_mapping, output_file, transform='none'):
        """
        参数：
            input_dir: 包含类似 "sequence_123.txt" 文件的目录，每行格式为 "sequence_456, 1e-10"
            seq_mapping: {sequence_id: actual_amino_acid_sequence}
            output_file: 输出 TSV 文件路径
            transform: 可选值 {'none', 'neg_log10', 'exp_neg'}
        """
        self.input_dir = input_dir
        self.seq_mapping = seq_mapping
        self.output_file = output_file
        if transform not in {'none', 'neg_log10', 'exp_neg'}:
            raise ValueError("transform 必须是以下值之一：'none', 'neg_log10', 'exp_neg'")
        self.transform = transform

        if self.transform == 'exp_neg':
            print("[警告] 使用 exp(-E) 转换在生物学上存在争议。"
                  "建议改用 'neg_log10'。")

    def _transform_score(self, e_val):
        """安全地应用分数转换。"""
        if self.transform == 'none':
            return e_val
        elif self.transform == 'neg_log10':
            # 避免 log(0)；E-value 始终 >= 0
            return -math.log10(e_val + 1e-180)
        elif self.transform == 'exp_neg':
            # 限制 E 值以避免下溢（exp(-700) ~ 1e-304，接近浮点数最小值）
            if e_val > 700:
                return 0.0
            return math.exp(-e_val)
        else:
            return e_val  # 回退

    def process_line(self, line):
        """
        解析一行 "sequence_1234, 1e-45"
        返回 (hit_seq, transformed_score) 或 None
        """
        match = re.match(r"(sequence_\d+),\s*([\deE\.\-\+]+)", line.strip())
        if not match:
            return None

        seq_id, e_val_str = match.groups()
        try:
            e_val = float(e_val_str)
        except ValueError:
            return None

        if e_val > 0.5:
            return None

        if seq_id not in self.seq_mapping:
            return None

        hit_seq = self.seq_mapping[seq_id]
        score = self._transform_score(e_val)




        return hit_seq, score

    def process_file(self, file_path, output_buffer):
        """
        从 E-value 文件中提取命中结果并写入缓冲区。
        """
        basename = os.path.basename(file_path)
        # 提取查询 ID，如 "sequence_123"
        query_match = re.search(r"(sequence_\d+)", basename)
        if not query_match:
            return
        query_id = query_match.group(1)
        query_seq = self.seq_mapping.get(query_id)
        if query_seq is None:
            return

        with open(file_path) as f:
            for line in f:
                res = self.process_line(line)
                if res:
                    hit_seq, score = res
                    output_buffer.append(f"{query_seq},{hit_seq},{score:.6e}\n")

    def run(self):
        buffer = []

        for fname in tqdm(os.listdir(self.input_dir), desc="构建进化边"):
            if fname.endswith(".txt"):
                self.process_file(os.path.join(self.input_dir, fname), buffer)

        with open(self.output_file, "w") as f:
            f.writelines(buffer)

        print(f"[EvolutionEdgeBuilder] 已生成 {len(buffer)} 条进化边。输出文件：{self.output_file}")