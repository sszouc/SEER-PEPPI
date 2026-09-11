import os
import pandas as pd


class SequencePreprocessor:
    """
    负责从 CSV 文件读取序列，并生成每个序列单独的 FASTA 文件以及一个 all_sequences.fasta 文件。
    """

    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = csv_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_sequences(self):
        """
        从 CSV 读取序列，并返回去重后的序列列表。
        """
        df = pd.read_csv(self.csv_path, header=None)
        sequences = df[0].dropna().unique().tolist()
        return sequences

    def save_individual_fastas(self, sequences):
        """
        为每个序列生成单独的 FASTA 文件。
        """
        for i, seq in enumerate(sequences):
            fasta_path = os.path.join(self.output_dir, f"sequence_{i+1}.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">sequence_{i+1}\n{seq}\n")

    def save_all_fasta(self, sequences):
        """
        将所有序列写入 all_sequences.fasta 文件。
        """
        all_path = os.path.join(self.output_dir, "all_sequences.fasta")
        with open(all_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">sequence_{i+1}\n{seq}\n")

    def run(self):
        """
        主流程：读取序列并保存每个序列的 FASTA 文件，保存 all_sequences.fasta 文件。
        """
        sequences = self.load_sequences()
        self.save_individual_fastas(sequences)
        self.save_all_fasta(sequences)
        print(f"[SequencePreprocessor] 已完成: {len(sequences)} 条序列")