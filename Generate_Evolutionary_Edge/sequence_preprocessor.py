import os
import pandas as pd


class SequencePreprocessor:

    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = csv_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_sequences(self):
        df = pd.read_csv(self.csv_path, header=None)
        sequences = df[0].dropna().unique().tolist()
        return sequences

    def save_individual_fastas(self, sequences):
        for i, seq in enumerate(sequences):
            fasta_path = os.path.join(self.output_dir, f"sequence_{i+1}.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">sequence_{i+1}\n{seq}\n")

    def save_all_fasta(self, sequences):
        all_path = os.path.join(self.output_dir, "all_sequences.fasta")
        with open(all_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">sequence_{i+1}\n{seq}\n")

    def run(self):
        sequences = self.load_sequences()
        self.save_individual_fastas(sequences)
        self.save_all_fasta(sequences)
        print(f"{len(sequences)} sequences saved in {self.output_dir}")