import os
import re
import math
from tqdm import tqdm


class EvolutionEdgeBuilder:

    def __init__(self, input_dir, seq_mapping, output_file, transform='none'):
        self.input_dir = input_dir
        self.seq_mapping = seq_mapping
        self.output_file = output_file
        self.transform = transform

    def _transform_score(self, e_val):
        if self.transform == 'none':
            return e_val
        elif self.transform == 'neg_log10':
            return -math.log10(e_val + 1e-180)
        elif self.transform == 'exp_neg':
            if e_val > 700:
                return 0.0
            return math.exp(-e_val)
        else:
            return e_val  # 回退

    def process_line(self, line):
        match = re.match(r"(sequence_\d+),\s*([\deE\.\-\+]+)", line.strip())
        if not match:
            return None

        seq_id, e_val_str = match.groups()
        try:
            e_val = float(e_val_str)
        except ValueError:
            return None
        # You should change this
        if e_val > 0.5:
            return None

        if seq_id not in self.seq_mapping:
            return None

        hit_seq = self.seq_mapping[seq_id]
        score = self._transform_score(e_val)


        return hit_seq, score

    def process_file(self, file_path, output_buffer):
        basename = os.path.basename(file_path)
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

        for fname in tqdm(os.listdir(self.input_dir), desc="construct evolutionary edge"):
            if fname.endswith(".txt"):
                self.process_file(os.path.join(self.input_dir, fname), buffer)

        with open(self.output_file, "w") as f:
            f.writelines(buffer)

        print(f"{len(buffer)} evolutionary edges were generated。")