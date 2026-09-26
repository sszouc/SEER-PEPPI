import pickle
from pathlib import Path

class SampleMerger:

    def __init__(self, old_positive_file, old_negative_file, pkl_file, output_dir):

        self.old_positive_file = Path(old_positive_file)
        self.old_negative_file = Path(old_negative_file)
        self.pkl_file = Path(pkl_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_positive_pairs = set()
        self.all_negative_pairs = set()

        self.new_positive_file = self.output_dir / "new_positive_pairs.txt"
        self.new_negative_file = self.output_dir / "new_negative_pairs.txt"

    # 读取文本文件
    def _read_txt_file(self, file_path, target_set):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    peptide, receptor = line.split(maxsplit=1)
                    target_set.add((peptide, receptor))

    def _read_pkl_file(self):
        with open(self.pkl_file, 'rb') as f:
            data_list = pickle.load(f)

        for entry in data_list:
            receptor, _, peptide, _, label = entry
            label = int(label)
            pair = (peptide, receptor)
            if label == 1:
                self.all_positive_pairs.add(pair)
            elif label == 0:
                self.all_negative_pairs.add(pair)
            else:
                print(f"unknown label: {label}")

    def _write_new_file(self, target_set, file_path):
        with open(file_path, 'w') as f:
            for peptide, receptor in sorted(target_set):
                f.write(f"{peptide} {receptor}\n")

    def merge_and_save(self):
        self._read_txt_file(self.old_positive_file, self.all_positive_pairs)
        self._read_txt_file(self.old_negative_file, self.all_negative_pairs)

        self._read_pkl_file()

        self._write_new_file(self.all_positive_pairs, self.new_positive_file)
        self._write_new_file(self.all_negative_pairs, self.new_negative_file)
