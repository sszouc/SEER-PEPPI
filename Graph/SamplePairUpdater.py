import pickle
from pathlib import Path

class SampleMerger:
    """
    负责：
    makeblastdb.bat. 从文本文件中读取旧的阳性（正样本）和阴性（负样本）肽-受体对。
    2. 从pickle（.pkl）文件中读取额外的肽-受体数据。
    3. 合并所有数据并去除重复项。
    4. 写入新的阳性和阴性样本文件。
    """

    def __init__(self, old_positive_file, old_negative_file, pkl_file, output_dir):
        """
        初始化文件路径和存储集合。
        :param old_positive_file: 原始阳性样本对文件路径
        :param old_negative_file: 原始阴性样本对文件路径
        :param pkl_file: 包含额外数据的pickle文件路径
        :param output_dir: 保存新阳性和阴性文件的目录
        """
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
        """从文本文件中读取肽-受体对并添加到集合中。"""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    peptide, receptor = line.split(maxsplit=1)
                    target_set.add((peptide, receptor))

    # 读取pickle文件
    def _read_pkl_file(self):
        """从pickle文件中读取条目并添加到阳性/阴性集合中。"""
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
                print(f"警告：条目 {entry} 中存在未知标签 {label}")

    # 写入新文件
    def _write_new_file(self, target_set, file_path):
        """将集合中的排序后的对写入文本文件。"""
        with open(file_path, 'w') as f:
            for peptide, receptor in sorted(target_set):
                f.write(f"{peptide} {receptor}\n")

    # 主流程
    def merge_and_save(self):
        """合并新旧样本，去除重复项，并保存到新文件。"""
        # 1. 读取原始文本文件
        self._read_txt_file(self.old_positive_file, self.all_positive_pairs)
        self._read_txt_file(self.old_negative_file, self.all_negative_pairs)

        # 2. 读取pkl文件并更新集合
        self._read_pkl_file()

        # 3. 写入新文件
        self._write_new_file(self.all_positive_pairs, self.new_positive_file)
        self._write_new_file(self.all_negative_pairs, self.new_negative_file)

        print(f"新阳性样本对已保存至 {self.new_positive_file}")
        print(f"新阴性样本对已保存至 {self.new_negative_file}")