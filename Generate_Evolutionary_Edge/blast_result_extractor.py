import os
import re


class BlastResultExtractor:
    """
    从 psiblast 输出结果中提取“最后一轮”的比对结果，并将其写入提取后的文件中。
    """

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_last_round(self, file_path):
        """
        提取最后一轮“Results from round X”的内容。
        返回 (轮次编号, 内容)
        """
        try:
            lines = open(file_path, encoding="utf-8").readlines()
        except UnicodeDecodeError:
            lines = open(file_path, encoding="latin1").readlines()

        last_round_start = None
        last_round_id = None

        for i, line in enumerate(lines):
            match = re.search(r"Results from round (\d+)", line)
            if match:
                last_round_start = i
                last_round_id = int(match.group(1))

        if last_round_start is None:
            return None, None

        return last_round_id, lines[last_round_start:]

    def run(self):
        """
        遍历所有 *_results.txt 文件，提取最后一轮的内容。
        """
        for fname in os.listdir(self.input_folder):
            if not fname.endswith("_results.txt"):
                continue

            in_path = os.path.join(self.input_folder, fname)
            round_id, content = self.extract_last_round(in_path)

            if content:
                out_name = f"extracted_{fname.replace('_results.txt', '')}_round_{round_id}.txt"
                out_path = os.path.join(self.output_folder, out_name)
                with open(out_path, "w") as f:
                    f.writelines(content)

        print("[BlastResultExtractor] 提取完成。")