import os
import subprocess
from tqdm import tqdm


class PsiBlastRunner:
    """
    负责执行 psiblast，生成 result.txt 和 pssm 文件，并记录失败的任务。
    """

    def __init__(self, fasta_folder, result_folder, pssm_folder, db_path, error_log):
        self.fasta_folder = fasta_folder
        self.result_folder = result_folder
        self.pssm_folder = pssm_folder
        self.db_path = db_path
        self.error_log = error_log

        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(self.pssm_folder, exist_ok=True)

    def run_psiblast(self, fasta_file):
        """
        对单个 FASTA 文件运行 psiblast。
        """
        query_file = os.path.join(self.fasta_folder, fasta_file)
        basename = os.path.splitext(fasta_file)[0]
        result_path = os.path.join(self.result_folder, f"{basename}_results.txt")
        pssm_path = os.path.join(self.pssm_folder, f"{basename}_pssm.txt")

        command = [
            "psiblast",
            "-query", query_file,
            "-db", self.db_path,
            "-num_iterations", "10",
            "-out", result_path,
            "-out_ascii_pssm", pssm_path,
            "-comp_based_stats", "0",
            "-inclusion_ethresh", "0.005"
        ]
        subprocess.run(command, check=True)

    def run(self):
        """
        循环处理所有 fasta 文件并执行 psiblast。
        """
        fasta_files = [f for f in os.listdir(self.fasta_folder) if f.endswith(".fasta") and "all" not in f]
        #这个地方有问题

        with open(self.error_log, "w") as err_f:
            for fasta in tqdm(fasta_files, desc="PSI-BLAST", unit="个文件"):
                try:
                    self.run_psiblast(fasta)
                except subprocess.CalledProcessError as e:
                    err_f.write(f"{fasta}\n")
                    print(f"[错误] PSI-BLAST 执行失败: {fasta}")

        print("[PsiBlastRunner] 所有 PSI-BLAST 任务已完成。")