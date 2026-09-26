import os
import subprocess
from tqdm import tqdm


class PsiBlastRunner:

    def __init__(self, fasta_folder, result_folder, pssm_folder, db_path, error_log):
        self.fasta_folder = fasta_folder
        self.result_folder = result_folder
        self.pssm_folder = pssm_folder
        self.db_path = db_path
        self.error_log = error_log

        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(self.pssm_folder, exist_ok=True)

    def run_psiblast(self, fasta_file):
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
        fasta_files = [f for f in os.listdir(self.fasta_folder) if f.endswith(".fasta") and "all" not in f]

        with open(self.error_log, "w") as err_f:
            for fasta in tqdm(fasta_files, desc="PSI-BLAST", unit="files"):
                try:
                    self.run_psiblast(fasta)
                except subprocess.CalledProcessError as e:
                    err_f.write(f"{fasta}\n")

        print("PSI-BLAST run finished.")