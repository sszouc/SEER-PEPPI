import argparse
import os
import sys
import json

# 导入自定义处理模块
from sequence_preprocessor import SequencePreprocessor
from psi_blast_runner import PsiBlastRunner
from blast_result_extractor import BlastResultExtractor
from blast_result_parser import BlastResultParser
from evolution_edge_builder import EvolutionEdgeBuilder

parser = argparse.ArgumentParser(description="Generate embeddings and compute similarity")
parser.add_argument(
    "--type",
    choices=["protein", "peptide"],
    default="protein",
    help="Which sequence type to process (default: protein).",
)

args = parser.parse_args()


def main():

    INPUT_CSV = f"{args.type}_sequences.csv"
    WORK_DIR = f"workdir_{args.type}"

    FASTA_DIR = os.path.join(WORK_DIR, "fasta")
    RESULT_DIR = os.path.join(WORK_DIR, "psiblast_results")
    PSSM_DIR = os.path.join(WORK_DIR, "pssm")
    EXTRACTED_DIR = os.path.join(WORK_DIR, "extracted_rounds")
    EVALUE_DIR = os.path.join(WORK_DIR, "evalue_files")

    OUTPUT_JSON = os.path.join(WORK_DIR, "blast_hits.json")
    EDGE_OUTPUT = os.path.join(WORK_DIR, "evolution_edges.csv")

    BLAST_DB_PATH = f"{args.type}/database/{args.type}"

    ERROR_LOG = os.path.join(WORK_DIR, "psiblast_errors.log")

    os.makedirs(WORK_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        sys.exit(1)

    preprocessor = SequencePreprocessor(csv_path=INPUT_CSV, output_dir=FASTA_DIR)
    preprocessor.run()

    # 复用去重后的序列列表构建 ID 到序列的映射
    unique_sequences = preprocessor.load_sequences()
    fasta_mapping = {f"sequence_{i + 1}": seq for i, seq in enumerate(unique_sequences)}

    psiblast_runner = PsiBlastRunner(
        fasta_folder=FASTA_DIR,
        result_folder=RESULT_DIR,
        pssm_folder=PSSM_DIR,
        db_path=BLAST_DB_PATH,
        error_log=ERROR_LOG
    )
    psiblast_runner.run()

    extractor = BlastResultExtractor(
        input_folder=RESULT_DIR,
        output_folder=EXTRACTED_DIR
    )
    extractor.run()

    parser = BlastResultParser(
        extracted_dir=EXTRACTED_DIR,
        fasta_mapping=fasta_mapping,
        output_json=OUTPUT_JSON
    )
    parser.run()

    os.makedirs(EVALUE_DIR, exist_ok=True)

    with open(OUTPUT_JSON, "r") as f:
        all_hits = json.load(f)

    seq_to_id = {seq: seq_id for seq_id, seq in fasta_mapping.items()}

    for query_seq, hits in all_hits.items():
        query_id = seq_to_id.get(query_seq)
        if query_id is None:
            continue

        out_path = os.path.join(EVALUE_DIR, f"{query_id}.txt")
        with open(out_path, "w") as f_out:
            for hit_seq, e_val in hits.items():
                hit_id = seq_to_id.get(hit_seq, hit_seq)
                f_out.write(f"{hit_id}, {e_val}\n")

    edge_builder = EvolutionEdgeBuilder(
        input_dir=EVALUE_DIR,
        seq_mapping=fasta_mapping,
        output_file=EDGE_OUTPUT,
        transform='neg_log10'
    )
    edge_builder.run()



if __name__ == "__main__":
    main()