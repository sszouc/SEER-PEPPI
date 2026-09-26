import os
import json
import re


class BlastResultParser:

    def __init__(self, extracted_dir, fasta_mapping, output_json):

        self.extracted_dir = extracted_dir
        self.fasta_mapping = fasta_mapping
        self.output_json = output_json

    def parse_file(self, file_path):
        sequences = {}
        query_name = None
        start_extracting = False
        in_significant_section = False

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Query="):
                    query_name = line.split("=", 1)[1].strip()
                    continue

                if re.search(r"Results from round \d+", line):
                    start_extracting = True
                    in_significant_section = False
                    continue

                if not start_extracting:
                    continue

                if "Sequences producing significant alignments:" in line:
                    in_significant_section = True
                    continue

                if (
                    "Sequences used in model and found again:" in line or
                    "Sequences not found previously or not previously below threshold:" in line
                ):
                    in_significant_section = True
                    continue

                if line.startswith(">") or line.startswith("Lambda") or line.startswith("Effective search space"):
                    in_significant_section = False
                    break

                if in_significant_section:
                    parts = line.split()
                    if len(parts) >= 2:
                        seq_id = parts[0]
                        if not seq_id.startswith("sequence_"):
                            continue
                        try:
                            e_val = float(parts[-1])
                            sequences[seq_id] = e_val
                        except ValueError:
                            continue

        return query_name, sequences

    def load_extracted(self):

        all_results = {}
        for fname in os.listdir(self.extracted_dir):
            if fname.startswith("extracted_") and fname.endswith(".txt"):
                file_path = os.path.join(self.extracted_dir, fname)
                q, seqs = self.parse_file(file_path)
                if q and seqs:
                    all_results[q] = seqs
        return all_results

    def replace_ids(self, data):
        replaced = {}
        for q, hits in data.items():
            q_seq = self.fasta_mapping.get(q, q)
            new_hits = {self.fasta_mapping.get(h, h): v for h, v in hits.items()}
            replaced[q_seq] = new_hits
        return replaced

    def run(self):
        raw = self.load_extracted()
        replaced = self.replace_ids(raw)

        with open(self.output_json, "w") as f:
            json.dump(replaced, f, indent=4)