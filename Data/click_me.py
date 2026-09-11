import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple


def parse_test_fasta(fasta_path: Path) -> List[Dict[str, str]]:
    """Parse records like:
    >positive_1
    Peptide: AAA
    Protein: BBB
    """
    records: List[Dict[str, str]] = []
    current = {"header": "", "peptide": "", "protein": ""}
    active_field = None

    def flush_current() -> None:
        if not current["header"]:
            return
        if not current["peptide"] or not current["protein"]:
            raise ValueError(f"Record missing peptide/protein: {current['header']}")

        header_lower = current["header"].lower()
        if header_lower.startswith("positive"):
            label = "positive"
        elif header_lower.startswith("negative"):
            label = "negative"
        else:
            raise ValueError(f"Unknown label in header: {current['header']}")

        records.append(
            {
                "header": current["header"],
                "peptide": current["peptide"],
                "protein": current["protein"],
                "label": label,
            }
        )

    with fasta_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                flush_current()
                current = {"header": line[1:].strip(), "peptide": "", "protein": ""}
                active_field = None
                continue

            if line.startswith("Peptide:"):
                current["peptide"] = line.split(":", 1)[1].strip()
                active_field = "peptide"
                continue

            if line.startswith("Protein:"):
                current["protein"] = line.split(":", 1)[1].strip()
                active_field = "protein"
                continue

            # Support wrapped FASTA-style continuation lines after Peptide:/Protein:
            if active_field == "peptide":
                current["peptide"] += line
            elif active_field == "protein":
                current["protein"] += line
            else:
                raise ValueError(f"Unexpected line without active field: {line}")

    flush_current()
    return records


def write_single_column_csv(path: Path, sequences: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for seq in sequences:
            writer.writerow([seq])


def write_pairs_txt(path: Path, pairs: List[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for peptide, protein in pairs:
            f.write(f"{peptide} {protein}\n")


def write_sequence_pkl(path: Path, sequences: List[str]) -> None:
    # 去重并保持原始顺序
    seen = set()
    unique_sequences = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            unique_sequences.append(seq)

    # Keep item[0] == sequence for compatibility with Generate_Prot5_feature/Example.py
    payload = [(seq,) for seq in unique_sequences]
    with path.open("wb") as f:
        pickle.dump(payload, f)


def build_outputs(fasta_path: Path, output_dir: Path) -> None:
    records = parse_test_fasta(fasta_path)

    # Keep raw record order and duplicates for easier manual comparison.
    peptides = [item["peptide"] for item in records]
    proteins = [item["protein"] for item in records]

    positive_pairs = [
        (item["peptide"], item["protein"]) for item in records if item["label"] == "positive"
    ]
    negative_pairs = [
        (item["peptide"], item["protein"]) for item in records if item["label"] == "negative"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_sequence_pkl(output_dir / "input_sequences_protein.pkl", proteins)
    write_sequence_pkl(output_dir / "input_sequences_peptide.pkl", peptides)

    write_single_column_csv(output_dir / "protein_sequences.csv", proteins)
    write_single_column_csv(output_dir / "peptide_sequences.csv", peptides)

    write_pairs_txt(output_dir / "positive_pairs.txt", positive_pairs)
    write_pairs_txt(output_dir / "negative_pairs.txt", negative_pairs)

    print(f"Records parsed: {len(records)}")
    print(f"Peptide rows: {len(peptides)}")
    print(f"Protein rows: {len(proteins)}")
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Negative pairs: {len(negative_pairs)}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate files")
    parser.add_argument(
        "--fasta",
        default="Train_8622.fasta",
        help="Input FASTA path with positive/negative peptide-protein records",
    )
    parser.add_argument(
        "--output-dir",
        default="Train_8622",
        help="Directory to save generated files",
    )
    args = parser.parse_args()

    build_outputs(Path(args.fasta), Path(args.output_dir))


if __name__ == "__main__":
    main()

