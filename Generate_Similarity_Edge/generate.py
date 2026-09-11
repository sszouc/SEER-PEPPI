import argparse

from Example import generate_protT5_embeddings, compute_similarity_from_embeddings

parser = argparse.ArgumentParser(description="Generate embeddings and compute similarity")
parser.add_argument(
    "--type",
    choices=["protein", "peptide"],
    default="protein",
    help="Which sequence type to process (default: protein).",
)

args = parser.parse_args()

# Configure file paths
INPUT_PKL = f"input_sequences_{args.type}.pkl"
PROT5_EMBEDDING_PKL = f"prot5_embeddings_{args.type}.pkl"
PROT5_SIMILARITY_PKL = f"prot5_similarity_{args.type}.pkl"

PROT5_MODEL_PATH = "prot_t5_xl_uniref50"
PROT5_TOKENIZER_PATH = PROT5_MODEL_PATH

# Step 1: Generate embeddings
print("Generating ProtT5 embeddings...")
generate_protT5_embeddings(
    input_pkl=INPUT_PKL,
    output_pkl=PROT5_EMBEDDING_PKL,
    model_path=PROT5_MODEL_PATH,
    tokenizer_path=PROT5_TOKENIZER_PATH
)

# Step 2: Compute pairwise cosine similarity
print("Computing Prot5 cosine similarity...")
compute_similarity_from_embeddings(
    embedding_pkl=PROT5_EMBEDDING_PKL,
    output_similarity_pkl=PROT5_SIMILARITY_PKL
)