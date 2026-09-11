import pickle

from Generate_Similarity_Edge.mean import EmbeddingAnalyzer
from protT5 import ProT5Embedder

def generate_protT5_embeddings(input_pkl, output_pkl, model_path, tokenizer_path):
    embedder = ProT5Embedder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device_id=0
    )

    with open(input_pkl, "rb") as f:
        sequences = [item[0] for item in pickle.load(f)]

    embeddings = embedder.embed_batch(sequences)
    embedder.save_embeddings(embeddings, output_pkl)



def compute_similarity_from_embeddings(embedding_pkl, output_similarity_pkl):
    analyzer = EmbeddingAnalyzer(embedding_pkl)
    analyzer.compute_mean_embeddings()
    analyzer.compute_cosine_similarity()
    analyzer.save_similarity(output_similarity_pkl)