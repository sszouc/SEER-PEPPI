import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingAnalyzer:
    def __init__(self, embedding_pkl_path: str):
        with open(embedding_pkl_path, "rb") as f:
            self.embeddings = pickle.load(f)
        self.sequences = list(self.embeddings.keys())

    def compute_mean_embeddings(self) -> dict:
        mean_emb = {}
        for seq, emb in self.embeddings.items():
            mean_emb[seq] = np.mean(emb, axis=0)
        self.mean_embeddings = mean_emb
        return mean_emb

    def compute_cosine_similarity(self) -> dict:
        if not hasattr(self, "mean_embeddings"):
            self.compute_mean_embeddings()

        emb_list = list(self.mean_embeddings.values())
        sim_matrix = cosine_similarity(emb_list)

        sim_dict = {}
        for i, s1 in enumerate(self.sequences):
            sim_dict[s1] = {}
            for j, s2 in enumerate(self.sequences):
                sim_dict[s1][s2] = sim_matrix[i, j]

        self.similarity = sim_dict
        return sim_dict

    def save_similarity(self, output_path: str):
        if not hasattr(self, "similarity"):
            self.compute_cosine_similarity()

        with open(output_path, "wb") as f:
            pickle.dump(self.similarity, f)
