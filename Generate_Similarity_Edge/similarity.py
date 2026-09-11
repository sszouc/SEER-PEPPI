import pickle
from typing import Dict

import numpy as np


def cosine_similarity_dict(embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    sequences = list(embeddings.keys())
    matrix = np.stack([embeddings[seq] for seq in sequences], axis=0).astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / (norms + 1e-12)
    sim_matrix = normed @ normed.T

    sim_dict = {}
    for i, s1 in enumerate(sequences):
        sim_dict[s1] = {}
        for j, s2 in enumerate(sequences):
            sim_dict[s1][s2] = float(sim_matrix[i, j])

    return sim_dict


def save_similarity(similarity: Dict[str, Dict[str, float]], output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(similarity, f)

