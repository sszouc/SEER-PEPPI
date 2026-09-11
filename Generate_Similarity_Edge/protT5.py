import re
import torch
import pickle
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel


class ProT5Embedder:
    def __init__(self, model_path, tokenizer_path, device_id=0):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_path).to(self.device)

        if self.device.type == "cpu":
            self.model = self.model.to(torch.float32)

    def preprocess_sequence(self, seq: str) -> str:
        seq = re.sub(r"[UZOB]", "X", seq)
        return " ".join(list(seq))

    def embed_sequence(self, sequence: str) -> np.ndarray:
        sequence = self.preprocess_sequence(sequence)
        ids = self.tokenizer([sequence], add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)

        seq_len = len(sequence.replace(" ", ""))
        emb = embedding_repr.last_hidden_state[0, :seq_len]

        return emb.cpu().numpy()  # (L, 1024)

    def embed_batch(self, sequences: list) -> dict:
        result = {}
        for seq in sequences:
            result[seq] = self.embed_sequence(seq)
        return result

    def save_embeddings(self, embeddings: dict, output_path: str):
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)

