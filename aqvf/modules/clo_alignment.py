import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class CLOAlignment:

    def __init__(self, clo_json_path, similarity_threshold=0.3):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.clo_list = self.load_clo(clo_json_path)
        self.similarity_threshold = similarity_threshold
        self.build_embeddings()

    # ==========================
    # Load CLO
    # ==========================
    def load_clo(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==========================
    # Build embeddings
    # ==========================
    def build_embeddings(self):
        descriptions = [clo["description"] for clo in self.clo_list]
        self.clo_embeddings = self.model.encode(
            descriptions,
            normalize_embeddings=True
        )

    # ==========================
    # Map question + CCI
    # ==========================
    def map_question(self, question_text):

        q_embedding = self.model.encode(
            [question_text],
            normalize_embeddings=True
        )

        sims = np.dot(q_embedding, self.clo_embeddings.T)[0]

        best_idx = int(np.argmax(sims))

        # ===== CCI Calculation =====
        matched_clos = [s for s in sims if s >= self.similarity_threshold]
        cci = len(matched_clos) / len(self.clo_list)

        return {
            "best_clo": self.clo_list[best_idx]["clo_id"],
            "similarity": float(sims[best_idx]),
            "all_scores": {
                self.clo_list[i]["clo_id"]: float(sims[i])
                for i in range(len(sims))
            },
            "cci": float(cci),
            "matched_clo_count": len(matched_clos),
            "total_clo": len(self.clo_list)
        }