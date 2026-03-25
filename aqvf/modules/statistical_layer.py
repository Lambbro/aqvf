import numpy as np

class StatisticalLayer:

    def simulate_difficulty(self, semantic_relevance, bloom_level=None):

        # logistic transform
        k = 6
        semantic_component = 1 / (1 + np.exp(-k * (0.5 - semantic_relevance)))

        bloom_weight = {
            "BT1": 0.2,
            "BT2": 0.3,
            "BT3": 0.5,
            "BT4": 0.7,
            "BT5": 0.85,
            "BT6": 0.95
        }

        if bloom_level:
            bloom_component = bloom_weight.get(bloom_level, 0.5)
            difficulty = 0.5 * semantic_component + 0.5 * bloom_component
        else:
            difficulty = semantic_component

        return float(np.clip(difficulty, 0, 1))

    def simulate_discrimination(self, difficulty):

        # Parabolic curve (max at difficulty=0.5)
        discrimination = 4 * difficulty * (1 - difficulty)

        return float(discrimination)