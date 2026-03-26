import numpy as np

class StatisticalLayer:

    def simulate_difficulty(self, semantic_relevance, bloom_level=None):

        # logistic transform
        k = 6
        mu = 0.5
        diff_s = 1 / (1 + np.exp(k * (mu - semantic_relevance)))

        bloom_weight = {
            "BT1": 0.2,
            "BT2": 0.3,
            "BT3": 0.5,
            "BT4": 0.7,
            "BT5": 0.85,
            "BT6": 0.95
        }
        alpha = 0.5
        if bloom_level:
            bloom_component = bloom_weight.get(bloom_level, 0.5)
            difficulty = alpha * diff_s + (1 - alpha) * bloom_component
        else:
            difficulty = diff_s

        return float(np.clip(difficulty, 0, 1))

    def simulate_discrimination(self, difficulty):

        # Parabolic curve (max at difficulty=0.5)
        discrimination = 4 * difficulty * (1 - difficulty)

        return float(discrimination)
