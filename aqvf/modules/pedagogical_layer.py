class PedagogicalLayer:

    def map_bloom_consistency(self, predicted_bloom, labeled_bloom):
        return predicted_bloom == labeled_bloom

    def map_to_clo(self, question, clo_text):
        # đơn giản: keyword match
        score = sum([1 for w in clo_text.split() if w.lower() in question.lower()])
        return score