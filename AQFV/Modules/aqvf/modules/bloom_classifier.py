import re
import string
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np

class BloomClassifier:

    def __init__(self):
        self.model = None

    # =====================
    # 1. Preprocess
    # =====================
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # =====================
    # 2. Train model
    # =====================
    def train(self, csv_path):

        df = pd.read_csv(csv_path)

        # chuẩn hoá tên cột
        df.columns = df.columns.str.strip().str.lower()

        if "question" not in df.columns or "label" not in df.columns:
            raise Exception("CSV phải có cột: question, label")

        df["question"] = df["question"].apply(self.preprocess_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df["question"],
            df["label"],
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )),
            ("clf", LinearSVC())
        ])

        self.model.fit(X_train, y_train)

        acc = self.model.score(X_test, y_test)
        print(f"Bloom classifier accuracy: {acc:.4f}")

    # =====================
    # 3. Predict
    # =====================
    def predict(self, text, return_confidence=False):

        if self.model is None:
            raise Exception("Model chưa train")

        text = self.preprocess_text(text)

        prediction = self.model.predict([text])[0]

        if return_confidence:
            decision_scores = self.model.decision_function([text])
            confidence = float(max(decision_scores[0]))
            confidence = 1 / (1 + np.exp(-confidence))
            return prediction, confidence

        return prediction