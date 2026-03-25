import os
from tqdm import tqdm

from modules.utils import read_pdf, read_questions, save_json
from modules.semantic_layer import SemanticLayer
from modules.statistical_layer import StatisticalLayer
from modules.bloom_classifier import BloomClassifier
from modules.clo_alignment import CLOAlignment


# ==============================
# 1. PATH CONFIG
# ==============================

BASE_PATH = r"G:\My Drive\Hoạt động nghiên cứu\Đề tài\15032026_AI nâng cao hiệu quả giáo dục\AI bloom taxonomy\Modules\aqvf\data"

CLO_JSON = os.path.join(BASE_PATH, "clo.json")
LECTURE_PDF = os.path.join(BASE_PATH, "lecturer.pdf")
PRACTICE_PDF = os.path.join(BASE_PATH, "pratical.pdf")
TRAIN_CSV = os.path.join(BASE_PATH, "OOP.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")


# ==============================
# 2. LOAD DATA
# ==============================

lecture_text = read_pdf(LECTURE_PDF)
practice_text = read_pdf(PRACTICE_PDF)

knowledge_base = [lecture_text, practice_text]

train_df = read_questions(TRAIN_CSV)
test_df = read_questions(TEST_CSV)


# ==============================
# 3. INIT MODELS
# ==============================

# Bloom classifier (TF-IDF + SVM)
bloom_model = BloomClassifier()
bloom_model.train(TRAIN_CSV)

# Semantic layer
semantic_layer = SemanticLayer()
semantic_layer.build_knowledge_base(knowledge_base)

# CLO alignment (embedding-based)
clo_mapper = CLOAlignment(CLO_JSON)

# Statistical simulation
stat_layer = StatisticalLayer()


# ==============================
# 4. PIPELINE
# ==============================

qa_results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):

    question_id = int(idx) + 1
    question_text = row["question"]
    labeled_bloom = row["label"]

    # 1️⃣ Semantic Analysis
    semantic_scores = semantic_layer.analyze_question(question_text)

    # 2️⃣ Bloom Prediction
    predicted_bloom, bloom_conf = bloom_model.predict(
        question_text,
        return_confidence=True
    )

    bloom_consistency = predicted_bloom == labeled_bloom

    # 3️⃣ CLO Alignment (NEW VERSION)
    clo_result = clo_mapper.map_question(question_text)

    # 4️⃣ Statistical Estimation
    difficulty = stat_layer.simulate_difficulty(
        semantic_scores["semantic_relevance"],
        predicted_bloom
    )

    discrimination = stat_layer.simulate_discrimination(difficulty)

    qa_results.append({
        "question_id": question_id,
        "question": question_text,

        # Semantic
        "semantic_relevance": semantic_scores["semantic_relevance"],
        "avg_similarity": semantic_scores["avg_similarity"],

        # Bloom
        "bloom_label": labeled_bloom,
        "predicted_label": predicted_bloom,
        "bloom_confidence": bloom_conf,
        "bloom_consistent": bloom_consistency,

        # CLO
        "best_clo": clo_result["best_clo"],
        "clo_similarity": clo_result["similarity"],
        "all_clo_scores": clo_result["all_scores"],

        # Statistical
        "difficulty_estimated": difficulty,
        "discrimination_estimated": discrimination
    })


# ==============================
# 5. SAVE RESULT
# ==============================

output_path = os.path.join(BASE_PATH, "aqvf_results.json")

save_json({
    "total_questions": len(qa_results),
    "results": qa_results
}, output_path)

print("\n✅ AQVF analysis completed successfully.")
print("📁 Saved to:", output_path)