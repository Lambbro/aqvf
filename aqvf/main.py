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

BASE_PATH = r"G:\My Drive\Hoạt động nghiên cứu\Đề tài\31032026_Tạp chí khoa học giáo dục VN\AI bloom taxonomy\Modules\aqvf\data"

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
all_questions = test_df["Question"].tolist()
computing_coverage = semantic_layer.compute_coverage(all_questions)

# BƯỚC 2: Lấy danh sách điểm CCI và giá trị bao phủ toàn cục
list_all_cci = computing_coverage["question_cci"]  # Đây là một list các số thực [0.85, 0.72, ...]
global_corpus_coverage = computing_coverage["corpus_coverage"]

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):

    question_id = int(idx) + 1
    question_text = row["Question"]
    labeled_bloom = row["Label"]

    # 1️⃣ Semantic Analysis
    semantic_scores = semantic_layer.analyze_question(question_text)
    
    # BƯỚC 3: TRÍCH XUẤT GIÁ TRỊ TCI RIÊNG CHO CÂU HỎI HIỆN TẠI
    # Thay vì lưu cả list, ta chỉ lấy phần tử tại vị trí idx
    current_tci = list_all_cci[idx]

    # 2️⃣ Bloom Prediction
    predicted_bloom, bloom_conf = bloom_model.predict(
        question_text,
        return_confidence=True
    )
    bloom_consistency = predicted_bloom == labeled_bloom

    # 3️⃣ CLO Alignment
    clo_result = clo_mapper.map_question(question_text)

    # 4️⃣ Statistical Estimation
    difficulty = stat_layer.simulate_difficulty(
        semantic_scores["sr_max"],
        predicted_bloom
    )
    discrimination = stat_layer.simulate_discrimination(difficulty)

    qa_results.append({
        "question_id": question_id,
        "question": question_text,

        # Semantic
        "sr_max": semantic_scores["sr_max"],
        "sr_avg": semantic_scores["sr_avg"],
        
        # CHỈ LƯU 1 GIÁ TRỊ SỐ THỰC TẠI ĐÂY
        "tci": float(current_tci), 
        
        "n_cov": global_corpus_coverage,

        # Bloom & CLO ... (giữ nguyên các phần dưới)
        "bloom_label": labeled_bloom,
        "predicted_label": predicted_bloom,
        "bloom_confidence": bloom_conf,
        "bloom_consistent": bloom_consistency,
        "best_clo": clo_result["best_clo"],
        "sr_clo": clo_result["similarity"],
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