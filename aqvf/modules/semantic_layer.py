import re
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticLayer:

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    # ==========================
    # 1️⃣ Clean text (loại hành chính)
    # ==========================
    def clean_text(self, text):
        text = re.sub(r'Trên lớp.*', '', text)
        text = re.sub(r'Tổng số giờ.*', '', text)
        text = re.sub(r'Bảng\s*\d+.*', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ==========================
    # 2️⃣ Chunk text
    # ==========================
    def chunk_text(self, text, chunk_size=300):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk.strip()) > 50:  # bỏ đoạn quá ngắn
                chunks.append(chunk)
        return chunks

    # ==========================
    # 3️⃣ Build KB
    # ==========================
    def build_knowledge_base(self, pdf_texts):

        all_chunks = []

        for doc in pdf_texts:
            cleaned = self.clean_text(doc)
            chunks = self.chunk_text(cleaned)
            all_chunks.extend(chunks)

        self.documents = all_chunks

        self.doc_embeddings = self.model.encode(
            all_chunks,
            normalize_embeddings=True
        )

    # ==========================
    # 4️⃣ Analyze question
    # ==========================
    def analyze_question(self, question):

        q_embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        sims = np.dot(q_embedding, self.doc_embeddings.T)[0]

        # chuẩn hóa [-1,1] → [0,1]
        sims = (sims + 1) / 2
        sims = np.clip(sims, 0, 1)

        return {
            "sr_max": float(np.max(sims)),
            "sr_avg": float(np.mean(sims))
        }
    def compute_coverage(self, questions, threshold=0.4):
        q_embeddings = self.model.encode(
            questions,
            normalize_embeddings=True
        )

        # Ma trận tương đồng: (số câu hỏi) x (số lượng chunk)
        sims = np.dot(q_embeddings, self.doc_embeddings.T)

        # Chuẩn hóa về đoạn [0, 1]
        sims = (sims + 1) / 2
        sims = np.clip(sims, 0, 1)

        # ===== CÁCH SỬA: Question-level CCI =====
        # Thay vì đếm chunk, ta lấy giá trị tương đồng cao nhất của câu hỏi đó 
        # đối với kiến thức trong Database.
        # Điều này giúp phân biệt câu hỏi nào "sát" tài liệu hơn.
        
        question_cci = []
        for q_sim in sims:
            # Lấy top 3 điểm cao nhất để tính trung bình cộng cci cho câu hỏi đó
            # Hoặc đơn giản là lấy max nếu bạn muốn biết độ khớp cao nhất
            top_scores = np.sort(q_sim)[-3:] 
            cci_val = np.mean(top_scores) 
            question_cci.append(float(cci_val))

        # ===== Corpus-level Coverage (Độ phủ toàn bộ tài liệu) =====
        chunk_covered = np.max(sims, axis=0) >= threshold
        corpus_coverage = np.sum(chunk_covered) / len(self.documents)

        return {
            "question_cci": question_cci, # Sẽ trả về danh sách [0.85, 0.72, 0.61, ...]
            "corpus_coverage": float(corpus_coverage),
            "total_chunks": len(self.documents)
        }