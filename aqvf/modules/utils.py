import fitz
import pandas as pd

def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def read_questions(csv_path):
    df = pd.read_csv(csv_path)
    return df


def save_json(data, path="qa_report.json"):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)