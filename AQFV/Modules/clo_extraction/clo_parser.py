import re
import fitz


def extract_clean_clos(pdf_path):

    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    # Chuẩn hoá khoảng trắng
    full_text = re.sub(r'\s+', ' ', full_text)

    # Tìm đúng format: CLO1: ....
    pattern = r'(CLO\d+:\s.*?)(?=CLO\d+:|$)'
    matches = re.findall(pattern, full_text)

    clo_list = []

    for match in matches:

        # Lấy ID
        id_match = re.search(r'CLO\d+', match)
        clo_id = id_match.group()

        # Cắt bỏ phần "Thang đánh giá..."
        description = match.split("Thang đánh giá")[0]

        # Làm sạch
        description = description.strip()

        clo_list.append({
            "clo_id": clo_id,
            "description": description
        })

    return clo_list