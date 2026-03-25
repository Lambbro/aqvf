from clo_parser import extract_clean_clos

clo_list = extract_clean_clos(r"G:\My Drive\Hoạt động nghiên cứu\Đề tài\15032026_AI nâng cao hiệu quả giáo dục\AI bloom taxonomy\Modules\aqvf\data\OOP.pdf")

for clo in clo_list:
    print(clo["clo_id"])
    print(clo["description"][:200])
    print("-" * 50)