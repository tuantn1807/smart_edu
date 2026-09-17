# Dữ liệu chạy thực tế

PAAF dùng **hai dataset đúng vai trò kiến trúc**:

1. **Eedi 2024** → Diagnostic Agent (câu hỏi + nhãn hiểu lầm)
2. **Junyi** → Knowledge Graph Agent (đồ thị khái niệm / tiên quyết)

Ánh xạ Eedi construct → node Junyi là **heuristic** (`src/data/concept_mapping.py`), không phải bảng chuyên gia. Construct không khớp thì KG Agent không duyệt tiên quyết giả.

## Eedi – Mining Misconceptions in Mathematics (2024)

- `raw/eedi/train.csv`: toàn bộ 1.869 câu hỏi của training split, giữ nguyên tiếng Anh và LaTeX.
- `raw/eedi/misconception_mapping.csv`: 2.587 nhãn hiểu lầm trong bảng tra.
- `sources.json`: URL nguồn, bản mirror, commit cố định và SHA-256 từng file.

Nguồn gốc: [cuộc thi Eedi 2024](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data).
Bản tải: [data/raw của lời giải Eedi_kaggle](https://github.com/wangqihanginthesky/Eedi_kaggle/tree/17a1bc3825c5354b7f70adbb5124198cca448e5f/data/raw).
Đây là mirror cộng đồng, không phải máy chủ Eedi. Điều kiện sử dụng theo nguồn cuộc thi; không gán giấy phép mã nguồn của dự án cho dữ liệu.

```bash
python3 -m src.data.download_datasets
python3 demo.py
python3 evaluate.py
python3 -m unittest discover -s tests
```

`evaluate.py` chấm protocol trên toàn bộ Eedi + Junyi (tra cứu nhãn, ánh xạ, duyệt graph, cấu trúc lộ trình ZPD) và ghi `eval/results.json`. Đây không phải bảng điểm LLM hay so sánh DKT/NCD.

Downloader dùng thư viện chuẩn Python, bỏ qua file đúng checksum, chỉ thay file sau khi kiểm tra SHA-256. Demo chạy offline sau khi tải. Thiếu CSV hoặc ID không tồn tại sẽ báo lỗi, không dùng câu hỏi giả thay thế.

## Giới hạn dữ liệu Eedi

- Đây là Eedi **2024**, không phải NeurIPS 2020. Không tải dữ liệu sinh tổng hợp từ thư mục khác của mirror.
- Mỗi câu hỏi có 4 đáp án; một số đáp án sai không có nhãn. Không tự bổ sung nhãn thiếu. `severity=unknown` vì nguồn không chú thích mức độ nghiêm trọng.
- Graph Eedi (`load_concept_graph`) chỉ có node construct, **không có cạnh tiên quyết**. Demo không dùng graph này cho KG Agent.
- Các học sinh, đáp án được chọn trong demo và điểm mastery vẫn là kịch bản/heuristic ứng dụng; CSV này không phải nhật ký tương tác học sinh. Tutor vẫn dùng template, chưa gọi LLM.
- Hai JSON cũ nằm trong `fixtures/synthetic_*.json`, chỉ lưu ví dụ giả lập để tham khảo; loader không đọc chúng.

## Junyi

KG Agent đọc Junyi theo thứ tự:

1. `raw/junyi/junyi_Exercise_table.csv` nếu có — cột `prerequisite` / `prerequisites` (bảng chuyên gia DataShop 1198).
2. Nếu không, `junyi/archive/Info_Content.csv` — đồ thị phân cấp `level2 → level3 → level4 → exercise` từ dataset Kaggle 2019 đã có sẵn. Đây là **hierarchy nội dung**, không phải cạnh tiên quyết do chuyên gia gán.

`junyi/archive/Log_Problem.csv` (~2,9 GB) **không được nạp**; không dùng nhật ký làm cạnh đồ thị.

Nếu bổ sung bảng exercise thật, đặt tại `data/raw/junyi/junyi_Exercise_table.csv`. Loader giữ quan hệ trong cột prerequisite (bỏ cạnh tự trỏ); dữ liệu nguồn có thể chứa chu trình.

Nguồn Info_Content: [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy). Bảng expert optional: [DataShop 1198](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198) / [EduData junyi](https://github.com/bigdata-ustc/EduData).
