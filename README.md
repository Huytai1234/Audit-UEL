# Audit.py

Dự án môn Phân tích dữ liệu trong kế toán – kiểm toán, lớp K23405E, nhóm 5.
Chương trình kiểm tra sổ kế toán năm 2024, lập BCTC rút gọn, phân tích tài chính
và hỗ trợ đánh giá chấp nhận khách hàng kiểm toán.

Báo cáo: [Word](<Midterm report - group 5 .docx>) · [PDF](<Midterm report - group 5.pdf>).

## Chạy chương trình

Dùng Python. Mở Terminal tại thư mục `Audit-UEL`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 main.py
```

Windows: kích hoạt bằng `.venv\Scripts\activate`.
Sau khi chạy, mở `outputs/dashboard.html` bằng trình duyệt; không cần Internet.
Chạy lại main để tạo lại các file kết quả từ dữ liệu và cấu hình hiện tại.

## Dữ liệu và cấu hình

Hai file nguồn trong `data/raw/`:

- `SO CHI TIET 01.01-31.12.2024.xlsx` — dòng hạch toán và các dòng kiểm soát.
- `BANG CAN DOI PS TK 01.01-31.12.2024.xlsx` — số dư và phát sinh đối chiếu.

Cấu hình hiện dùng **dạng A**: một tài khoản và hai cột phát sinh Nợ/Có mỗi dòng.
Loader hỗ trợ CSV/Excel và dạng B: tài khoản Nợ, tài khoản Có, số tiền trên một dòng.

| File trong `config/` | Mục đích |
|---|---|
| `column_map.json` | Định dạng, sheet, tiêu đề và ánh xạ cột. |
| `account_map.json` | Ánh xạ BCTC, kết chuyển và giả định kỳ hạn. |
| `audit_config.json` | Kỳ kế toán, dung sai, trọng yếu, JE, CFO và nguồn ngưỡng. |
| `benchmarks.csv` | Ngưỡng và chiều đánh giá 19 tỷ số. |
| `risk_weights.json` | Điểm, trọng số, ngưỡng quyết định và quy tắc chặn. |
| `calendar_2024.json` | Lịch tham chiếu kiểm tra ngày hạch toán. |

Đổi bộ dữ liệu: cập nhật tên file trong main và cấu hình tương ứng. Căn cứ ngưỡng
ở `financial_analysis.benchmark_references` trong `audit_config.json`, đồng thời
xuất vào sheet `Benchmarks`; ngưỡng minh họa không mặc định là trung bình ngành.

## Kết quả

| Module | File trong `outputs/` |
|---|---|
| 1 — Toàn vẹn dữ liệu và đối chiếu nguồn | `01_data_integrity.xlsx` |
| 2 — Mười thủ tục bút toán, gồm Benford | `02_je_findings.xlsx` |
| 3 — Sổ cái, cân đối, BCTC và số liệu tháng | `03_ledger_statements.xlsx` |
| 4 — Tỷ số, CFO ước tính và biến động tháng | `05_financial_analysis.xlsx` |
| 5 — Phương án và ba mức trọng yếu | `04_materiality.xlsx` |
| 6 — Chín biểu đồ và dashboard tương tác | `charts/`, `dashboard.html` |
| 7 — Điểm, quy tắc chặn, độ nhạy và hành động | `06_acceptance_sensitivity.xlsx` |

`audit_log.jsonl` ghi nối tiếp tên thủ tục, thời gian, số lượng và trạng thái theo
`run_id`. Đọc `input_counts`/`output_counts` khi có nhiều bảng; sheet `Run_Log`
trong báo cáo JE chỉ chứa các sự kiện JE. Khi có lỗi hoặc bước bị bỏ qua,
kiểm tra nhật ký; file cũ không chứng minh lượt mới đã hoàn tất.

## Mã nguồn

`main.py` điều phối. Trong `auditpy/`: `models.py` định nghĩa đối tượng;
`loaders.py` đọc và chuẩn hóa; `validators.py` kiểm tra; `ledger.py` và
`statements.py` lập sổ/BCTC; `je_tests.py` và `benford.py` kiểm tra bút toán;
`materiality.py`, `ratios.py`, `acceptance.py` tính trọng yếu, tài chính và điểm;
`charts.py` vẽ; `utils.py` chứa hàm chung, xuất Excel và ghi log.
`tests/` kiểm chứng bằng mẫu nhỏ và tình huống lỗi/thiếu dữ liệu.

OOP gồm đóng gói, kế thừa/ABC/đa hình, dataclass frozen, Enum, dunder,
classmethod/staticmethod, composition, ngoại lệ và Factory/Strategy/Registry.
FP gồm hàm thuần, lambda, map/filter, reduce, closure, decorator, partial,
generator, itertools, pipeline, bất biến và comprehension.


- Giữ nguyên raw và tham chiếu file/sheet/dòng; không tự điền tiền thiếu, sửa
  tiền để ép cân hay cộng trùng tài khoản cha/con. Sổ giữ kết chuyển; KQKD loại
  phần kết chuyển đã nhận diện.
- Tiền dùng Decimal, xuất Excel dạng text để giữ chính xác. Đơn vị nguồn giả
  định là đồng VND; hình tiền dùng tỷ đồng. Thiếu đầu vào hoặc không thể chia
  an toàn trả kết quả kèm lý do; `None` khác số 0.
- Kỳ hạn, CFO và độ phù hợp Altman còn cần xác minh. Phát sinh theo đối tượng
  không thay số dư khi thiếu đầu kỳ; không tạo giả giờ/người lập hoặc số liệu
  năm trước để tính Beneish.
- Cờ JE, Benford, ngoại lệ thống kê và điểm rủi ro là tín hiệu xem xét. Đọc cùng
  giả định, nguồn và hành động còn mở; chúng không xác nhận gian lận, tổn thất
  hay việc phê duyệt nhận khách hàng. Kiểm thử không thay bằng chứng kiểm toán.
