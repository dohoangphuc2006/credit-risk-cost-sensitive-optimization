# Dữ liệu, nguồn gốc và quy trình chuẩn bị

## 1. Nguồn chính thức

| Hạng mục | Giá trị |
| --- | --- |
| Tên | Statlog (German Credit Data) |
| Tác giả | Hans Hofmann |
| Năm công bố | 1994 |
| Nguồn | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Bản tải dùng trong code | [OpenML ID 31](https://www.openml.org/d/31), `credit-g` |
| Citation | Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. |
| DOI | [10.24432/C5NC77](https://doi.org/10.24432/C5NC77) |
| Giấy phép | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), theo trang UCI |

Trích dẫn dataset khi sử dụng/chia sẻ. Điểm cộng dữ liệu do giảng viên đánh
giá theo minh chứng; dữ liệu ở đây được dùng dưới dạng benchmark công bố.

## 2. Schema

Bản OpenML dùng 20 đặc trưng, 1.000 hồ sơ, không có giá trị thiếu theo tài
liệu nguồn. Nhãn gốc `good`/`bad` ánh xạ thành `target=0`/`target=1`.

| Nhóm | Các đặc trưng |
| --- | --- |
| Số | `duration`, `credit_amount`, `installment_commitment`, `residence_since`, `age`, `existing_credits`, `num_dependents` |
| Danh mục | `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment`, `personal_status`, `other_parties`, `property_magnitude`, `other_payment_plans`, `housing`, `job`, `own_telephone`, `foreign_worker` |
| Nhãn | `target`: 0 good (700), 1 bad (300) |
| Định danh kỹ thuật | `source_row_id`: vị trí hàng gốc, chỉ để truy vết; **không đưa vào mô hình** |

Nhóm cột thực tế được ghi tự động trong metadata sau khi tải. Không gán thứ
tự số học tùy ý cho các giá trị danh mục trước khi chia dữ liệu.

## 3. Artifact v2

Chạy từ thư mục gốc repo:

```powershell
.\.venv\Scripts\python.exe -m src.data_utils --force
```

| File được ghi | Nội dung |
| --- | --- |
| `credit_train.csv` | 800 mẫu, 20 đặc trưng chưa mã hóa + `target` + `source_row_id` |
| `credit_test.csv` | 200 mẫu theo schema tương tự, giữ riêng cho đánh giá cuối |
| `cv_folds.json` | Source row IDs validation cho năm folds train |
| `dataset_metadata.json` | Nguồn, schema, class counts, split IDs, config, environment và checksum |

`--force` cho phép thay artifact hiện có. Giữ split seed 42 để duy trì phép
chia gốc; không đổi seed nhằm tìm kết quả test tốt hơn.

Hai CSV dùng đặc trưng gốc. `load_training_data()` bỏ định danh kỹ thuật khỏi
features và chỉ đọc file train; `load_cv_splits()` trả chỉ số vị trí để sklearn
dùng lại đúng folds. SHA-256 phát hiện file thay đổi sau khi chuẩn bị;
config khác snapshot khiến loader dừng và yêu cầu tạo lại artifact.

## 4. Trạng thái dữ liệu tuần 1

Sinh viên đã chạy thành công quy trình chuẩn bị dữ liệu v2:

- Hai CSV train/test đã được tạo lại với 800/200 mẫu và đặc trưng chưa mã hóa.
  Train có 560 good và 240 bad; test có 140 good và 60 bad theo metadata.
- Metadata có `artifact_format_version=2`, ghi checksum và môi trường thực.
  Đã kiểm tra checksum các CSV, CV folds và cấu hình khớp metadata.
- `cv_folds.json` đã lưu năm folds cố định. `requirements-lock.txt` ở thư mục
  gốc lưu phiên bản thư viện sau khi cài đặt và chuẩn bị dữ liệu thành công.
- `german_credit_data.csv` là file cũ có 9 đặc trưng, không có nhãn good/bad,
  và chưa xác minh provenance. File được giữ để tránh mất dữ liệu cũ,
  không được script tự đọc hoặc dùng thay cho OpenML ID 31.

## 5. Tiền xử lý và holdout

Chia stratified train/test trước khi fit preprocessing. Chỉ dùng train để
chia năm folds. `src/preprocessing.py` tạo imputer và One-Hot encoder chưa
fit; đặt transformer trong cùng pipeline RF để mỗi fold fit riêng.

Test được giữ riêng theo protocol. Tuần 1 chỉ chuẩn bị dữ liệu và thống kê
split, không tính accuracy, chi phí hay chọn ngưỡng trên test. Chỉ đánh giá
test sau khi chốt phương pháp và cấu hình.

## 6. Chi phí và giới hạn

UCI công bố chi phí phân loại nhầm bad thành good bằng 5, good thành bad
bằng 1. Sau khi đổi nhãn, tương ứng `C_FN=5`, `C_FP=1`. Đây là đơn vị tương
đối; không có căn cứ coi là tiền thật, mất toàn bộ vốn hoặc hành vi gian lận.
Ràng buộc tỷ lệ phê duyệt 65% do nghiên cứu giả định.
