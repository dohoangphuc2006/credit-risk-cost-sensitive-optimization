# Dataset Documentation & Provenance

## 1. Nguồn dữ liệu (Benchmark Provenance)

| Thuộc tính | Giá trị |
| :--- | :--- |
| **Tên bộ dữ liệu** | Statlog (German Credit Data) |
| **Tác giả** | Prof. Dr. Hans Hofmann, University of Hamburg (1994) |
| **Kho lưu trữ UCI** | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| **OpenML ID** | 31 – https://www.openml.org/d/31 |
| **Quyền sử dụng** | Công khai phục vụ nghiên cứu và giảng dạy (+0.25 điểm cộng) |
| **Truy xuất qua** | `sklearn.datasets.fetch_openml(name="credit-g", version=1)` |

## 2. Cấu trúc dữ liệu (Schema)

- **Số mẫu:** 1.000 hồ sơ khách hàng vay vốn tại Đức.
- **Số biến đặc trưng gốc:** 20 thuộc tính tài chính và kinh tế xã hội bao gồm: Lịch sử tín dụng, Số tiền vay, Mục đích vay, Số dư tiết kiệm, Tình trạng việc làm, Tuổi tác, Số người phụ thuộc...
- **Nhãn mục tiêu (`target`):**
  - `0`: Good Credit – Khách hàng tín dụng tốt (700 mẫu – 70%).
  - `1`: Bad Credit – Khách hàng có nguy cơ nợ xấu (300 mẫu – 30%).

## 3. Ma trận Chi phí Chuẩn (Cost Matrix – Hofmann, 1994)

Tỷ lệ chi phí $C_{\text{FN}} : C_{\text{FP}} = 5 : 1$ được định nghĩa chính thức trong hồ sơ gốc của tập dữ liệu, là chuẩn benchmark quốc tế được trích dẫn rộng rãi trong các nghiên cứu về Cost-Sensitive Learning.

## 4. Quy trình chia tách chống rò rỉ (Anti-Leakage Split)

Dữ liệu được tải và chia tách tự động bằng `src/data_utils.py` với hạt giống cố định `SEED = 42`:

- **80% Train (800 mẫu):** Dùng cho 5-Fold Stratified CV trong suốt quá trình tối ưu hóa (Tuần 2–4).
- **20% Test (200 mẫu):** Khóa kín hoàn toàn. Tuyệt đối không sử dụng trong quá trình tìm kiếm siêu tham số. Chỉ được mở để đánh giá nghiệm tối ưu cuối cùng ở Tuần 4–5.
