# BẢN ĐĂNG KÝ ĐỀ TÀI & MÔ HÌNH HÓA BÀI TOÁN (TUẦN 1)

**Môn học:** Nhập môn Tối ưu hóa  
**Chuyên ngành:** Trí tuệ nhân tạo – Khoa Công nghệ thông tin – Trường Đại học Văn Lang  
**Họ và tên:** Đỗ Hoàng Phúc  

---

## PHẦN 1: MẪU ĐĂNG KÝ ĐỀ TÀI (01 TRANG)

| Hạng mục | Nội dung chi tiết |
| :--- | :--- |
| **Tên đề tài** | Tối ưu hóa Mô hình Học máy Hướng Chi phí (Cost-Sensitive Machine Learning Optimization) trong Đánh giá Rủi ro Tín dụng Ngân hàng: Cực tiểu hóa Tổn thất Tài chính qua Tối ưu Ngưỡng Phân loại và Siêu tham số Mô hình. |
| **Bối cảnh / vấn đề** | Trong quản trị rủi ro tín dụng, duyệt nhầm kẻ quỵt nợ (FN) gây tổn thất vốn gốc nặng gấp 5 lần so với từ chối nhầm khách tốt (FP). Mô hình AI truyền thống tối ưu theo Accuracy/F1 với ngưỡng cố định $\tau = 0.5$ bỏ qua hoàn toàn bất đối xứng chi phí này. Cần mô hình hóa bài toán tối ưu hóa để tìm đồng thời ngưỡng $\tau^*$ và siêu tham số mô hình tối thiểu hóa tổng tổn thất tài chính thực tế. |
| **Dataset / nguồn** | **Statlog (German Credit Data)** – 1.000 hồ sơ khách hàng, 20 biến tài chính/kinh tế (Tuổi, Số tiền vay, Lịch sử nợ, Tiết kiệm, Việc làm...). Nguồn: UCI ML Repository / OpenML (ID: 31). Loại dữ liệu: Public benchmark công bố chính thức (+0.25 điểm cộng). |
| **Decision variables $x$** | Vector 5 biến quyết định hỗn hợp: $x = (\tau, n_{\text{est}}, d_{\max}, s_{\text{split}}, w_{\text{class}})$. |
| **Objective $f(x)$** | Cực tiểu hóa chi phí tổn thất tài chính trung bình trên mỗi hồ sơ thông qua 5-Fold Stratified CV: $\min_{x \in \Omega} f(x) = \frac{1}{K}\sum_{k=1}^{K}\frac{C_{\text{FN}} \cdot \text{FN}_k(x) + C_{\text{FP}} \cdot \text{FP}_k(x)}{N_k}$, với $C_{\text{FN}}=5$, $C_{\text{FP}}=1$. |
| **Constraints / trade-off** | Ràng buộc nghiệp vụ: $g_1(x) = \text{RejectionRate}(x) - 0.35 \le 0$ (Ngân hàng phải duyệt ít nhất 65% hồ sơ để đảm bảo chỉ tiêu kinh doanh). Đánh đổi trực tiếp giữa giảm nợ xấu và giữ chân khách hàng tốt. |
| **Baseline** | `RandomForestClassifier` mặc định `scikit-learn` với ngưỡng xác suất truyền thống $\tau = 0.5$ và trọng số lớp đồng đều. |
| **Phương pháp tối ưu chính** | Tối ưu hóa Bayes / TPE (Tree-structured Parzen Estimator) qua thư viện `Optuna`, kết hợp quét lưới định hướng chi phí (Cost-directed Search). |
| **Metric** | Chi phí tổn thất trung bình (Cost/Sample), % Tiết kiệm chi phí so với Baseline, Recall (Bad Credit), Tỷ lệ phê duyệt hồ sơ. |
| **Budget / điều kiện dừng** | $B = 50$ trials / seed; lặp lại trên 5 seeds cố định: `{42, 100, 2024, 2025, 2026}`. |
| **Sản phẩm dự kiến Tuần 5** | Mã nguồn Python tái lập, log `CSV/JSON`, đồ thị Cost vs. Threshold Curve, bảng đối sánh tiết kiệm chi phí thực tế, GitHub Portfolio hoàn chỉnh, Release `v1.0` và slide demo 5–7 phút. |
| **GitHub repository** | `https://github.com/dohoangphuc2006/credit-risk-cost-sensitive-optimization` (Public). |

---

## PHẦN 2: CHI TIẾT MÔ HÌNH HÓA TOÁN HỌC (PROBLEM FORMULATION)

### 1. Không gian biến quyết định ($\Omega$)

Vector biến quyết định hỗn hợp (liên tục và nguyên) gồm 5 thành phần điều khiển đồng thời ngưỡng phân loại và cấu trúc mô hình:

$$x = (\tau,\ n_{\text{est}},\ d_{\max},\ s_{\text{split}},\ w_{\text{class}}) \in \Omega$$

$$\Omega = [0.10,\ 0.90] \times [20,\ 200]_\mathbb{Z} \times [2,\ 15]_\mathbb{Z} \times [2,\ 20]_\mathbb{Z} \times [1.0,\ 10.0]$$

- $\tau \in [0.10, 0.90]$: Ngưỡng xác suất phân loại – nếu $P(\text{Bad} \mid x) \ge \tau$ thì từ chối hồ sơ.
- $n_{\text{est}} \in [20, 200] \cap \mathbb{Z}$: Số lượng cây quyết định trong rừng (`n_estimators`).
- $d_{\max} \in [2, 15] \cap \mathbb{Z}$: Độ sâu tối đa của mỗi cây (`max_depth`).
- $s_{\text{split}} \in [2, 20] \cap \mathbb{Z}$: Số mẫu tối thiểu để tách nhánh (`min_samples_split`).
- $w_{\text{class}} \in [1.0, 10.0]$: Trọng số phạt lớp nợ xấu trong quá trình huấn luyện (`class_weight`).

### 2. Ma trận chi phí tài chính (Cost Matrix – Chuẩn Benchmark Quốc tế)

Theo quy chuẩn benchmark của German Credit Data (Hofmann, 1994), ma trận chi phí được định nghĩa:

| | Thực tế: Tốt ($y=0$) | Thực tế: Nợ xấu ($y=1$) |
| :--- | :---: | :---: |
| **Dự đoán: Duyệt** ($\hat{y}=0$) | $C_{\text{TN}} = 0$ (đúng) | $C_{\text{FN}} = 5$ (lọt nợ xấu – nặng) |
| **Dự đoán: Từ chối** ($\hat{y}=1$) | $C_{\text{FP}} = 1$ (từ chối nhầm người tốt) | $C_{\text{TP}} = 0$ (đúng) |

### 3. Hàm mục tiêu tài chính ($f: \Omega \to \mathbb{R}_+$)

Cho tập dữ liệu huấn luyện $\mathcal{D}_{\text{train}}$ được chia thành $K = 5$ folds phân tầng. Với mỗi cấu hình khả thi $x \in \Omega$:

**Bước 1:** Huấn luyện mô hình $\mathcal{M}(x)$ trên tập $\mathcal{D}_{\text{train}}^{(-k)}$ và lấy xác suất dự đoán trên tập kiểm định $\mathcal{D}_{\text{val}}^{(k)}$.

**Bước 2:** Áp dụng ngưỡng $\tau$ để phân loại:
$$\hat{y}_i = \begin{cases} 1\ (\text{Từ chối}) & \text{nếu } P(\text{Bad} \mid \mathbf{x}_i) \ge \tau \\ 0\ (\text{Duyệt}) & \text{nếu } P(\text{Bad} \mid \mathbf{x}_i) < \tau \end{cases}$$

**Bước 3:** Tính chi phí tổn thất trung bình trên từng fold:
$$f(x) = \frac{1}{K} \sum_{k=1}^{K} \left[ \frac{C_{\text{FN}} \cdot \text{FN}_k(x) + C_{\text{FP}} \cdot \text{FP}_k(x)}{N_k} \right]$$

### 4. Ràng buộc và Phạt vi phạm ($g_1(x) \le 0$)

$$g_1(x) = \frac{\text{FP}(x) + \text{TP}(x)}{N} - 0.35 \le 0 \iff \text{RejectionRate}(x) \le 35\%$$

Khi ràng buộc bị vi phạm ($g_1(x) > 0$), nghiệm $x$ bị coi là **không khả thi (infeasible)** và bị loại khỏi quá trình tìm kiếm nghiệm tối ưu của Optuna thông qua cơ chế phạt (Penalty Constraint Handling).

### 5. Phần tử lưu ý về Đánh đổi (Trade-off Insight)

Khi tăng $\tau$ (ngưỡng từ chối cao hơn):
- **Lợi ích:** Lọc thêm được nhiều nợ xấu hơn $\Rightarrow$ Giảm $\text{FN}$ $\Rightarrow$ Giảm thiệt hại vốn gốc.
- **Hậu quả:** Từ chối thêm nhiều khách hàng tốt $\Rightarrow$ Tăng $\text{FP}$ $\Rightarrow$ Mất tiền lãi vay và vi phạm ràng buộc nghiệp vụ.

Đây chính là bài toán đánh đổi (Trade-off) cốt lõi mà kỹ thuật tối ưu hóa phải giải quyết.
