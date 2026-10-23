# Cost-Sensitive Machine Learning Optimization for Credit Risk Assessment

**Bài tập lớn cá nhân – Môn Nhập môn Tối ưu hóa (05 Tuần)**

- **Sinh viên:** Đỗ Hoàng Phúc
- **Chuyên ngành:** Trí tuệ nhân tạo – Khoa Công nghệ thông tin – Trường Đại học Văn Lang

---

## 1. Problem – Đặt vấn đề & Bối cảnh Kinh doanh

Trong nghiệp vụ quản trị rủi ro tín dụng tại ngân hàng thương mại, **hai loại sai lầm của mô hình AI có chi phí bất đối xứng nghiêm trọng**:

| Loại sai lầm | Ý nghĩa kinh doanh | Chi phí chuẩn |
| :--- | :--- | :---: |
| **False Negative (FN)** | Duyệt vay cho kẻ quỵt nợ → Mất trắng vốn gốc | $C_{\text{FN}} = 5$ |
| **False Positive (FP)** | Từ chối nhầm khách tốt → Mất tiền lãi vay | $C_{\text{FP}} = 1$ |

Mô hình AI truyền thống chỉ tối ưu Accuracy/F1 với ngưỡng cố định $\tau = 0.5$ bỏ qua hoàn toàn sự bất đối xứng này, gây thiệt hại hàng tỷ đồng. **Giải pháp:** Tối ưu hóa đồng thời siêu tham số mô hình lẫn ngưỡng phân loại $\tau^*$ để cực tiểu hóa tổng tổn thất tài chính thực tế.

### Sơ đồ Pipeline Tối ưu hóa Chi phí

```
[German Credit Dataset – 1.000 hồ sơ khách hàng]
        │
        ├──► (20% Locked Test – 200 hồ sơ) ──────────────────────────────────┐
        │                                                                     ▼
        └──► (80% Train – 800 hồ sơ) ──► [Không gian quyết định x ∈ Ω] ──► [Đánh giá cuối]
                                                  │                           ▲
                                     (B = 50 trials / seed × 5 seeds)        │
                                                  ▼                           │
                                    [5-Fold Stratified CV]                    │
                                    Tính chi phí: f(x) = (5·FN + 1·FP) / N  │
                                    Kiểm tra ràng buộc: Rejection Rate ≤ 35% │
                                                  │                           │
                                                  ▼                           │
                              [Optuna – Tối ưu Bayes / TPE] ─────────────────┘
```

---

## 2. Optimization Formulation – Mô hình hóa Bài toán Tối ưu

### 2.1. Biến quyết định (Decision Variables $x \in \Omega$)

$$x = (\tau,\ n_{\text{est}},\ d_{\max},\ s_{\text{split}},\ w_{\text{class}}) \in \Omega$$

| Ký hiệu | Tham số mô hình | Miền giá trị |
| :--- | :--- | :--- |
| $x_1 = \tau$ | Ngưỡng xác suất phân loại (`threshold`) | $[0.10,\ 0.90]$ |
| $x_2 = n_{\text{est}}$ | Số lượng cây quyết định (`n_estimators`) | $\{20, 21, \ldots, 200\}$ |
| $x_3 = d_{\max}$ | Độ sâu tối đa của cây (`max_depth`) | $\{2, 3, \ldots, 15\}$ |
| $x_4 = s_{\text{split}}$ | Mẫu tối thiểu để tách nhánh (`min_samples_split`) | $\{2, 3, \ldots, 20\}$ |
| $x_5 = w_{\text{class}}$ | Trọng số phạt lớp nợ xấu (`class_weight`) | $[1.0,\ 10.0]$ |

### 2.2. Hàm mục tiêu (Objective Function $f: \Omega \to \mathbb{R}_+$)

Cực tiểu hóa chi phí tổn thất tài chính trung bình trên mỗi hồ sơ khách hàng qua kiểm định chéo phân tầng $K = 5$ folds:

$$\min_{x \in \Omega}\ f(x) = \frac{1}{K} \sum_{k=1}^{K} \left[ \frac{C_{\text{FN}} \cdot \text{FN}_k(x) + C_{\text{FP}} \cdot \text{FP}_k(x)}{N_k} \right]$$

trong đó $C_{\text{FN}} = 5$, $C_{\text{FP}} = 1$ theo chuẩn benchmark của Hofmann (1994).

### 2.3. Ràng buộc khả thi (Constraint $g(x) \le 0$)

$$g_1(x) = \text{RejectionRate}(x) - 0.35 \le 0$$

Ngân hàng bắt buộc duyệt tối thiểu $65\%$ hồ sơ để duy trì chỉ tiêu tăng trưởng tín dụng.

### 2.4. Baseline (Mốc đối sánh)

`RandomForestClassifier` mặc định của `scikit-learn` với:
- Ngưỡng phân loại truyền thống: $\tau = 0.5$
- Trọng số lớp đồng đều: `class_weight = None`

---

## 3. Dataset – Dữ liệu Thực nghiệm & Chống Rò rỉ

| Thuộc tính | Giá trị |
| :--- | :--- |
| **Tên** | Statlog (German Credit Data) |
| **Nguồn** | UCI ML Repository / OpenML (ID: 31) |
| **Tác giả** | Prof. Dr. Hans Hofmann, University of Hamburg (1994) |
| **Số mẫu** | 1.000 hồ sơ khách hàng |
| **Số biến** | 20 biến tài chính/kinh tế (Tuổi, Số tiền vay, Lịch sử nợ, Tiết kiệm...) |
| **Nhãn** | `0` = Good Credit (700 mẫu), `1` = Bad Credit (300 mẫu) |
| **Phân chia** | 80% Train (5-Fold CV) / 20% Locked Test (SEED = 42) |

---

## 4. Method – Phương pháp Tối ưu hóa

- **Phương pháp chính:** Tối ưu hóa Bayes / TPE qua thư viện `Optuna`
- **Ngân sách:** $B = 50$ trials × 5 seeds (`42, 100, 2024, 2025, 2026`)

---

## 5. Kế hoạch 05 Tuần

| Tuần | Nội dung | Trạng thái |
| :---: | :--- | :---: |
| **1** | Mô hình hóa bài toán $x, f(x), g(x)$; thiết lập repo, pipeline dữ liệu | ✅ Hoàn thành |
| **2** | Cài đặt hàm chi phí, chạy Baseline, tạo `notebooks/01_baseline.ipynb` | 🔄 Đang thực hiện |
| **3** | Tối ưu Optuna, vẽ đồ thị Cost vs. Threshold, lưu log `results/` | 📅 Dự kiến |
| **4** | So sánh 5 seeds, tính % tiết kiệm chi phí, kiểm thử trên tập Test | 📅 Dự kiến |
| **5** | Hoàn thiện báo cáo 8–12 trang, Release `v1.0`, slide demo | 📅 Dự kiến |

---

## 6. How to Run – Tái lập Thực nghiệm

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Tải dữ liệu và khóa tập Test (SEED = 42)
python src/data_utils.py
```

---

## 7. Repository Structure

```
credit-risk-cost-sensitive-optimization/
├── README.md                        # Tổng quan dự án và mô hình hóa
├── requirements.txt                 # Danh sách thư viện
├── .gitignore                       # Loại bỏ file tạm
├── data/
│   ├── README.md                    # Nguồn gốc và schema dữ liệu
│   ├── dataset_metadata.json        # Thông số tập dữ liệu
│   ├── credit_train.csv             # 80% Train (800 mẫu)
│   └── credit_test.csv              # 20% Locked Test (200 mẫu)
├── notebooks/                       # Jupyter Notebooks theo tuần
├── src/
│   └── data_utils.py                # Script tải và chuẩn bị dữ liệu
├── results/                         # Log thực nghiệm CSV/JSON
├── figures/                         # Đồ thị Cost Curve, Trade-off
└── report/
    └── week1_formulation.md         # Bản mô hình hóa toán học Tuần 1
```

---

## 8. Limitations & AI Assistance

- **Hạn chế:** Tỷ lệ chi phí $C_{\text{FN}} / C_{\text{FP}} = 5:1$ lấy theo benchmark chuẩn quốc tế của UCI; trong thực tế mỗi ngân hàng có biểu phí tổn thất riêng tùy theo chính sách lãi suất và tài sản thế chấp.
- **AI Assistance:** Sử dụng AI hỗ trợ rà soát cấu trúc tài liệu và định dạng LaTeX. Sinh viên trực tiếp kiểm chứng toàn bộ mô hình hóa toán học, chạy thực nghiệm và phân tích kết quả.
