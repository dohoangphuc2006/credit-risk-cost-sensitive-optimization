# Cost-Sensitive Credit Risk Optimization

**Tối ưu chi phí phân loại tín dụng dưới ràng buộc tỷ lệ phê duyệt**

Bài tập lớn cá nhân môn Nhập môn Tối ưu hóa, thực hiện trong 5 tuần.

- **Sinh viên:** Đỗ Hoàng Phúc
- **Chuyên ngành:** Trí tuệ nhân tạo, Trường Đại học Văn Lang
- **Sản phẩm:** GitHub repository gồm mã nguồn, mô hình hóa, log và hình kết quả.
- **Checkpoint:** Đã hoàn thành chuẩn bị dữ liệu tuần 1: 800 mẫu train, 200 mẫu test, 5 CV folds và file phiên bản môi trường. Đã kiểm tra dữ liệu và checksum; chờ commit/push.

## 1. Problem

Trong phân loại tín dụng, hai loại sai lầm có thể mang chi phí khác nhau.
Với nhãn `0 = good`, `1 = bad`, quyết định dự đoán `1` tương ứng với từ chối:

| Sai lầm | Diễn giải trong nghiên cứu | Chi phí tương đối |
| --- | --- | ---: |
| False Negative (FN) | Chấp thuận hồ sơ thuộc lớp tín dụng xấu | 5 |
| False Positive (FP) | Từ chối hồ sơ thuộc lớp tín dụng tốt | 1 |

Ma trận 5:1 lấy từ tài liệu German Credit của UCI. Đây là chi phí benchmark,
không phải số tiền thiệt hại đã đo tại một ngân hàng.

**Câu hỏi nghiên cứu:** Với giả định phải phê duyệt ít nhất 65% hồ sơ, tối ưu
đồng thời ngưỡng phân loại và siêu tham số Random Forest có giảm chi phí so
với mô hình mặc định và phương án chỉ tối ưu ngưỡng không?

## 2. Optimization formulation

Gọi cấu hình quyết định là $z=(\tau,n,d,s,w)$; ký hiệu $\mathbf{u}_i$ dành
cho đặc trưng hồ sơ để tránh nhầm với vector cấu hình.

| Biến | Ý nghĩa | Miền tìm kiếm |
| --- | --- | --- |
| $\tau$ | Ngưỡng từ chối: $\hat y_i=1$ nếu $p_i\ge\tau$ | $[0.10,0.90]$ |
| $n$ | `n_estimators` | Số nguyên từ 20 đến 200 |
| $d$ | `max_depth` | Số nguyên từ 2 đến 15 |
| $s$ | `min_samples_split` | Số nguyên từ 2 đến 20 |
| $w$ | Trọng số lớp xấu | $[1,10]$, ánh xạ `class_weight={0: 1.0, 1: w}` |

Trên toàn bộ dự đoán out-of-fold (OOF) của **800 hồ sơ train**:

$$\min_{z\in\Omega} f(z)=\frac{5\,FN_{\mathrm{OOF}}(z)+FP_{\mathrm{OOF}}(z)}{N_{\mathrm{train}}}$$

$$g(z)=\frac{TP_{\mathrm{OOF}}(z)+FP_{\mathrm{OOF}}(z)}{N_{\mathrm{train}}}-0.35\le0.$$

Mức phê duyệt 65% là **giả định chính sách của nghiên cứu**, không phải quy
định chung của ngành ngân hàng. Với cùng mô hình, giảm $\tau$ làm tỷ lệ từ
chối tăng hoặc giữ nguyên, FN giảm hoặc giữ nguyên, FP tăng hoặc giữ nguyên.

Kế hoạch tối ưu dùng objective phạt cho trial không khả thi; lựa chọn cuối
chỉ xét các trial đáp ứng ràng buộc. Công thức phạt, quy tắc hòa điểm và
protocol được ghi tại [mô hình hóa tuần 1](report/week1_formulation.md).
Đây là thiết kế cho tuần 3; chưa có kết quả tối ưu ở checkpoint tuần 1.

## 3. Dataset and protocol

- **Dataset:** Statlog (German Credit Data), 1.000 mẫu, 20 đặc trưng.
- **Nguồn cố định:** [OpenML ID 31](https://www.openml.org/d/31), `credit-g`.
- **Nguồn gốc và ma trận chi phí:** [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
- **Citation:** Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. [DOI: 10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
- **Giấy phép dữ liệu:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), theo trang UCI; giữ ghi nguồn khi chia sẻ.
- **Chia dữ liệu:** stratified 80% train / 20% test, seed 42; giữ `source_row_id` để truy vết.
- **CV:** 5 folds phân tầng cố định trên train, shuffle và seed 42.
- **Preprocessing:** chia dữ liệu trước; tạo One-Hot encoder và imputer trong `Pipeline` của từng fold. Tuần 1 chỉ tạo transformer chưa fit.
- **Test holdout:** chỉ đánh giá sau khi chốt phương pháp và cấu hình bằng train CV. Tính hash và ghi thống kê chia dữ liệu không phải đánh giá mô hình trên test.

```mermaid
flowchart TD
    A[OpenML ID 31: đặc trưng gốc và nhãn] --> B[Chia stratified: seed 42]
    B --> C[800 hồ sơ train]
    B --> D[200 hồ sơ test holdout]
    C --> E[5 folds cố định]
    E --> F[Fit preprocessing và mô hình trên train của từng fold]
    F --> G[Dự đoán validation: ghép OOF]
    G --> H[Chi phí và tỷ lệ từ chối OOF]
    H --> I[Tìm kiếm cấu hình: tuần 3]
    I --> E
    H --> J[Chốt phương pháp và cấu hình bằng CV]
    J --> K[Fit pipeline trên toàn bộ train]
    K --> L[Đánh giá cuối và nêu giới hạn]
    D --> L
```

Hai CSV train/test đã được tạo lại với đặc trưng chưa mã hóa.
Metadata có `artifact_format_version = 2`; đã lưu năm CV folds và kiểm tra
checksum của dữ liệu, folds và cấu hình. Loader từ chối artifact v1.
CSV `german_credit_data.csv` cũ có 9 đặc trưng và không có nhãn đích; không
được dùng trong pipeline này. Xem [mô tả dữ liệu](data/README.md).

## 4. Baselines and optimization plan

| Phương án dự kiến | Vai trò |
| --- | --- |
| RF mặc định, `class_weight=None`, $\tau=0.5$ | Baseline ban đầu |
| RF mặc định, ngưỡng chọn bằng OOF trên lưới 0.10 đến 0.90, bước 0.01 | Xác định đóng góp riêng của ngưỡng |
| Random Search, tối ưu 5 biến | Đối chứng thuật toán tìm kiếm |
| Optuna TPE, tối ưu 5 biến | Phương pháp chính |

Random Search và TPE dùng cùng search space, folds, preprocessing và ngân sách
**50 trials / sampler seed**, lặp với `42, 100, 2024, 2025, 2026`.
Seed mô hình RF giữ ở 42 để tập trung đo biến động do thuật toán tìm kiếm.
Một trial fit 5 mô hình theo CV; tương ứng 1.250 lần fit cho mỗi phương pháp
khi chạy đủ 5 sampler seeds. Số lần fit và runtime sẽ được ghi lại.

Tất cả phương án được báo cùng cost/sample, FN, FP, recall lớp bad, tỷ lệ
phê duyệt và tính khả thi. RF mặc định vẫn được báo nếu vi phạm ràng buộc,
nhưng không được xem là một phương án triển khai khả thi.

## 5. Key results and five-week plan

**Chưa có số liệu baseline hoặc tối ưu.** Không dùng số liệu dự kiến làm kết quả.

| Tuần | Nội dung | Sản phẩm trong repo | Trạng thái |
| --- | --- | --- | --- |
| 1 | Chốt câu hỏi, mô hình hóa, nguồn dữ liệu, config, split/CV và preprocessing | Code, README, formulation, dữ liệu v2, CV folds, metadata và requirements-lock.txt | Đã chuẩn bị và kiểm tra dữ liệu; chờ commit/push |
| 2 | Chạy baseline mặc định và baseline tối ưu ngưỡng | Notebook/source, bảng CV và log | Dự kiến |
| 3 | Cài objective, penalty, Random Search và TPE | Source, CSV/JSON trial history, best-feasible-so-far plot | Dự kiến |
| 4 | Chạy nhiều sampler seeds; phân tích độ ổn định và độ nhạy trên CV | Bảng mean/std, runtime, cost/approval trade-off, sensitivity | Dự kiến |
| 5 | Chốt nghiệm bằng CV, đánh giá test cuối, hoàn thiện portfolio | Bảng kết quả, figures, hướng dẫn tái lập, release `v1.0` | Dự kiến |

Tài liệu mô hình hóa nằm trong repository; sản phẩm của dự án là repo GitHub.

## 6. How to run Week 1 (PowerShell)

Mở terminal tại thư mục gốc repo. Khuyến nghị Python 3.11; code yêu cầu Python
3.10 trở lên. Các lệnh dùng trực tiếp Python trong venv nên không cần thay
PowerShell execution policy để kích hoạt môi trường.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data_utils --force
```

`--force` thay thế hai CSV train/test và metadata cũ, đồng thời ghi
`data/cv_folds.json`. Lệnh tải đúng OpenML ID 31 và cần Internet ở lần đầu;
cache nằm trong `.cache/`, đã được Git bỏ qua. Split seed vẫn là 42.

Đầu ra của lần chuẩn bị dữ liệu thành công (đường dẫn rút gọn):

```text
Prepared OpenML ID 31 in <repository>/data
Train: 800 rows; test: 200 rows.
Saved 5 fixed CV folds and SHA-256 checksums.
Features are unencoded. No model training or test evaluation has been performed.
```

Đọc lại train và folds, không huấn luyện mô hình:

```powershell
.\.venv\Scripts\python.exe -c "from src.data_utils import load_training_data, load_cv_splits; X, y = load_training_data(); folds = load_cv_splits(X); print('Train shape:', X.shape); print('Labels:', y.value_counts().to_dict()); print('CV folds:', len(folds))"
```

Kỳ vọng: train `(800, 20)`, nhãn `{0: 560, 1: 240}`, 5 folds.
Metadata v2 lưu cấu hình, phiên bản môi trường, row IDs và SHA-256 của các file.
Khi thay config, cần tạo lại artifact; không đổi split để chọn kết quả đẹp hơn.

Sau khi chạy thành công, lưu phiên bản môi trường và cập nhật trạng thái checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements-lock.txt
```

Người khác có thể cài lại môi trường Python tương ứng bằng
`python -m pip install -r requirements-lock.txt`. File này đã được sinh viên
tạo sau khi cài thư viện và chuẩn bị dữ liệu thành công.

## 7. Repository structure

```text
credit-risk-cost-sensitive-optimization/
├── README.md
├── requirements.txt
├── requirements-lock.txt        # Phiên bản thư viện của môi trường đã chạy
├── .gitignore
├── configs/
│   └── experiment.json          # Config chuẩn: data, seeds, costs, policy, budget
├── src/
│   ├── __init__.py
│   ├── config.py                # Đọc và kiểm tra config
│   ├── data_utils.py            # Chuẩn bị dữ liệu; đọc train và CV folds
│   └── preprocessing.py         # Transformer chưa fit, dùng trong pipeline CV
├── data/
│   ├── README.md
│   ├── credit_train.csv         # 800 mẫu với đặc trưng chưa mã hóa
│   ├── credit_test.csv
│   ├── dataset_metadata.json
│   ├── cv_folds.json            # Năm folds đã lưu từ lần chạy v2
│   └── german_credit_data.csv   # File cũ, không dùng trong thực nghiệm
├── notebooks/
├── results/
├── figures/
└── report/
    └── week1_formulation.md     # Ghi chú mô hình hóa trong repo
```

## 8. Limitations and AI assistance

- German Credit là benchmark nhỏ, công bố năm 1994; không đại diện cho tín dụng
  Việt Nam hiện nay. Nhãn bad không chứng minh hành vi cố tình không trả nợ.
- Chi phí 5:1 và mức phê duyệt 65% không đủ để suy ra tiết kiệm tiền thật.
- Feasibility trên CV không đảm bảo tỷ lệ phê duyệt tương lai; test cuối phải
  báo cả chi phí lẫn mức vi phạm ràng buộc, nếu có.
- Năm sampler seeds đo độ ổn định tìm kiếm trên một split/CV cố định,
  không đo đầy đủ bất định lấy mẫu hay thay đổi phân phối dữ liệu.
- **AI assistance:** AI hỗ trợ mô hình hóa, viết/chỉnh code chuẩn bị dữ liệu,
  preprocessing và tài liệu tuần 1. Sinh viên đã chạy chuẩn bị dữ liệu v2
  và lưu phiên bản môi trường; dữ liệu và checksum đã được kiểm tra khi
  cập nhật checkpoint. Chưa huấn luyện baseline hoặc chạy tối ưu.
  Sinh viên chịu trách nhiệm hiểu và giải thích từng bước của dự án.

## 9. Commit and push after running

Sau khi chuẩn bị dữ liệu thành công, cập nhật trạng thái tuần 1 trong README
và kiểm tra thay đổi trước khi đẩy lên GitHub:

```powershell
git status --short
git diff --stat
git diff --check
git add README.md .gitignore requirements.txt configs src data report/week1_formulation.md
git add requirements-lock.txt
git diff --cached --stat
git commit -m "feat(week1): finalize formulation and reproducible data preparation"
git push origin main
```

Nếu Git báo remote có commit mới, dùng `git pull --rebase origin main` rồi
`git push origin main`; nếu có conflict, xử lý conflict trước khi tiếp tục.
Không dùng force-push. Sau khi push, mở GitHub để xác nhận commit và các file mới.
