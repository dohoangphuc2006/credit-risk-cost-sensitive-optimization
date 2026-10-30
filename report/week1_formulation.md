# Tuần 1: Đăng ký đề tài và mô hình hóa

**Sinh viên:** Đỗ Hoàng Phúc - Trí tuệ nhân tạo, Trường Đại học Văn Lang.

Đây là ghi chú thiết kế trong GitHub repo. Sinh viên đã chạy chuẩn bị dữ liệu
v2, lưu phiên bản môi trường; dữ liệu và checksum đã được kiểm tra.
Checkpoint tuần 1 đang chờ commit/push.

## 1. Đăng ký đề tài

| Hạng mục | Nội dung |
| --- | --- |
| Tên | Tối ưu chi phí phân loại tín dụng dưới ràng buộc tỷ lệ phê duyệt |
| Câu hỏi | Tối ưu cả ngưỡng và siêu tham số có giảm chi phí so với RF mặc định và RF chỉ tối ưu ngưỡng, khi yêu cầu phê duyệt ít nhất 65% không? |
| Dataset | Statlog German Credit, OpenML ID 31, 1.000 mẫu, 20 đặc trưng, 700 good / 300 bad |
| Quyền sử dụng | CC BY 4.0 theo trang UCI; ghi nguồn và DOI khi chia sẻ |
| Biến quyết định | Ngưỡng, số cây, độ sâu, mẫu tối thiểu tách nhánh, trọng số lớp bad |
| Objective | Chi phí phân loại trung bình trên dự đoán OOF của tập train |
| Constraint | Tỷ lệ từ chối OOF không vượt 35% theo giả định nghiên cứu |
| Baselines | RF mặc định tại 0.5; RF mặc định chỉ tối ưu ngưỡng trên OOF |
| Phương pháp | Optuna TPE; Random Search đối chứng cùng ngân sách |
| Metric | Cost/sample chính; FN, FP, recall bad, approval rate và feasibility phụ |
| Budget | 50 trials / sampler seed / phương pháp; 5 sampler seeds; 5-fold CV |
| Sản phẩm | Repo có source/notebook, config, logs, bảng/hình và hướng dẫn tái lập |
| Repository | [credit-risk-cost-sensitive-optimization](https://github.com/dohoangphuc2006/credit-risk-cost-sensitive-optimization) |

## 2. Giả định và ký hiệu

- Nhãn $y=0$: good; $y=1$: bad. Dự đoán 1 tương ứng từ chối trong mô phỏng.
- Chi phí $C_{FN}=5$, $C_{FP}=1$ là đơn vị tương đối từ benchmark UCI.
  Không quy đổi thành tiền thật hay giả định mọi trường hợp bad mất toàn bộ vốn.
- Phê duyệt tối thiểu 65% là giả định nghiên cứu, không phải quy định ngân hàng.
- $\mathbf{u}_i$ là đặc trưng hồ sơ; $z$ là vector cấu hình thuật toán thay đổi.
- Dùng phép chia ngẫu nhiên phân tầng để nghiên cứu benchmark; chưa đánh giá
  theo thời gian hoặc xác nhận khả năng triển khai thực tế.

## 3. Biến quyết định và miền tìm kiếm

$$z=(\tau,n,d,s,w)\in\Omega$$

$$\Omega=[0.10,0.90]\times\{20,\ldots,200\}\times\{2,\ldots,15\}\times\{2,\ldots,20\}\times[1,10].$$

| Biến | Ánh xạ cài đặt |
| --- | --- |
| $\tau$ | `threshold`: từ chối nếu xác suất bad lớn hơn hoặc bằng ngưỡng |
| $n$ | `n_estimators` |
| $d$ | `max_depth` |
| $s$ | `min_samples_split` |
| $w$ | `bad_class_weight`, chuyển thành `class_weight={0: 1.0, 1: w}` |

Giữ các tham số RF khác ở mặc định của phiên bản ghi trong môi trường,
và đặt `random_state=42`. Config chuẩn: `configs/experiment.json`.

## 4. Hàm mục tiêu trên dự đoán OOF

Tập train có $N=800$ mẫu, chia cố định thành $K=5$ folds. Với mỗi cấu hình,
fit pipeline mới trên train của fold, rồi dự đoán xác suất bad trên validation.
Mỗi hồ sơ train có đúng một dự đoán OOF:

$$p_i(z)=P_{\mathcal M_{z,-k(i)}}(y=1\mid\mathbf u_i),\qquad
\hat y_i(z)=\mathbb 1[p_i(z)\ge\tau].$$

$$f(z)=\frac{1}{N}\sum_{i=1}^{N}\left(5\,\mathbb 1[y_i=1,\hat y_i=0]+\mathbb 1[y_i=0,\hat y_i=1]\right).$$

Đây là $\left(5FN_{OOF}+FP_{OOF}\right)/N$, cần **cực tiểu hóa**.
Với năm folds bằng nhau, công thức bằng trung bình chi phí/sample từng fold.
Khi kích thước fold khác nhau, dùng công thức gộp OOF để mỗi hồ sơ có trọng số bằng nhau.

| Quyết định / Nhãn thật | Good (0) | Bad (1) |
| --- | ---: | ---: |
| Chấp thuận (0) | TN, chi phí 0 | FN, chi phí 5 |
| Từ chối (1) | FP, chi phí 1 | TP, chi phí 0 |

## 5. Ràng buộc và cơ chế phạt dự kiến

$$r(z)=\frac{TP_{OOF}(z)+FP_{OOF}(z)}{N},\qquad g(z)=r(z)-0.35\le0.$$

Ràng buộc đo **trên toàn bộ OOF gộp**, không yêu cầu riêng từng fold đều dưới
35%. Vẫn lưu tỷ lệ từng fold để phân tích biến động.

Ở tuần 3, objective tìm kiếm thống nhất cho Random Search và TPE được thiết kế:

$$J(z)=\begin{cases}f(z),&g(z)\le0,\\C_{max}+f(z)+\lambda g(z),&g(z)>0,\end{cases}\quad C_{max}=\max(C_{FN},C_{FP})=5,\ \lambda=10.$$

Mỗi hồ sơ chịu tối đa $C_{max}$ đơn vị nên $0\le f(z)\le C_{max}$.
Mọi trial vi phạm có $J(z)>C_{max}$, kém hơn mọi trial khả thi.
Đây là penalty do code dự án cài, không phải Optuna tự động loại trial.
Trial vi phạm vẫn ghi log $f,r,g,J$ và `feasible=False`.

Chỉ chọn các trial khả thi. Nếu không có, báo `no feasible solution found
within budget`; không tự nới ràng buộc và không chọn bằng test.

## 6. Trade-off và quy tắc chọn nghiệm

Với **cùng mô hình và cùng dự đoán xác suất**:

| Thay đổi | Tỷ lệ từ chối | FN | FP |
| --- | --- | --- | --- |
| Giảm $\tau$ | Tăng hoặc giữ nguyên | Giảm hoặc giữ nguyên | Tăng hoặc giữ nguyên |
| Tăng $\tau$ | Giảm hoặc giữ nguyên | Tăng hoặc giữ nguyên | Giảm hoặc giữ nguyên |

Tổng chi phí không nhất thiết đơn điệu vì hai loại lỗi thay đổi ngược chiều.
Khi đổi cả siêu tham số, không dùng bảng này để suy ra quan hệ giữa hai mô hình khác nhau.

Chọn trial khả thi có $f$ OOF thấp nhất trong phương pháp đã chọn bằng CV.
Nếu bằng nhau, ưu tiên ít cây hơn, rồi độ sâu nhỏ hơn, rồi sampler seed theo
thứ tự config và trial number nhỏ hơn. Giữ mọi lần chạy để báo mean/std,
không chỉ đưa seed tốt nhất. Phương pháp cũng chọn bằng CV trước đánh giá test.
Nếu hòa giữa các phương pháp, ưu tiên RF chỉ tối ưu ngưỡng, rồi Random Search,
rồi TPE để dùng phương án đơn giản hơn. RF mặc định tại 0.5 là mốc tham khảo.

## 7. Protocol tái lập và chống rò rỉ

1. Tải `fetch_openml(data_id=31, as_frame=True)`, kiểm tra shape và nhãn.
2. Đổi **nhãn** good/bad thành 0/1; giữ nguyên đặc trưng danh mục.
3. Chia stratified 80/20 bằng seed 42; lưu source row IDs và checksum.
4. Lưu năm folds phân tầng chỉ trên train, CV seed 42. Mọi phương án dùng
   cùng folds, thứ tự train và preprocessing.
5. Mỗi fold dùng pipeline preprocessing và RF; chỉ fit imputer/One-Hot và RF
   trên train của fold. Không fit encoder trên toàn bộ train trước CV hoặc trên test.
6. Lấy OOF để tính cost, constraint và chọn ngưỡng/cấu hình. Baseline chỉ
   tối ưu ngưỡng dùng lưới 0.10-0.90, bước 0.01, áp cùng ràng buộc.
7. Sau khi chốt phương pháp và nghiệm, fit pipeline mới trên 800 mẫu train;
   dùng test cho đánh giá cuối. Báo cả feasibility test; không dùng vi phạm
   test để chỉnh ngưỡng rồi gọi đó là cùng test cuối.

Test được giữ riêng **theo protocol**; việc file có trên GitHub không có
nghĩa là được dùng nhãn test để chọn nghiệm. Chuẩn bị dữ liệu chỉ ghi thống
kê lớp và checksum, không đánh giá mô hình.

| Nguồn ngẫu nhiên | Giá trị | Ý nghĩa |
| --- | --- | --- |
| Split seed | 42 cố định | Giữ nguyên 800/200 hồ sơ |
| CV seed | 42 cố định | Giữ nguyên năm folds |
| RF seed | 42 cố định | Giảm nhiễu khi so cấu hình và thuật toán |
| Sampler seeds | 42, 100, 2024, 2025, 2026 | Đo độ ổn định do Random Search/TPE |

Mỗi phương pháp tìm kiếm: 50 trials x 5 sampler seeds x 5 folds = 1.250 lần
fit RF. Baseline và quét ngưỡng ghi chi phí riêng; quét ngưỡng dùng lại OOF
nên không fit RF cho từng ngưỡng. Chưa dùng pruning/early stopping để giữ
số lần đánh giá giữa Random Search/TPE tương đương.

## 8. Kế hoạch kết quả ở các tuần tiếp theo

- Bảng cost/sample, FN, FP, recall bad, approval rate, feasibility và runtime.
- Trial history: method, sampler seed, trial number, params, cost, rejection
  rate, constraint violation, penalized objective, feasible flag và duration.
- Đồ thị best-feasible-cost-so-far theo số lần đánh giá; để trống trước trial
  khả thi đầu tiên, không vẽ penalty như chi phí kinh doanh.
- Đồ thị cost/approval theo threshold, phân bố kết quả qua sampler seeds.
- Độ nhạy CV: chi phí FN trong `{2, 5, 10}` và trần từ chối trong
  `{0.25, 0.35, 0.45}`. Giữ kịch bản chính 5:1/35%.
- Tiết kiệm tương đối: $100(f_{baseline}-f_{chosen})/f_{baseline}$ nếu
  $f_{baseline}>0$; báo không xác định khi mẫu số bằng 0. Nêu rõ baseline
  có khả thi hay không và có thể báo cải thiện âm.
- Không quy đổi chi phí benchmark thành tiền tiết kiệm thực tế.

## 9. Trạng thái và giới hạn

- [x] Câu hỏi nghiên cứu, biến, objective, constraint và giả định.
- [x] Baselines, metric, budget, seeds, quy tắc chọn nghiệm và protocol.
- [x] Code chuẩn bị dữ liệu v2 và transformer chưa fit.
- [x] Sinh viên chạy v2; xác nhận 800/200 mẫu và 5 folds.
- [x] Lưu `requirements-lock.txt` sau khi cài môi trường và chạy thành công.
- [ ] Commit/push checkpoint sau khi kiểm tra output.

Chưa có số liệu thực nghiệm. German Credit nhỏ và cũ; chi phí benchmark và
ràng buộc giả định chưa phản ánh sản phẩm ngân hàng thực. Năm sampler seeds
không đại diện cho nhiều mẫu dữ liệu độc lập.

## 10. Nguồn và AI assistance

- [UCI: Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
- [OpenML ID 31](https://www.openml.org/d/31).
- Hofmann, H. (1994). Statlog (German Credit Data). [DOI](https://doi.org/10.24432/C5NC77).
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- AI hỗ trợ mô hình hóa, chỉnh code, kiểm tra artifact và cập nhật tài liệu.
  Sinh viên đã chạy chuẩn bị dữ liệu v2 và lưu phiên bản môi trường.
  Chưa huấn luyện baseline hoặc chạy tối ưu; sinh viên chịu trách nhiệm
  hiểu và giải thích dữ liệu, công thức và pipeline.
