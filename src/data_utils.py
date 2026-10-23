"""
src/data_utils.py
=================
Module tải tập dữ liệu chuẩn Statlog (German Credit Data) từ OpenML,
tiền xử lý bảng dữ liệu kinh tế tài chính và chia tách tập Train (80%) / Test (20%)
cố định với SEED = 42 để chống rò rỉ dữ liệu (Anti-Test-Leakage).

Đề tài: Cost-Sensitive Machine Learning Optimization for Credit Risk Assessment
Môn học: Nhập môn Tối ưu hóa – Trường Đại học Văn Lang
Sinh viên: Đỗ Hoàng Phúc
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# Cấu hình cố định – Không thay đổi trong suốt 5 tuần thực nghiệm
# ==============================================================================
SEED = 42

# Ma trận chi phí chuẩn của German Credit Data (Hofmann, 1994)
# FN: Duyệt nhầm kẻ quỵt nợ → mất vốn gốc          (nặng gấp 5 lần)
# FP: Từ chối nhầm khách hàng tốt → mất tiền lãi   (nhẹ hơn)
COST_FN = 5   # Chi phí 1 trường hợp False Negative (đơn vị chuẩn)
COST_FP = 1   # Chi phí 1 trường hợp False Positive (đơn vị chuẩn)


def load_and_split_data(
    test_size: float = 0.20,
    random_state: int = SEED
):
    """
    Tải tập dữ liệu Statlog (German Credit) từ kho lưu trữ chuẩn OpenML,
    mã hóa các biến phân loại danh mục và chia tập Train / Test cố định.

    Parameters
    ----------
    test_size     : Tỉ lệ tập Test khóa kín (mặc định 20%).
    random_state  : Hạt giống ngẫu nhiên cố định (mặc định 42).

    Returns
    -------
    X_train, X_test, y_train, y_test : Dữ liệu đã chia tách.
    metadata                          : Thông số mô tả tập dữ liệu (dict).
    """
    print("Đang tải Statlog (German Credit Data) từ OpenML (ID: 31)...")
    dataset = fetch_openml(
        name="credit-g",
        version=1,
        as_frame=True,
        parser="auto"
    )
    df: pd.DataFrame = dataset.frame.copy()

    # ------------------------------------------------------------------
    # Chuyển nhãn gốc:  'good' → 0 (Tín dụng tốt)
    #                   'bad'  → 1 (Rủi ro nợ xấu – lớp thiểu số cần phát hiện)
    # ------------------------------------------------------------------
    df["target"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns=["class"])

    # ------------------------------------------------------------------
    # Mã hóa biến phân loại bằng Label Encoding
    # (phù hợp với Random Forest – không cần One-Hot)
    # ------------------------------------------------------------------
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["target"])
    y = df["target"]

    # ------------------------------------------------------------------
    # Chia Stratified 80% Train / 20% Test – Khóa cứng tập Test
    # KHÔNG sử dụng tập Test trong bất kỳ bước tối ưu nào (Tuần 2–4)
    # Chỉ mở khóa tập Test để đánh giá nghiệm tối ưu cuối cùng (Tuần 4–5)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    n_bad_train  = int(y_train.sum())
    n_good_train = int(len(y_train) - n_bad_train)
    n_bad_test   = int(y_test.sum())
    n_good_test  = int(len(y_test) - n_bad_test)

    metadata = {
        "dataset_name":        "Statlog (German Credit Data)",
        "openml_id":           31,
        "source_url":          "https://www.openml.org/d/31",
        "uci_url":             "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "author":              "Prof. Dr. Hans Hofmann, University of Hamburg (1994)",
        "license":             "Public – phục vụ nghiên cứu và giảng dạy",
        "n_total_samples":     int(len(df)),
        "n_features_raw":      20,
        "n_features_encoded":  int(X.shape[1]),
        "label_encoding":      "0 = Good Credit (Tín dụng tốt), 1 = Bad Credit (Rủi ro nợ xấu)",
        "class_distribution":  {
            "good_credit":     int((y == 0).sum()),
            "bad_credit":      int((y == 1).sum())
        },
        "cost_matrix": {
            "cost_fn": COST_FN,
            "cost_fp": COST_FP,
            "ratio_fn_to_fp": f"{COST_FN}:{COST_FP}",
            "note": "Nguồn: Hofmann (1994) – chuẩn benchmark quốc tế"
        },
        "train_set": {
            "n_samples":    int(len(X_train)),
            "n_good":       n_good_train,
            "n_bad":        n_bad_train,
            "bad_ratio_pct": round(n_bad_train / len(X_train) * 100, 2)
        },
        "test_set": {
            "n_samples":    int(len(X_test)),
            "n_good":       n_good_test,
            "n_bad":        n_bad_test,
            "bad_ratio_pct": round(n_bad_test / len(X_test) * 100, 2)
        },
        "split_config": {
            "ratio":        "80% Train (5-Fold Stratified CV) / 20% Locked Test",
            "random_seed":  random_state,
            "stratified":   True
        }
    }

    return X_train, X_test, y_train, y_test, metadata


if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, meta = load_and_split_data()

    # Lưu metadata dạng JSON để kiểm chứng
    with open(out_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    # Lưu tập Train và Test dạng CSV
    train_df = X_train.copy()
    train_df["target"] = y_train.values
    train_df.to_csv(out_dir / "credit_train.csv", index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(out_dir / "credit_test.csv", index=False)

    # In tóm tắt để xác nhận
    print("\n" + "=" * 60)
    print("  Khởi tạo dữ liệu German Credit thành công!")
    print("=" * 60)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("=" * 60)
    print(f"  credit_train.csv : {len(train_df):>5} dòng  →  data/credit_train.csv")
    print(f"  credit_test.csv  : {len(test_df):>5} dòng  →  data/credit_test.csv")
    print(f"  dataset_metadata.json           →  data/dataset_metadata.json")
    print("=" * 60)
