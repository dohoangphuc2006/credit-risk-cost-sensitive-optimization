"""
src/baseline.py
===============
Week 2: Measure default baseline and threshold-optimized baseline.

Run two baseline schemes on Train (800 samples) via 5-Fold Stratified CV:

  1) RF default (class_weight=None, threshold=0.5)
  2) RF default + sweep threshold on OOF (threshold in [0.10, 0.90], step 0.01)

Test set (200 samples) is NOT used in any computation.
Results saved to results/ and figures/.

Run:  .venv/Scripts/python -m src.baseline
"""

import json
import os
import sys
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path

# Fix encoding for Windows terminal
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")

import matplotlib
matplotlib.use("Agg")  # Render offline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    from config import load_config, PROJECT_ROOT, DATA_DIR, CONFIG_PATH
    from data_utils import load_training_data, load_cv_splits, file_sha256, write_json, _environment_versions
    from preprocessing import build_preprocessor
else:
    from .config import load_config, PROJECT_ROOT, DATA_DIR, CONFIG_PATH
    from .data_utils import load_training_data, load_cv_splits, file_sha256, write_json, _environment_versions
    from .preprocessing import build_preprocessor



# ═══════════════════════════════════════════════════════════════════════════════
# 1. CÁC HÀM TÍNH TOÁN CỐT LÕI CỦA BÀI TOÁN TỐI ƯU
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cost_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         cost_fn: float, cost_fp: float) -> dict:
    """
    Tính chi phí theo ma trận giả định (đơn vị tương đối) và các chỉ số.

    Parameters
    ----------
    y_true : Nhãn thật (0 = Good, 1 = Bad).
    y_pred : Nhãn dự đoán (0 = Duyệt, 1 = Từ chối).
    cost_fn : Chi phí mỗi ca False Negative (lọt nợ xấu).
    cost_fp : Chi phí mỗi ca False Positive (từ chối nhầm người tốt).

    Returns
    -------
    dict chứa: total_cost, cost_per_sample, confusion matrix counts,
               rejection_rate, approval_rate. Ràng buộc được xét ở bên gọi.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)

    total_cost = cost_fn * fn + cost_fp * fp
    cost_per_sample = total_cost / n
    rejection_rate = (tp + fp) / n
    approval_rate = 1.0 - rejection_rate

    return {
        "n_samples": n,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "total_cost": float(total_cost),
        "cost_per_sample": float(cost_per_sample),
        "rejection_rate": float(rejection_rate),
        "approval_rate": float(approval_rate),
    }


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Áp ngưỡng τ: P(Bad) ≥ τ → Từ chối (1), ngược lại → Duyệt (0)."""
    return (y_proba >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHẠY BASELINE: RF MẶC ĐỊNH VỚI NGƯỠNG CỐ ĐỊNH τ = 0.5
# ═══════════════════════════════════════════════════════════════════════════════

def run_default_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list,
    config: dict,
) -> dict:
    """
    Chạy RandomForest mặc định (class_weight=None, τ=0.5) trên 5-Fold CV.
    Thu thập dự đoán Out-of-Fold (OOF) rồi tính chi phí trên toàn bộ 800 mẫu.
    """
    cost_fn = config["cost"]["false_negative"]
    cost_fp = config["cost"]["false_positive"]
    model_seed = config["model_seed"]
    max_rej = config["policy"]["max_rejection_rate"]
    threshold = 0.5

    oof_proba = np.full(len(X_train), np.nan)
    fold_results = []

    for fold_idx, (train_pos, val_pos) in enumerate(cv_splits):
        X_fold_train = X_train.iloc[train_pos]
        y_fold_train = y_train.iloc[train_pos]
        X_fold_val = X_train.iloc[val_pos]
        y_fold_val = y_train.iloc[val_pos]

        # Pipeline: Preprocessing (fit trên fold train) → RF mặc định
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                random_state=model_seed,
                n_jobs=1,
            )),
        ])
        pipe.fit(X_fold_train, y_fold_train)

        # Xác suất P(Bad) trên validation fold
        proba_val = pipe.predict_proba(X_fold_val)[:, 1]
        oof_proba[val_pos] = proba_val

        # Tính chỉ số của từng fold
        y_pred_fold = apply_threshold(proba_val, threshold)
        fold_cost = compute_cost_metrics(
            y_fold_val.values, y_pred_fold, cost_fn, cost_fp
        )
        fold_cost["fold"] = fold_idx
        fold_results.append(fold_cost)

        print(f"  Fold {fold_idx}: cost/sample = {fold_cost['cost_per_sample']:.4f}, "
              f"FN = {fold_cost['FN']}, FP = {fold_cost['FP']}, "
              f"rejection_rate = {fold_cost['rejection_rate']:.2%}")

    # Tính chi phí gộp trên toàn bộ OOF (800 mẫu)
    assert not np.isnan(oof_proba).any(), "Có mẫu chưa có dự đoán OOF."
    oof_pred = apply_threshold(oof_proba, threshold)
    pooled = compute_cost_metrics(y_train.values, oof_pred, cost_fn, cost_fp)

    # Ràng buộc
    g_value = pooled["rejection_rate"] - max_rej
    feasible = g_value <= 0

    # Metric ML truyền thống (để đối chiếu)
    ml_metrics = {
        "accuracy": round(accuracy_score(y_train, oof_pred), 4),
        "precision_bad": round(precision_score(y_train, oof_pred, pos_label=1, zero_division=0), 4),
        "recall_bad": round(recall_score(y_train, oof_pred, pos_label=1, zero_division=0), 4),
        "f1_bad": round(f1_score(y_train, oof_pred, pos_label=1, zero_division=0), 4),
        "f1_macro": round(f1_score(y_train, oof_pred, average="macro", zero_division=0), 4),
    }

    return {
        "method": "RF_default_threshold_0.5",
        "threshold": threshold,
        "model_params": "scikit-learn defaults (n_estimators=100, max_depth=None, class_weight=None)",
        "pooled_oof": pooled,
        "constraint_g": round(float(g_value), 4),
        "feasible": feasible,
        "max_rejection_rate": max_rej,
        "ml_metrics": ml_metrics,
        "fold_details": fold_results,
        "oof_proba": oof_proba.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BASELINE TỐI ƯU NGƯỠNG: QUÉT τ ∈ [0.10, 0.90] TRÊN OOF CỦA RF MẶC ĐỊNH
# ═══════════════════════════════════════════════════════════════════════════════

def run_threshold_sweep(
    y_train: pd.Series,
    oof_proba: np.ndarray,
    config: dict,
) -> dict:
    """
    Giữ nguyên mô hình RF mặc định (đã fit), chỉ quét ngưỡng τ trên OOF.
    Tìm τ* cho chi phí thấp nhất thỏa ràng buộc rejection_rate ≤ 35%.
    """
    cost_fn = config["cost"]["false_negative"]
    cost_fp = config["cost"]["false_positive"]
    max_rej = config["policy"]["max_rejection_rate"]

    thresholds = np.arange(10, 91) / 100.0
    sweep_rows = []

    for tau in thresholds:
        y_pred = apply_threshold(oof_proba, tau)
        metrics = compute_cost_metrics(y_train.values, y_pred, cost_fn, cost_fp)
        g_val = metrics["rejection_rate"] - max_rej
        sweep_rows.append({
            "threshold": round(float(tau), 2),
            "cost_per_sample": metrics["cost_per_sample"],
            "rejection_rate": metrics["rejection_rate"],
            "approval_rate": metrics["approval_rate"],
            "FN": metrics["FN"],
            "FP": metrics["FP"],
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "constraint_g": round(float(g_val), 4),
            "feasible": g_val <= 0,
        })

    sweep_df = pd.DataFrame(sweep_rows)

    # Tìm nghiệm tối ưu trong các cấu hình khả thi
    feasible_df = sweep_df[sweep_df["feasible"]]
    if feasible_df.empty:
        print("  ⚠ Không có ngưỡng nào thỏa ràng buộc!")
        best_row = None
    else:
        best_idx = feasible_df["cost_per_sample"].idxmin()
        best_row = feasible_df.loc[best_idx].to_dict()
        print(f"  Ngưỡng tối ưu τ* = {best_row['threshold']:.2f}, "
              f"cost/sample = {best_row['cost_per_sample']:.4f}, "
              f"rejection_rate = {best_row['rejection_rate']:.2%}")

    # Metric ML tại ngưỡng tối ưu
    ml_metrics_best = None
    if best_row is not None:
        y_pred_best = apply_threshold(oof_proba, best_row["threshold"])
        ml_metrics_best = {
            "accuracy": round(accuracy_score(y_train, y_pred_best), 4),
            "precision_bad": round(precision_score(y_train, y_pred_best, pos_label=1, zero_division=0), 4),
            "recall_bad": round(recall_score(y_train, y_pred_best, pos_label=1, zero_division=0), 4),
            "f1_bad": round(f1_score(y_train, y_pred_best, pos_label=1, zero_division=0), 4),
            "f1_macro": round(f1_score(y_train, y_pred_best, average="macro", zero_division=0), 4),
        }

    return {
        "method": "RF_default_threshold_sweep",
        "model_params": "scikit-learn defaults (same OOF probabilities as baseline)",
        "sweep_range": {"low": 0.10, "high": 0.90, "step": 0.01},
        "best_feasible": best_row,
        "ml_metrics_best": ml_metrics_best,
        "sweep_table": sweep_rows,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VẼ ĐỒ THỊ
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cost_vs_threshold(sweep_data: dict, baseline_data: dict,
                           config: dict, fig_dir: Path) -> None:
    """
    Vẽ đồ thị Cost vs. Threshold — Biểu đồ trung tâm của bài toán Tối ưu hóa.
    Trục X: Ngưỡng τ, Trục Y: Chi phí tổn thất trung bình f(z).
    Đường đứt nét đỏ: Giới hạn ràng buộc rejection_rate = 35%.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sweep_data["sweep_table"])
    max_rej = config["policy"]["max_rejection_rate"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Week 2 – Baseline Analysis: Cost vs. Threshold",
                 fontsize=14, fontweight="bold")

    # ── Subplot 1: Chi phí theo ngưỡng ──
    ax1.plot(df["threshold"], df["cost_per_sample"],
             "o-", markersize=3, color="#2196F3", linewidth=1.5,
             label="Cost/sample $f(z)$")

    # Đánh dấu Baseline mặc định (τ = 0.5)
    bl_cost = baseline_data["pooled_oof"]["cost_per_sample"]
    ax1.axhline(y=bl_cost, color="gray", linestyle="--", alpha=0.7,
                label=f"Baseline τ=0.5 (cost={bl_cost:.4f})")
    ax1.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)

    # Đánh dấu ngưỡng tối ưu
    best = sweep_data["best_feasible"]
    if best is not None:
        ax1.plot(best["threshold"], best["cost_per_sample"],
                 "r*", markersize=15, zorder=5,
                 label=f"Best feasible τ*={best['threshold']:.2f} "
                       f"(cost={best['cost_per_sample']:.4f})")

    # Tô vùng vi phạm ràng buộc
    infeasible = df[~df["feasible"]]
    if not infeasible.empty:
        ax1.axvspan(infeasible["threshold"].min(), infeasible["threshold"].max(),
                    alpha=0.08, color="red", label="Infeasible region (rej > 35%)")

    ax1.set_ylabel("Cost per sample $f(z)$", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Tỷ lệ từ chối theo ngưỡng ──
    ax2.plot(df["threshold"], df["rejection_rate"],
             "o-", markersize=3, color="#FF9800", linewidth=1.5,
             label="Rejection rate $r(z)$")
    ax2.axhline(y=max_rej, color="red", linestyle="--", linewidth=2,
                label=f"Constraint: rej ≤ {max_rej:.0%}")
    ax2.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)

    if best is not None:
        ax2.plot(best["threshold"], best["rejection_rate"],
                 "r*", markersize=15, zorder=5,
                 label=f"τ*={best['threshold']:.2f} "
                       f"(rej={best['rejection_rate']:.2%})")

    ax2.set_xlabel("Threshold τ", fontsize=11)
    ax2.set_ylabel("Rejection rate $r(z)$", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = fig_dir / "week2_cost_vs_threshold.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu đồ thị: {out_path}")


def plot_confusion_matrix(y_true, y_pred, title: str,
                          config: dict, fig_dir: Path, filename: str) -> None:
    """Vẽ ma trận nhầm lẫn gộp (Pooled OOF) kèm chi phí mỗi ô."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    cost_fn = config["cost"]["false_negative"]
    cost_fp = config["cost"]["false_positive"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    labels_text = [
        [f"TN = {tn}\ncost = 0", f"FP = {fp}\ncost = {fp}×{cost_fp:.0f} = {fp*cost_fp:.0f}"],
        [f"FN = {fn}\ncost = {fn}×{cost_fn:.0f} = {fn*cost_fn:.0f}", f"TP = {tp}\ncost = 0"],
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, labels_text[i][j], ha="center", va="center",
                    fontsize=10, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Duyệt (0)", "Từ chối (1)"])
    ax.set_yticklabels(["Good (0)", "Bad (1)"])
    ax.set_xlabel("Dự đoán", fontsize=11)
    ax.set_ylabel("Nhãn thật", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu ma trận nhầm lẫn: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. IN BẢNG KẾT QUẢ RA TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(baseline: dict, sweep: dict, config: dict) -> None:
    """In bảng so sánh Baseline mặc định vs. Baseline tối ưu ngưỡng."""
    cost_fn = config["cost"]["false_negative"]
    cost_fp = config["cost"]["false_positive"]
    max_rej = config["policy"]["max_rejection_rate"]

    bl = baseline["pooled_oof"]
    best = sweep["best_feasible"]

    print("\n" + "=" * 72)
    print("  BẢNG KẾT QUẢ BASELINE – TUẦN 2")
    print(f"  Ma trận chi phí: C_FN = {cost_fn}, C_FP = {cost_fp}")
    print(f"  Ràng buộc: Rejection Rate ≤ {max_rej:.0%}")
    print("=" * 72)

    header = f"{'Chỉ số':<30} {'RF mặc định (τ=0.5)':>20} {'Tối ưu ngưỡng':>20}"
    print(header)
    print("-" * 72)

    def row(label, v1, v2, fmt=".4f"):
        s1 = f"{v1:{fmt}}" if v1 is not None else "N/A"
        s2 = f"{v2:{fmt}}" if v2 is not None else "N/A"
        print(f"  {label:<28} {s1:>20} {s2:>20}")

    best_cost = best["cost_per_sample"] if best else None
    best_rej = best["rejection_rate"] if best else None
    best_fn = best["FN"] if best else None
    best_fp = best["FP"] if best else None
    best_th = best["threshold"] if best else None

    row("Ngưỡng τ", 0.5, best_th, ".2f")
    row("Chi phí / mẫu f(z)", bl["cost_per_sample"], best_cost)
    row("Tổng chi phí (800 mẫu)", bl["total_cost"], best["FN"]*cost_fn + best["FP"]*cost_fp if best else None, ".0f")
    row("False Negative (FN)", bl["FN"], best_fn, "d")
    row("False Positive (FP)", bl["FP"], best_fp, "d")
    row("Rejection Rate r(z)", bl["rejection_rate"], best_rej)
    row("Approval Rate", bl["approval_rate"], 1 - best_rej if best_rej else None)

    print("-" * 72)
    row("Feasible?", "Có" if baseline["feasible"] else "KHÔNG",
        "Có" if (best and best["feasible"]) else "KHÔNG", "s")

    # % tiết kiệm
    if best and bl["cost_per_sample"] > 0:
        saving = (bl["cost_per_sample"] - best_cost) / bl["cost_per_sample"] * 100
        print(f"\n  → Tiết kiệm chi phí so với Baseline: {saving:+.2f}%")
        if saving > 0:
            print(f"    Chỉ cần dịch chuyển ngưỡng từ 0.50 → {best_th:.2f} "
                  f"đã giảm được {saving:.1f}% tổn thất tài chính!")
        print("    (Mức giảm trên OOF trong lưới đã xét; chưa xác nhận trên test.)")
        print("    → Tuần 3: So sánh Random Search và TPE để kiểm tra khả năng cải thiện.")

    # In thêm metric ML
    print("\n  Các chỉ số ML truyền thống (OOF):")
    bl_ml = baseline["ml_metrics"]
    best_ml = sweep["ml_metrics_best"]
    print(f"    {'Metric':<20} {'τ=0.5':>12} {'τ*':>12}")
    for key in ["accuracy", "recall_bad", "precision_bad", "f1_bad", "f1_macro"]:
        v1 = bl_ml.get(key, 0)
        v2 = best_ml.get(key, 0) if best_ml else 0
        print(f"    {key:<20} {v1:>12.4f} {v2:>12.4f}")

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    started = perf_counter()
    config = load_config()
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Bước 1: Nạp dữ liệu Train (chỉ Train, không Test) ──
    print("=" * 60)
    print("  TUẦN 2 – BASELINE EXPERIMENT")
    print("=" * 60)
    print("\n[1/4] Nạp dữ liệu Train và CV folds đã cố định từ Tuần 1...")
    X_train, y_train = load_training_data()
    cv_splits = load_cv_splits(X_train)
    print(f"  Train: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng")
    print(f"  Phân bố nhãn: Good = {(y_train == 0).sum()}, Bad = {(y_train == 1).sum()}")
    print(f"  CV folds: {len(cv_splits)}")

    # ── Bước 2: Chạy Baseline mặc định (τ = 0.5) ──
    print("\n[2/4] Chạy Baseline mặc định: RF default + τ = 0.5 ...")
    baseline_result = run_default_baseline(X_train, y_train, cv_splits, config)
    oof_proba = np.array(baseline_result["oof_proba"])

    # ── Bước 3: Quét ngưỡng tối ưu trên cùng OOF ──
    print("\n[3/4] Quét ngưỡng τ ∈ [0.10, 0.90] trên OOF probabilities...")
    sweep_result = run_threshold_sweep(y_train, oof_proba, config)

    # ── Bước 4: In bảng so sánh & Lưu kết quả ──
    print_summary_table(baseline_result, sweep_result, config)

    # Lưu JSON kết quả
    save_baseline = {k: v for k, v in baseline_result.items() if k != "oof_proba"}
    save_sweep = {k: v for k, v in sweep_result.items()}

    with open(results_dir / "baseline_default.json", "w", encoding="utf-8") as f:
        json.dump(save_baseline, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n  Đã lưu: {results_dir / 'baseline_default.json'}")

    with open(results_dir / "baseline_threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(save_sweep, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"  Đã lưu: {results_dir / 'baseline_threshold_sweep.json'}")

    # Lưu bảng quét ngưỡng dạng CSV
    sweep_df = pd.DataFrame(sweep_result["sweep_table"])
    sweep_df.to_csv(results_dir / "threshold_sweep.csv", index=False)
    fold_ids = np.full(len(y_train), -1, dtype=int)
    for fold_id, (_, val_pos) in enumerate(cv_splits):
        fold_ids[val_pos] = fold_id
    pd.DataFrame({
        "source_row_id": X_train.index,
        "target": y_train.to_numpy(),
        "fold": fold_ids,
        "probability_bad": oof_proba,
    }).to_csv(results_dir / "baseline_oof.csv", index=False)
    print(f"  Đã lưu: {results_dir / 'threshold_sweep.csv'}")

    # ── Vẽ đồ thị ──
    print("\n[Đồ thị]")
    plot_cost_vs_threshold(sweep_result, baseline_result, config, figures_dir)

    # Ma trận nhầm lẫn Baseline mặc định
    oof_pred_default = apply_threshold(oof_proba, 0.5)
    plot_confusion_matrix(
        y_train.values, oof_pred_default,
        "Confusion Matrix – RF Default (τ=0.5, OOF)",
        config, figures_dir, "week2_cm_baseline_default.png"
    )

    # Ma trận nhầm lẫn Baseline tối ưu ngưỡng
    best = sweep_result["best_feasible"]
    if best is not None:
        oof_pred_best = apply_threshold(oof_proba, best["threshold"])
        plot_confusion_matrix(
            y_train.values, oof_pred_best,
            f"Confusion Matrix – RF Default (τ*={best['threshold']:.2f}, OOF)",
            config, figures_dir, "week2_cm_baseline_threshold_opt.png"
        )

    write_json(results_dir / "baseline_run_metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "environment": _environment_versions(),
        "config": config,
        "config_sha256": file_sha256(CONFIG_PATH),
        "input_sha256": {name: file_sha256(DATA_DIR / name) for name in
                         ("credit_train.csv", "cv_folds.json", "dataset_metadata.json")},
        "code_sha256": {name: file_sha256(PROJECT_ROOT / "src" / name) for name in
                        ("baseline.py", "preprocessing.py", "data_utils.py", "config.py")},
        "result_sha256": {name: file_sha256(results_dir / name) for name in
                          ("baseline_default.json", "baseline_threshold_sweep.json",
                           "threshold_sweep.csv", "baseline_oof.csv")},
        "evaluation_scope": "pooled_train_oof; threshold selected on same OOF; test unused",
        "tie_break": "lowest threshold among equal minimum feasible costs on grid",
        "model_params": RandomForestClassifier(random_state=config["model_seed"], n_jobs=1).get_params(),
    })
    print("\n" + "=" * 60)
    print("  TUẦN 2 HOÀN TẤT!")
    print("  Kết quả: results/")
    print("  Đồ thị:  figures/")
    print("  ⚠ Tập Test 200 mẫu KHÔNG được sử dụng.")
    print("=" * 60)


if __name__ == "__main__":
    main()
