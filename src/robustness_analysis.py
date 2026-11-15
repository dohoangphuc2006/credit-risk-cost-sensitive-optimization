"""Aggregate outer predictions only; never use outer scores for model selection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate_final import score_predictions


def summarize(rows, predictions, expected_ids, config, protocol):
    folds = protocol["outer_folds"]
    frame = pd.DataFrame(rows)
    output = []
    for method, group in frame.groupby("method", sort=False):
        if len(group) != folds or set(group.outer_fold) != set(range(folds)):
            raise ValueError("Expected exactly one result per outer fold and method.")
        selected = group[group.selection_status == "selected"]
        pred = predictions[predictions.method == method]
        if pred.source_row_id.duplicated().any() or not set(pred.source_row_id).issubset(expected_ids):
            raise ValueError("Outer predictions must be unique original training rows.")
        for _, row in group.iterrows():
            part = pred[pred.outer_fold == row.outer_fold]
            expected_n = int(row.n_outer) if row.selection_status == "selected" else 0
            if len(part) != expected_n:
                raise ValueError("Outer predictions do not match fold selection status/count.")
        complete = len(selected) == folds
        result = {"method": method, "n_selected_folds": len(selected), "n_outer_folds": folds,
                  "n_predicted_rows": len(pred), "complete_coverage": complete,
                  "outer_folds_policy_passed": int(selected.feasible.sum()) if len(selected) else 0,
                  "cost_fold_mean": float(selected.cost_per_sample.mean()) if len(selected) else None,
                  "cost_fold_std": float(selected.cost_per_sample.std(ddof=1)) if len(selected) > 1 else None,
                  "rejection_fold_std": float(selected.rejection_rate.std(ddof=1)) if len(selected) > 1 else None,
                  "pooled_cost": None, "pooled_rejection_rate": None, "pooled_feasible": None}
        if complete:
            if set(pred.source_row_id) != set(expected_ids):
                raise ValueError("Complete outer predictions must cover every training row exactly once.")
            m = score_predictions(pred.target.to_numpy(), pred.prediction.to_numpy(), config)
            result.update(pooled_cost=m["cost_per_sample"], pooled_rejection_rate=m["rejection_rate"],
                          pooled_feasible=m["feasible"], pooled_FN=m["FN"], pooled_FP=m["FP"],
                          pooled_recall_bad=m["recall_bad"], pooled_accuracy=m["accuracy"])
        output.append(result)
    return pd.DataFrame(output)


def plot_results(aggregate, per_fold, output):
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for _, row in aggregate.iterrows():
        if not row.complete_coverage:
            continue
        tpe = row.method.startswith("tpe")
        ax.scatter(row.pooled_rejection_rate, row.pooled_cost, s=85, marker="o" if tpe else "s",
                   color="#cc7832" if tpe else "#337baa", label=row.method)
        ax.annotate(row.method, (row.pooled_rejection_rate, row.pooled_cost),
                    xytext=(7, 9 if tpe else -17), textcoords="offset points", fontsize=9)
    ax.axvline(.35, color="#b52b2b", ls="--", label="Original policy cap 35%")
    ax.set(xlabel="Pooled outer rejection rate", ylabel="Pooled outer cost / sample",
           title="Exploratory nested CV: cost versus rejection\nOriginal 800 training rows; original test excluded")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
    ax.margins(x=.3, y=.2)
    ax.grid(alpha=.2)
    fig.savefig(output / "cost_rejection_tradeoff.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for method, group in per_fold.groupby("method", sort=False):
        group = group.sort_values("outer_fold")
        ax.plot(group.outer_fold + 1, group.rejection_rate, marker="o", label=method,
                linestyle="-" if method.startswith("tpe") else "--", alpha=.85)
    ax.axhline(.35, color="black", ls=":", label="Policy cap 35%")
    ax.set(xlabel="Outer fold (not sampler seed)", ylabel="Rejection rate",
           title="Policy satisfaction varies across held-out outer folds", xticks=range(1, 6))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(alpha=.2)
    fig.savefig(output / "outer_fold_rejection.png", dpi=160)
    plt.close(fig)
