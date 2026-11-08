"""Tables and plots for repeated searches and one-factor sensitivity experiments."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baseline import apply_threshold, compute_cost_metrics

METHODS = ("random", "tpe")
METRICS = ("cost_per_sample", "rejection_rate", "approval_rate", "recall_bad",
           "FN", "FP", "f1_macro", "search_seconds", "feasible_trial_fraction")


def per_run_row(summary, scenario="main"):
    best = summary["best_feasible"]
    return {"scenario": scenario, "method": summary["method"],
            "sampler_seed": summary["sampler_seed"], "feasible": best is not None,
            "completed_trials": summary["completed_trials"],
            "completed_fit_count": summary["completed_fit_count"],
            "search_seconds": summary["elapsed_seconds"],
            "feasible_trial_fraction": summary["feasible_trials"] / summary["completed_trials"],
            **(best["metrics"] if best else {}),
            **(best["params"] if best else {}),
            "trial_number": best["trial_number"] if best else None}


def aggregate_runs(frame):
    """Sample SD across seeds, not across folds/trials. Never hide unsuccessful seeds."""
    rows = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method, "n_seeds": len(group),
               "n_feasible_seeds": int(group["feasible"].sum()),
               "success_rate": float(group["feasible"].mean())}
        for metric in METRICS:
            values = pd.to_numeric(group.reindex(columns=[metric])[metric], errors="coerce").dropna()
            row[metric + "_n"] = len(values)
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) >= 2 else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(frame, seeds):
    rows = []
    for seed in seeds:
        pair = frame[frame.sampler_seed == seed].set_index("method")
        comparable = all(bool(pair.loc[m, "feasible"]) for m in METHODS)
        delta = float(pair.loc["tpe", "cost_per_sample"] - pair.loc["random", "cost_per_sample"]) if comparable else None
        rows.append({"sampler_seed": seed, "both_feasible": comparable,
                     "tpe_minus_random_cost": delta,
                     "lower_cost_method": ("tie" if abs(delta) < 1e-12 else
                                           "tpe" if delta < 0 else "random") if comparable else "not_comparable"})
    return pd.DataFrame(rows)


def baseline_scenario(y, proba, config, scenario):
    """Re-score and re-select threshold using fixed RF baseline OOF (no RF refit)."""
    rows = []
    for threshold in np.arange(10, 91) / 100.:
        metrics = compute_cost_metrics(y, apply_threshold(proba, threshold),
                                       config["cost"]["false_negative"], config["cost"]["false_positive"])
        metrics["recall_bad"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
        rows.append({"threshold": float(threshold), **metrics,
                     "feasible": metrics["rejection_rate"] <= config["policy"]["max_rejection_rate"]})
    default = next(r for r in rows if r["threshold"] == .5)
    feasible = [r for r in rows if r["feasible"]]
    best = min(feasible, key=lambda r: (r["cost_per_sample"], r["threshold"])) if feasible else {"feasible": False}
    return [{"scenario": scenario, "method": "baseline_default", **default},
            {"scenario": scenario, "method": "baseline_threshold", **best}]


def convergence_table(histories, seeds):
    rows = []
    for method in METHODS:
        paths = np.array([[np.nan if r["best_feasible_cost_so_far"] is None else r["best_feasible_cost_so_far"]
                           for r in histories[method, seed]] for seed in seeds], dtype=float)
        for i in range(paths.shape[1]):
            values = paths[:, i]
            valid = np.isfinite(values)
            # Do not compare a mean over different subsets of lucky seeds.
            complete = valid.all()
            rows.append({"method": method, "evaluation_number": i + 1,
                         "n_feasible_seeds": int(valid.sum()), "n_seeds": len(seeds),
                         "mean_best_cost": float(values.mean()) if complete else None,
                         "std_best_cost": float(values.std(ddof=1)) if complete and len(seeds) > 1 else None})
    return pd.DataFrame(rows)


def plot_results(frame, convergence, sensitivity, baselines, folder):
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, method in enumerate(METHODS):
        group = frame[frame.method == method]
        values = pd.to_numeric(group.reindex(columns=["cost_per_sample"])["cost_per_sample"], errors="coerce").dropna()
        if len(values):
            axes[0].boxplot([values], positions=[i + 1], widths=.4)
            axes[0].scatter(np.linspace(i + .9, i + 1.1, len(values)), values, zorder=3)
        axes[1].scatter(np.repeat(i + 1, len(group)), group.search_seconds)
    for b in baselines:
        if b["feasible"]:
            axes[0].axhline(b["cost_per_sample"], linestyle="--", label=b["method"], alpha=.7)
    axes[0].set_ylabel("Best feasible OOF cost / sample")
    axes[0].set_title("Seed results (missing = no feasible solution)")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Search wall time (seconds, includes logging)")
    axes[1].set_title("Runtime by seed; same machine recommended")
    for ax in axes:
        ax.set_xticks([1, 2], METHODS)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "seed_stability_runtime.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for method in METHODS:
        g = convergence[convergence.method == method]
        x = g.evaluation_number.to_numpy()
        mean = pd.to_numeric(g.mean_best_cost).to_numpy(dtype=float)
        sd = pd.to_numeric(g.std_best_cost).to_numpy(dtype=float)
        axes[0].plot(x, mean, label=method)
        axes[0].fill_between(x, mean - sd, mean + sd, alpha=.15)
        axes[1].step(x, g.n_feasible_seeds / g.n_seeds, where="post", label=method)
    axes[0].set(title="Mean ± sample SD across seeds (only when all seeds feasible)", ylabel="Best feasible OOF cost")
    axes[1].set(xlabel="Trial evaluations per seed", ylabel="Fraction of seeds with a feasible trial", ylim=(-.05, 1.05))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "convergence_across_seeds.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scenarios, column, label in (
        (axes[0], ["fn2", "main", "fn10"], "cost_fn", "False-negative cost (FP cost = 1)"),
        (axes[1], ["rejection25", "main", "rejection45"], "max_rejection_rate", "Maximum rejection rate"),
    ):
        for method in ("baseline_default", "baseline_threshold", *METHODS):
            g = sensitivity[(sensitivity.scenario.isin(scenarios)) & (sensitivity.method == method)].sort_values(column)
            cost = pd.to_numeric(g.reindex(columns=["cost_per_sample"])["cost_per_sample"], errors="coerce").where(g.feasible)
            ax.plot(g[column], cost, "o-", label=method)
        ax.set(xlabel=label, ylabel="Feasible OOF cost / sample")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("One-factor sensitivity: searches rerun per scenario, seed 42 only\nCost values across different cost matrices have different meanings; gaps = infeasible")
    fig.tight_layout()
    fig.savefig(folder / "sensitivity.png", dpi=150)
    plt.close(fig)
