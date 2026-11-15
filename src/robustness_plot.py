"""Render a readable presentation plot from saved nested-CV scores, without any fitting.

The original experiment artifacts and their checksums remain unchanged.
"""
import argparse
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import write_json
from .config import PROJECT_ROOT
from .data_utils import file_sha256
from .evaluate_final import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/robustness/run-20261109T024250942600Z")
    args = parser.parse_args()
    source = (PROJECT_ROOT / args.run).resolve()
    if not source.is_relative_to((PROJECT_ROOT / "results" / "robustness").resolve()):
        parser.error("Run must be inside results/robustness.")
    metadata = read_json(source / "run_metadata.json")
    if metadata["status"] != "complete" or file_sha256(source / "aggregate.csv") != metadata["output_sha256"]["aggregate.csv"]:
        raise ValueError("A verified complete aggregate is required.")
    data = pd.read_csv(source / "aggregate.csv")
    output = source / "presentation"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    for i, row in data.iterrows():
        if not row.complete_coverage:
            continue
        ax.scatter(row.pooled_rejection_rate, row.pooled_cost, s=90, color=f"C{i}",
                   marker="o" if row.method.startswith("tpe") else "s", label=row.method, zorder=3)
    cap = metadata["identity"]["protocol"]["evaluation_rejection_cap"]
    ax.axvline(cap, color="black", ls="--", label=f"Policy cap {cap:.0%}")
    ax.set(xlabel="Pooled outer rejection rate", ylabel="Pooled outer cost / sample",
           title="Exploratory nested CV: cost versus rejection\nOriginal 800 training rows; original test excluded")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
    ax.grid(alpha=.2)
    ax.margins(x=.15, y=.12)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    path = output / "cost_rejection_tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    write_json(output / "render_metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Legend replaces overlapping annotations; presentation only. No models or predictions recomputed.",
        "source_aggregate_sha256": file_sha256(source / "aggregate.csv"),
        "renderer_sha256": file_sha256(PROJECT_ROOT / "src" / "robustness_plot.py"),
        "image_sha256": file_sha256(path)})
    print(f"Rendered verified saved results: {path}")


if __name__ == "__main__":
    main()
