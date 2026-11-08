"""Sequential Optuna checkpoints. Load only checkpoints produced locally by this project."""
import base64
import hashlib
import json
import pickle
from time import perf_counter

import optuna
import pandas as pd

from .artifacts import write_csv, write_json
from .data_utils import _environment_versions, file_sha256
from .config import PROJECT_ROOT
from .objective import candidate_key, evaluate_candidate, suggest_params
from .optimize import history_rows, make_sampler


def run_search(method, seed, n_trials, X, y, splits, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "search.checkpoint.json"
    identity = {
        "version": 1, "method": method, "seed": seed, "budget": n_trials, "config": config,
        "environment": _environment_versions(),
        "data": hashlib.sha256(pickle.dumps((X, y, splits))).hexdigest(),
        "code": {name: file_sha256(PROJECT_ROOT / "src" / name) for name in
                 ("resumable_search.py", "objective.py", "optimize.py", "preprocessing.py", "artifacts.py")},
    }
    if path.exists():
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope["identity"] != identity:
            raise ValueError("Checkpoint differs in data, code, config, environment or budget.")
        blob = base64.b64decode(envelope["state"])
        if hashlib.sha256(blob).hexdigest() != envelope["sha256"]:
            raise ValueError("Checkpoint content checksum mismatch.")
        state = pickle.loads(blob)  # Trusted local file only; never load downloaded checkpoints.
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Existing search has no checkpoint; use a new run directory.")
        state = {"study": optuna.create_study(direction="minimize", sampler=make_sampler(method, seed),
                                              pruner=optuna.pruners.NopPruner()),
                 "pending": None, "best": None, "best_oof": None, "elapsed": 0., "sessions": 0}
    state["sessions"] += 1
    study = state["study"]
    tick = perf_counter()

    def checkpoint():
        nonlocal tick
        now = perf_counter()
        state["elapsed"] += now - tick
        tick = now
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(path, {"identity": identity, "sha256": hashlib.sha256(blob).hexdigest(),
                          "state": base64.b64encode(blob).decode("ascii")})

    def export():
        rows = [{"method": method, "sampler_seed": seed, **row} for row in history_rows(study)]
        write_csv(output_dir / "trials.csv", pd.DataFrame(rows))
        write_json(output_dir / "trials.json", {"trials": [
            {**r, "fold_details": t.user_attrs.get("fold_details")} for r, t in zip(rows, study.trials)]})
        if state["best"] is not None:
            write_json(output_dir / "best_feasible.json", state["best"])
            write_csv(output_dir / "best_oof.csv", state["best_oof"])
        complete = sum(r["state"] == "COMPLETE" for r in rows)
        summary = {"method": method, "sampler_seed": seed,
                   "status": "complete" if complete == n_trials else "interrupted_or_failed",
                   "requested_trials": n_trials, "completed_trials": complete,
                   "completed_fit_count": sum(r.get("fit_count") or 0 for r in rows),
                   "feasible_trials": sum(r.get("feasible") is True for r in rows),
                   "elapsed_seconds": state["elapsed"], "checkpoint_sessions": state["sessions"],
                   "runtime_note": "Saved active time; downtime and work since last durable checkpoint excluded.",
                   "best_feasible": state["best"],
                   "selection_status": "feasible_found" if state["best"] else "no feasible solution found within budget"}
        write_json(output_dir / "summary.json", summary)
        return summary, rows

    # Commit before evaluation and after tell, retaining the sampler RNG and pending params.
    checkpoint()
    try:
        while sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) < n_trials:
            if state["pending"] is None:
                trial = study.ask()
                params = suggest_params(trial, config)
                state["pending"] = {"trial_id": trial._trial_id, "params": params}
                checkpoint()
            else:
                trial = optuna.trial.Trial(study, state["pending"]["trial_id"])
                params = state["pending"]["params"]
            result = evaluate_candidate(params, X, y, splits, config)
            oof = result.pop("oof_proba")
            fold_ids = result.pop("fold_ids")
            for key, value in result.items():
                trial.set_user_attr(key, value)
            record = {"method": method, "sampler_seed": seed, "trial_number": trial.number,
                      "params": params, **result}
            if result["feasible"] and (state["best"] is None or candidate_key(record) < candidate_key(state["best"])):
                state["best"] = record
                state["best_oof"] = pd.DataFrame({"source_row_id": X.index, "target": y.to_numpy(),
                    "fold": fold_ids, "probability_bad": oof, "prediction": (oof >= params["threshold"]).astype(int)})
            study.tell(trial, result["objective"])
            state["pending"] = None
            checkpoint()
            export()
            print(f"{method} seed={seed} trial={trial.number + 1}/{n_trials} "
                  f"cost={result['metrics']['cost_per_sample']} feasible={result['feasible']}", flush=True)
    except BaseException:
        # The last durable state is authoritative. Do not overwrite it with a partial trial.
        raise
    return export()
