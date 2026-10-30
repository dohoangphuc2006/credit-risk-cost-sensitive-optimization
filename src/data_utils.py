"""Prepare unencoded train/test files and fixed CV assignments for Week 1.

Run from the repository root: python -m src.data_utils --force
No feature transformer or predictive model is fitted by this command.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split

if __package__ in {None, ""}:
    from config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config
else:
    from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config

ARTIFACT_FORMAT_VERSION = 2
OUTPUT_FILES = (
    "credit_train.csv", "credit_test.csv", "cv_folds.json", "dataset_metadata.json"
)


def file_sha256(path: Path) -> str:
    """Compute a checksum without loading an entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _environment_versions() -> dict:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "optuna", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {"python": platform.python_version(), "packages": packages}


def load_and_split_data(
    test_size: float | None = None,
    random_state: int | None = None,
):
    """Fetch OpenML ID 31, validate labels, then split unencoded features.

    Overrides are for exploratory use. The CLI uses the canonical configuration.
    Row indices refer to original source order and are kept in the CSV files.
    """
    config = load_config()
    dataset_config = config["dataset"]
    test_size = config["split"]["test_size"] if test_size is None else test_size
    random_state = config["split"]["seed"] if random_state is None else random_state
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    dataset = fetch_openml(
        data_id=dataset_config["openml_id"],
        as_frame=True,
        parser="auto",
        data_home=str(PROJECT_ROOT / ".cache" / "openml"),
    )
    X = dataset.data.copy().reset_index(drop=True)
    labels = pd.Series(dataset.target).reset_index(drop=True)
    expected_shape = (
        dataset_config["expected_samples"], dataset_config["expected_features"]
    )
    if X.shape != expected_shape:
        raise ValueError(f"Unexpected dataset shape {X.shape}; expected {expected_shape}.")
    if labels.isna().any() or set(labels.astype(str)) != {"good", "bad"}:
        raise ValueError("Expected exactly the original 'good' and 'bad' labels.")
    if X.isna().any().any():
        raise ValueError("Unexpected missing features in the documented dataset.")
    if not X.columns.is_unique or {"target", "source_row_id"}.intersection(X.columns):
        raise ValueError("Feature names must be unique and avoid reserved names.")

    # Map only the target. Fit feature encoders inside CV in subsequent weeks.
    y = labels.map({"good": 0, "bad": 1}).astype("int64").rename("target")
    X.index.name = "source_row_id"
    y.index.name = "source_row_id"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if set(X_train.index).intersection(X_test.index):
        raise ValueError("Train/test source row IDs overlap.")
    if y_train.value_counts().min() < config["cv"]["n_splits"]:
        raise ValueError("Insufficient training examples per class for stratified CV.")

    def summarize(target: pd.Series) -> dict:
        return {
            "n_samples": len(target),
            "n_good": int(target.eq(0).sum()),
            "n_bad": int(target.eq(1).sum()),
        }

    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "Statlog (German Credit Data)",
        "openml_id": dataset_config["openml_id"],
        "openml_version": dataset.details.get("version"),
        "source_url": "https://www.openml.org/d/31",
        "uci_url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "citation": "Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.",
        "doi": "10.24432/C5NC77",
        "license": "CC BY 4.0 (UCI dataset page)",
        "representation": "raw_features_no_fitted_preprocessing",
        "n_total_samples": len(X),
        "n_features_raw": X.shape[1],
        "feature_columns": X.columns.tolist(),
        "categorical_columns": X.select_dtypes(
            include=["category", "object", "string"]
        ).columns.tolist(),
        "numeric_columns": X.select_dtypes(include="number").columns.tolist(),
        "target_mapping": {"good": 0, "bad": 1},
        "class_distribution": summarize(y),
        "train_set": summarize(y_train),
        "test_set": summarize(y_test),
        "split_config": {"test_size": test_size, "seed": random_state, "stratified": True},
        "train_source_row_ids": X_train.index.tolist(),
        "test_source_row_ids": X_test.index.tolist(),
        "experiment_config": config,
        "config_sha256": file_sha256(CONFIG_PATH),
        "environment": _environment_versions(),
        "test_usage": "Final evaluation after method and configuration selection on train CV.",
        "experiment_results": "Not generated by Week 1 data preparation.",
    }
    return X_train, X_test, y_train, y_test, metadata


def prepare_data(force: bool = False) -> dict:
    """Write data artifacts; require explicit replacement of existing files."""
    existing = [name for name in OUTPUT_FILES if (DATA_DIR / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "Data artifacts already exist: " + ", ".join(existing)
            + ". To regenerate legacy files, run python -m src.data_utils --force."
        )
    X_train, X_test, y_train, y_test, metadata = load_and_split_data()
    config = metadata["experiment_config"]
    cv_config = config["cv"]
    splitter = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["seed"],
    )
    folds = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "cv_config": cv_config,
        "index_type": "source_row_id",
        "folds": [
            {"fold": fold, "validation_row_ids": X_train.index[validation].tolist()}
            for fold, (_, validation) in enumerate(splitter.split(X_train, y_train))
        ],
    }
    validation_ids = [
        row_id for fold in folds["folds"] for row_id in fold["validation_row_ids"]
    ]
    if len(validation_ids) != len(set(validation_ids)) or set(validation_ids) != set(X_train.index):
        raise ValueError("Each training row must occur in exactly one validation fold.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, features, target in (
        ("credit_train.csv", X_train, y_train),
        ("credit_test.csv", X_test, y_test),
    ):
        features.assign(target=target).to_csv(
            DATA_DIR / name, index=True, index_label="source_row_id", encoding="utf-8"
        )
    write_json(DATA_DIR / "cv_folds.json", folds)
    metadata["file_sha256"] = {
        name: file_sha256(DATA_DIR / name)
        for name in OUTPUT_FILES if name != "dataset_metadata.json"
    }
    # Metadata is written last; checksum validation detects mixed/partial outputs.
    write_json(DATA_DIR / "dataset_metadata.json", metadata)
    return metadata


def _read_metadata(data_dir: Path) -> dict:
    with (data_dir / "dataset_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Legacy artifacts detected. Run python -m src.data_utils --force first.")
    if metadata["config_sha256"] != file_sha256(CONFIG_PATH):
        raise ValueError("Configuration changed since preparation; regenerate data artifacts.")
    return metadata


def _verify_artifact(path: Path, metadata: dict) -> None:
    if file_sha256(path) != metadata["file_sha256"][path.name]:
        raise ValueError(f"Checksum mismatch: {path.name}. Regenerate data artifacts.")


def load_training_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Read only training CSV and metadata; never read the test CSV."""
    data_dir = Path(data_dir)
    metadata = _read_metadata(data_dir)
    path = data_dir / "credit_train.csv"
    _verify_artifact(path, metadata)
    frame = pd.read_csv(path, index_col="source_row_id")
    y = frame.pop("target").astype("int64")
    if frame.index.tolist() != metadata["train_source_row_ids"]:
        raise ValueError("Training row IDs differ from the saved split.")
    if frame.columns.tolist() != metadata["feature_columns"]:
        raise ValueError("Training schema differs from the saved schema.")
    return frame, y


def load_cv_splits(X_train: pd.DataFrame, data_dir: Path = DATA_DIR) -> list:
    """Convert saved source row IDs into positional indices for sklearn CV."""
    data_dir = Path(data_dir)
    metadata = _read_metadata(data_dir)
    if not X_train.index.is_unique or X_train.index.tolist() != metadata["train_source_row_ids"]:
        raise ValueError("Pass the original ordered data from load_training_data().")
    path = data_dir / "cv_folds.json"
    _verify_artifact(path, metadata)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    splits = []
    for fold in payload["folds"]:
        validation_mask = X_train.index.isin(fold["validation_row_ids"])
        splits.append(((~validation_mask).nonzero()[0], validation_mask.nonzero()[0]))
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing data artifacts.")
    args = parser.parse_args()
    try:
        metadata = prepare_data(force=args.force)
    except (FileExistsError, ValueError) as error:
        parser.exit(status=1, message=f"Preparation stopped: {error}\n")
    print(f"Prepared OpenML ID {metadata['openml_id']} in {DATA_DIR}")
    print(f"Train: {metadata['train_set']['n_samples']} rows; test: {metadata['test_set']['n_samples']} rows.")
    print(f"Saved {metadata['experiment_config']['cv']['n_splits']} fixed CV folds and SHA-256 checksums.")
    print("Features are unencoded. No model training or test evaluation has been performed.")


if __name__ == "__main__":
    main()
