from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARTIFACT_FORMAT_VERSION = 1
DEFAULT_MODEL_PATH = Path("models/house_price_model.joblib")


def _clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The input dataset has no rows.")
    if not frame.columns.is_unique:
        raise ValueError("The input dataset has duplicate column names.")
    cleaned = frame.copy()
    numeric = cleaned.select_dtypes(include=["number"]).columns
    if len(numeric):
        cleaned.loc[:, numeric] = cleaned.loc[:, numeric].replace(
            [np.inf, -np.inf], np.nan
        )
    empty = cleaned.columns[cleaned.isna().all()].tolist()
    if empty:
        raise ValueError("Feature columns have no usable values: " + ", ".join(empty))
    return cleaned


def build_pipeline(features: pd.DataFrame, alpha: float = 1.0) -> Pipeline:
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero.")
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    if not numeric and not categorical:
        raise ValueError("At least one feature column is required.")

    steps = []
    if numeric:
        steps.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        steps.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(steps, remainder="drop")),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def load_training_data(
    data_path: str | None, dataset: str | None, target_name: str | None
) -> tuple[pd.DataFrame, str, str]:
    if dataset == "california":
        bunch = fetch_california_housing(as_frame=True)
        return bunch.frame.copy(), bunch.target.name, "100,000 USD units"
    if not data_path:
        raise ValueError("Provide either --data or --dataset california.")
    if not target_name:
        raise ValueError("--target is required when training from CSV.")
    return pd.read_csv(data_path), target_name, "user-defined units"


def train_model(
    frame: pd.DataFrame,
    target_name: str,
    target_units: str,
    model_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 1.0,
) -> dict[str, Any]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required.")
    if not frame.columns.is_unique:
        raise ValueError("The dataset has duplicate column names.")
    if target_name not in frame.columns:
        raise ValueError(f"Target column {target_name!r} is missing.")

    target = pd.to_numeric(frame[target_name], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("The target must contain finite numeric values.")
    features = _clean_features(frame.drop(columns=[target_name]))
    if features.shape[1] == 0:
        raise ValueError("The dataset must include feature columns.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = build_pipeline(x_train, alpha)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": float(r2_score(y_test, predicted)),
    }

    # Report holdout metrics, then train the saved model on all available rows.
    model.fit(features, target)
    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model": model,
        "feature_columns": features.columns.tolist(),
        "target_name": target_name,
        "target_units": target_units,
        "metrics": metrics,
        "training_rows": int(len(features)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination, compress=3)
    return {
        "model_path": str(destination),
        "target_name": target_name,
        "target_units": target_units,
        "training_rows": int(len(features)),
        "holdout_metrics": metrics,
    }


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {source}. Train a model first."
        )
    artifact = joblib.load(source)
    if not isinstance(artifact, dict):
        raise ValueError("Unsupported model artifact.")
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")
    required = {"model", "feature_columns", "target_name", "target_units"}
    if not required.issubset(artifact):
        raise ValueError("Model artifact is missing required metadata.")
    return artifact


def predict_frame(artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    columns = artifact["feature_columns"]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The prediction input has no rows.")
    features = _clean_features(frame.loc[:, columns])
    return artifact["model"].predict(features)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and serve a house-price model.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train and save a regression model.")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="CSV training file.")
    source.add_argument("--dataset", choices=["california"])
    train.add_argument("--target", help="Numeric target column for CSV training.")
    train.add_argument("--target-units", default="user-defined units")
    train.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--alpha", type=float, default=1.0)

    predict = commands.add_parser("predict", help="Predict prices from a CSV file.")
    predict.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    predict.add_argument("--data", required=True, help="CSV with feature columns.")
    predict.add_argument("--output", default="predictions.csv")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if args.data and not args.target:
            raise SystemExit("--target is required when training from CSV.")
        frame, target, units = load_training_data(
            args.data, args.dataset, args.target
        )
        if args.data:
            units = args.target_units
        result = train_model(
            frame,
            target,
            units,
            args.model_out,
            args.test_size,
            args.random_state,
            args.alpha,
        )
        print(json.dumps(result, indent=2))
        return

    artifact = load_artifact(args.model)
    frame = pd.read_csv(args.data)
    frame = frame.copy()
    frame["predicted_price"] = predict_frame(artifact, frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(
        json.dumps(
            {
                "output_path": str(output),
                "predictions": len(frame),
                "target_units": artifact["target_units"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
