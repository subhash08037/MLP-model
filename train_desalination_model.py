#!/usr/bin/env python3
"""Train a deep learning model for desalination efficiency prediction.

This script reads process parameters from an Excel file, splits the data into
70% training / 15% validation / 15% test sets, trains a feed-forward neural
network, and reports final metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplits:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep learning for desalination efficiency prediction from Excel data."
    )
    parser.add_argument("--excel", required=True, help="Path to input Excel file (.xlsx/.xls).")
    parser.add_argument(
        "--target",
        default="desalination_efficiency",
        help="Name of target column to predict (default: desalination_efficiency).",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name/index (default: first sheet).",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--model-out",
        default="desalination_efficiency_model.keras",
        help="Output path for saved Keras model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_and_prepare_data(excel_path: Path, target_col: str, sheet: str | int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(excel_path, sheet_name=sheet)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    feature_df = df.drop(columns=[target_col])
    y = df[target_col].to_numpy(dtype=np.float32)

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found in the Excel file.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
    )

    x = preprocessor.fit_transform(feature_df)
    return x.astype(np.float32), y


def split_data(x: np.ndarray, y: np.ndarray, seed: int) -> DataSplits:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return DataSplits(x_train, x_val, x_test, y_train, y_val, y_test)


def build_model(input_dim: int, learning_rate: float, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{name} metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")


def main() -> None:
    args = parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    x, y = load_and_prepare_data(excel_path, args.target, args.sheet)
    splits = split_data(x, y, args.seed)

    print("Dataset shapes:")
    print(f"  Train: {splits.x_train.shape} ({len(splits.y_train)} rows)")
    print(f"  Val:   {splits.x_val.shape} ({len(splits.y_val)} rows)")
    print(f"  Test:  {splits.x_test.shape} ({len(splits.y_test)} rows)")

    model = build_model(input_dim=splits.x_train.shape[1], learning_rate=args.learning_rate, seed=args.seed)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
        )
    ]

    model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(splits.x_val, splits.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    train_pred = model.predict(splits.x_train, verbose=0).reshape(-1)
    val_pred = model.predict(splits.x_val, verbose=0).reshape(-1)
    test_pred = model.predict(splits.x_test, verbose=0).reshape(-1)

    evaluate_split("Train", splits.y_train, train_pred)
    evaluate_split("Validation", splits.y_val, val_pred)
    evaluate_split("Test", splits.y_test, test_pred)

    model.save(args.model_out)
    print(f"\nSaved trained model to: {args.model_out}")


if __name__ == "__main__":
    main()
