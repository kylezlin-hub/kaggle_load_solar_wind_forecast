from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import clean_and_interpolate, create_features, get_base_features


def build_model(model_type: str, random_state: int = 42):
    """Return a tree-based regressor based on selected backend."""
    model_type = model_type.lower()

    if model_type == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=random_state, max_iter=200)

    raise ValueError("model_type must be one of: 'lgbm', 'xgb', 'rf', 'hgb'")


def evaluate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def split_by_date(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    split_ts = pd.Timestamp(split_date)
    date_tz = df["date"].dt.tz
    if date_tz is not None and split_ts.tzinfo is None:
        split_ts = split_ts.tz_localize(date_tz)

    train_df = df[df["date"] < split_ts].copy()
    val_df = df[df["date"] >= split_ts].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"Invalid split_date={split_date}. Train/val split produced an empty partition.")

    return train_df, val_df, split_ts


def evaluate_split(df: pd.DataFrame, model_type: str, split_date: str) -> dict[str, float]:
    base_features = get_base_features()
    train_df, val_df, split_ts = split_by_date(df, split_date)

    print(f"\nSplit date: {split_ts}")
    print(f"Train size: {len(train_df):,}, Val size: {len(val_df):,}")

    target_col = "Electricity_balance_not_controllable"
    y_val = val_df[target_col]

    # Approach 1: Direct forecasting
    model_direct = build_model(model_type, random_state=42)
    model_direct.fit(train_df[base_features], train_df[target_col])
    preds_direct = model_direct.predict(val_df[base_features])
    direct_metrics = evaluate_metrics(y_val, preds_direct)

    # Approach 2: Component-wise forecasting
    load_features = base_features
    solar_features = ["nebulosity", "nebulosity_by_solar_power_weights", "tod", "month"]
    wind_features = ["wind", "wind_by_wind_power_weights", "tod", "month", "wind_sq", "wind_cube"]

    model_load = build_model(model_type, random_state=42)
    model_load.fit(train_df[load_features], train_df["Load"])
    preds_load = model_load.predict(val_df[load_features])

    model_solar = build_model(model_type, random_state=42)
    model_solar.fit(train_df[solar_features], train_df["Solar_power"])
    preds_solar = model_solar.predict(val_df[solar_features])

    model_wind = build_model(model_type, random_state=42)
    model_wind.fit(train_df[wind_features], train_df["Wind_power"])
    preds_wind = model_wind.predict(val_df[wind_features])

    preds_component = preds_load - preds_solar - preds_wind
    comp_metrics = evaluate_metrics(y_val, preds_component)

    print(
        "Direct     -> "
        f"MAE: {direct_metrics['mae']:.2f}, RMSE: {direct_metrics['rmse']:.2f}, MAPE: {direct_metrics['mape']:.2f}%"
    )
    print(
        "Component  -> "
        f"MAE: {comp_metrics['mae']:.2f}, RMSE: {comp_metrics['rmse']:.2f}, MAPE: {comp_metrics['mape']:.2f}%"
    )

    winner = "component" if comp_metrics["mae"] < direct_metrics["mae"] else "direct"
    print(f"Winner on MAE: {winner}")

    return {
        "split_date": split_ts,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "direct_mae": direct_metrics["mae"],
        "direct_rmse": direct_metrics["rmse"],
        "direct_mape": direct_metrics["mape"],
        "comp_mae": comp_metrics["mae"],
        "comp_rmse": comp_metrics["rmse"],
        "comp_mape": comp_metrics["mape"],
    }


def build_rolling_split_dates(df: pd.DataFrame, val_months: int, n_splits: int) -> list[pd.Timestamp]:
    max_date = df["date"].max()
    min_valid_split = df["date"].min() + pd.DateOffset(months=12)
    split_dates: list[pd.Timestamp] = []

    for offset in range(n_splits - 1, -1, -1):
        split_date = max_date - pd.DateOffset(months=val_months * (offset + 1))
        if split_date > min_valid_split:
            split_dates.append(split_date)

    return split_dates


def to_log_rows(result: dict[str, Any], model_type: str, scope: str, run_id: str, run_ts_utc: str) -> list[dict[str, Any]]:
    split_date = pd.Timestamp(result["split_date"]).strftime("%Y-%m-%d")
    winner = "component" if result["comp_mae"] < result["direct_mae"] else "direct"

    common = {
        "run_id": run_id,
        "run_ts_utc": run_ts_utc,
        "scope": scope,
        "split_date": split_date,
        "model_type": model_type,
        "train_size": int(result["train_size"]),
        "val_size": int(result["val_size"]),
        "winner_on_mae": winner,
    }

    return [
        {
            **common,
            "approach": "direct",
            "mae": float(result["direct_mae"]),
            "rmse": float(result["direct_rmse"]),
            "mape": float(result["direct_mape"]),
            "is_winner": winner == "direct",
        },
        {
            **common,
            "approach": "component",
            "mae": float(result["comp_mae"]),
            "rmse": float(result["comp_rmse"]),
            "mape": float(result["comp_mape"]),
            "is_winner": winner == "component",
        },
    ]


def append_experiment_log(rows: list[dict[str, Any]], log_path: str) -> None:
    if not rows:
        return

    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    log_df = pd.DataFrame(rows)
    file_exists = out.exists()
    log_df.to_csv(out, mode="a", index=False, header=not file_exists)
    print(f"Experiment log appended: {out}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare direct vs component-wise forecasting with consistent time-based validation."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(repo_root / "data" / "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["lgbm", "xgb", "rf", "hgb"],
        help="Tree model backend",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2020-07-01",
        help="Validation split date for single holdout run",
    )
    parser.add_argument(
        "--rolling-splits",
        type=int,
        default=0,
        help="If > 0, run this many rolling time splits using val-months",
    )
    parser.add_argument(
        "--val-months",
        type=int,
        default=6,
        help="Validation window length in months for rolling splits",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(repo_root / "docs" / "experiment_log.csv"),
        help="Path to append experiment metrics log CSV",
    )
    parser.add_argument(
        "--disable-log",
        action="store_true",
        help="Disable writing experiment metrics to CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    run_ts_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    log_rows: list[dict[str, Any]] = []

    print(f"Using model backend: {args.model_type.upper()}")
    print("Loading and cleaning data...")

    df = pd.read_csv(args.train_path)
    df = clean_and_interpolate(df)

    print("Engineering advanced time-series features...")
    df = create_features(df)

    print("\n--- Single Holdout Evaluation ---")
    holdout_result = evaluate_split(df, model_type=args.model_type, split_date=args.split_date)
    log_rows.extend(
        to_log_rows(
            result=holdout_result,
            model_type=args.model_type,
            scope="single_holdout",
            run_id=run_id,
            run_ts_utc=run_ts_utc,
        )
    )

    if args.rolling_splits > 0:
        print("\n--- Rolling Backtest Evaluation ---")
        split_dates = build_rolling_split_dates(df, val_months=args.val_months, n_splits=args.rolling_splits)
        if not split_dates:
            print("No valid rolling split dates could be generated. Try a smaller --rolling-splits value.")
            return

        rows = []
        for split_ts in split_dates:
            result = evaluate_split(df, model_type=args.model_type, split_date=str(split_ts.date()))
            rows.append(result)
            log_rows.extend(
                to_log_rows(
                    result=result,
                    model_type=args.model_type,
                    scope="rolling",
                    run_id=run_id,
                    run_ts_utc=run_ts_utc,
                )
            )

        bt = pd.DataFrame(rows)
        print("\nRolling summary (mean across splits):")
        print(
            bt[["direct_mae", "comp_mae", "direct_rmse", "comp_rmse", "direct_mape", "comp_mape"]]
            .mean()
            .round(2)
            .to_string()
        )

    if not args.disable_log:
        append_experiment_log(log_rows, log_path=args.log_path)


if __name__ == "__main__":
    main()
