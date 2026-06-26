from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

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


def train_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, model_type: str) -> pd.Series:
    features = get_base_features()
    target_col = "Electricity_balance_not_controllable"

    model = build_model(model_type, random_state=42)
    model.fit(train_df[features], train_df[target_col])
    preds = model.predict(test_df[features])
    return pd.Series(preds, index=test_df.index)


def train_component(train_df: pd.DataFrame, test_df: pd.DataFrame, model_type: str) -> pd.Series:
    base_features = get_base_features()

    load_features = base_features
    solar_features = ["nebulosity", "nebulosity_by_solar_power_weights", "tod", "month"]
    wind_features = ["wind", "wind_by_wind_power_weights", "tod", "month", "wind_sq", "wind_cube"]

    model_load = build_model(model_type, random_state=42)
    model_load.fit(train_df[load_features], train_df["Load"])
    preds_load = model_load.predict(test_df[load_features])

    model_solar = build_model(model_type, random_state=42)
    model_solar.fit(train_df[solar_features], train_df["Solar_power"])
    preds_solar = model_solar.predict(test_df[solar_features])

    model_wind = build_model(model_type, random_state=42)
    model_wind.fit(train_df[wind_features], train_df["Wind_power"])
    preds_wind = model_wind.predict(test_df[wind_features])

    preds = preds_load - preds_solar - preds_wind
    return pd.Series(preds, index=test_df.index)


def build_submission_log_row(
    preds: pd.Series,
    out_path: str,
    model_type: str,
    approach: str,
    blend_weight_component: float,
) -> dict[str, Any]:
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    run_ts_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "run_id": run_id,
        "run_ts_utc": run_ts_utc,
        "approach": approach,
        "model_type": model_type,
        "blend_weight_component": float(blend_weight_component) if approach == "blend" else None,
        "out_path": str(Path(out_path).resolve()),
        "rows": int(preds.shape[0]),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "pred_min": float(preds.min()),
        "pred_p05": float(preds.quantile(0.05)),
        "pred_median": float(preds.median()),
        "pred_p95": float(preds.quantile(0.95)),
        "pred_max": float(preds.max()),
    }


def append_submission_log(row: dict[str, Any], log_path: str) -> None:
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])
    file_exists = out.exists()
    df.to_csv(out, mode="a", index=False, header=not file_exists)
    print(f"Submission run log appended: {out}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate Kaggle submission with direct/component/blend models.")
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(repo_root / "data" / "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(repo_root / "data" / "test.csv"),
        help="Path to test.csv",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(repo_root / "data" / "submission.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["lgbm", "xgb", "rf", "hgb"],
        help="Tree model backend",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="component",
        choices=["direct", "component", "blend"],
        help="Prediction approach to generate final submission",
    )
    parser.add_argument(
        "--blend-weight-component",
        type=float,
        default=0.7,
        help="Weight for component model in blend mode. Direct weight = 1 - this value.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(repo_root / "docs" / "submission_run_log.csv"),
        help="Path to append submission run log CSV",
    )
    parser.add_argument(
        "--disable-log",
        action="store_true",
        help="Disable writing submission run log",
    )
    return parser.parse_args()


def generate_submission(
    train_path: str,
    test_path: str,
    out_path: str,
    model_type: str,
    approach: str,
    blend_weight_component: float,
) -> pd.Series:
    print("Loading data...")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    test_ids = test_raw["Id"].copy()

    print("Cleaning and interpolating...")
    train_clean = clean_and_interpolate(train_raw)
    test_clean = clean_and_interpolate(test_raw)

    print("Engineering advanced time-series features...")
    combined_df = pd.concat([train_clean, test_clean], ignore_index=True)
    combined_df = combined_df.sort_values("date")
    combined_df = create_features(combined_df)

    is_test_mask = combined_df["Id"].isin(test_ids)
    train_features_df = combined_df[~is_test_mask].copy()
    test_features_df = combined_df[is_test_mask].copy()

    print(f"Training approach: {approach} | backend: {model_type.upper()}")
    if approach == "direct":
        preds = train_direct(train_features_df, test_features_df, model_type=model_type)
    elif approach == "component":
        preds = train_component(train_features_df, test_features_df, model_type=model_type)
    else:
        comp_w = float(blend_weight_component)
        if not 0.0 <= comp_w <= 1.0:
            raise ValueError("blend_weight_component must be between 0 and 1")
        direct_preds = train_direct(train_features_df, test_features_df, model_type=model_type)
        component_preds = train_component(train_features_df, test_features_df, model_type=model_type)
        preds = comp_w * component_preds + (1.0 - comp_w) * direct_preds

    submission = pd.DataFrame(
        {
            "Id": test_features_df["Id"].astype(int),
            "Predicted": preds,
        }
    ).sort_values("Id")

    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}")
    return preds


def main() -> None:
    args = parse_args()
    preds = generate_submission(
        train_path=args.train_path,
        test_path=args.test_path,
        out_path=args.out_path,
        model_type=args.model_type,
        approach=args.approach,
        blend_weight_component=args.blend_weight_component,
    )

    if not args.disable_log:
        row = build_submission_log_row(
            preds=preds,
            out_path=args.out_path,
            model_type=args.model_type,
            approach=args.approach,
            blend_weight_component=args.blend_weight_component,
        )
        append_submission_log(row=row, log_path=args.log_path)


if __name__ == "__main__":
    main()
