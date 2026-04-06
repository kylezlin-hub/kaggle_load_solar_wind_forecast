import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_pipeline import clean_and_interpolate, create_features

# Suppress sklearn warnings for clean output
import warnings

warnings.filterwarnings("ignore")


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

    raise ValueError("model_type must be one of: 'lgbm', 'xgb'")


def train_and_compare(model_type: str = "lgbm"):
    train_path = r"c:\kaggle_load_solar_wind_forecast\data\train.csv"
    print(f"Using model backend: {model_type.upper()}")
    print("Loading and cleaning data...")
    df = pd.read_csv(train_path)
    df = clean_and_interpolate(df, is_train=True)

    print("Engineering advanced time-series features...")
    df = create_features(df)

    # We will use the last 20% of data for validation (time-series split)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Common features
    base_features = [
        "month",
        "tod",
        "week_number",
        "temperature",
        "nebulosity",
        "wind",
        "day_type_week",
        "day_type_jf",
        "day_type_week_jf",
        "period_holiday",
        "period_christmas",
        "period_summer",
        "nebulosity_by_solar_power_weights",
        "wind_by_wind_power_weights",
        "tod_sin",
        "tod_cos",
        "month_sin",
        "month_cos",
        "temperature_lag_1d",
        "temperature_lag_1w",
        "nebulosity_lag_1d",
        "nebulosity_lag_1w",
        "wind_lag_1d",
        "wind_lag_1w",
        "temperature_rolling_mean_6h",
        "temperature_rolling_std_6h",
        "wind_rolling_mean_6h",
        "wind_rolling_std_6h",
        "temp_x_hour",
        "wind_sq",
        "wind_cube",
    ]

    # ==========================================
    # Approach 1: Direct Forecasting
    # ==========================================
    print("\n--- Training Approach 1 (Direct) ---")
    target_direct = "Electricity_balance_not_controllable"

    X_train_dir = train_df[base_features]
    y_train_dir = train_df[target_direct]

    X_val_dir = val_df[base_features]
    y_val_dir = val_df[target_direct]

    model_direct = build_model(model_type, random_state=42)
    model_direct.fit(X_train_dir, y_train_dir)

    preds_direct = model_direct.predict(X_val_dir)
    mae_direct = mean_absolute_error(y_val_dir, preds_direct)
    rmse_direct = np.sqrt(mean_squared_error(y_val_dir, preds_direct))
    print(f"[Direct] MAE: {mae_direct:.2f}")
    print(f"[Direct] RMSE: {rmse_direct:.2f}")

    # ==========================================
    # Approach 2: Component Forecasting
    # ==========================================
    print("\n--- Training Approach 2 (Component) ---")

    # Sub-model 1: Load (depends heavily on temp, tod, day type)
    """     load_features = [
        "temperature",
        "tod",
        "month",
        "day_type_week",
        "day_type_jf",
        "period_holiday",
        "tod_sin",
        "tod_cos",
        "month_sin",
        "month_cos",
        "temperature_lag_1d",
        "temperature_lag_1w",
    ] """
    load_features = base_features  # Using all features for load as well
    model_load = build_model(model_type, random_state=42)
    model_load.fit(train_df[load_features], train_df["Load"])
    preds_load = model_load.predict(val_df[load_features])

    # Sub-model 2: Solar (depends heavily on nebulosity, tod)
    solar_features = ["nebulosity", "nebulosity_by_solar_power_weights", "tod", "month"]
    model_solar = build_model(model_type, random_state=42)
    model_solar.fit(train_df[solar_features], train_df["Solar_power"])
    preds_solar = model_solar.predict(val_df[solar_features])

    # Sub-model 3: Wind (depends heavily on wind and its polynomials)
    wind_features = [
        "wind",
        "wind_by_wind_power_weights",
        "tod",
        "month",
        "wind_sq",
        "wind_cube",
    ]
    model_wind = build_model(model_type, random_state=42)
    model_wind.fit(train_df[wind_features], train_df["Wind_power"])
    preds_wind = model_wind.predict(val_df[wind_features])

    # Combine predictions
    preds_component = preds_load - preds_solar - preds_wind

    mae_comp = mean_absolute_error(y_val_dir, preds_component)
    rmse_comp = np.sqrt(mean_squared_error(y_val_dir, preds_component))
    print(f"[Component] MAE: {mae_comp:.2f}")
    print(f"[Component] RMSE: {rmse_comp:.2f}")

    print("\n--- Conclusion ---")
    if mae_comp < mae_direct:
        print(f"Approach 2 (Component) is BETTER by {mae_direct - mae_comp:.2f} MAE.")
    else:
        print(f"Approach 1 (Direct) is BETTER by {mae_comp - mae_direct:.2f} MAE.")


# choices=["lgbm", "xgb"],
train_and_compare(model_type="lgbm")
