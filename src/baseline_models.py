import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from data_pipeline import clean_and_interpolate, create_features

# Suppress sklearn warnings for clean output
import warnings
warnings.filterwarnings('ignore')

def train_and_compare():
    train_path = r'c:\kaggle_load_solar_wind_forecast\data\train.csv'
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
        'month', 'tod', 'week_number', 'temperature', 'nebulosity', 'wind',
        'day_type_week', 'day_type_jf', 'day_type_week_jf',
        'period_holiday', 'period_christmas', 'period_summer',
        'nebulosity_by_solar_power_weights', 'wind_by_wind_power_weights',
        'tod_sin', 'tod_cos', 'month_sin', 'month_cos',
        'temperature_lag_1d', 'temperature_lag_1w',
        'nebulosity_lag_1d', 'nebulosity_lag_1w',
        'wind_lag_1d', 'wind_lag_1w',
        'temperature_rolling_mean_6h', 'temperature_rolling_std_6h',
        'wind_rolling_mean_6h', 'wind_rolling_std_6h',
        'temp_x_hour',
        'wind_sq', 'wind_cube'
    ]
    
    # ==========================================
    # Approach 1: Direct Forecasting
    # ==========================================
    print("\n--- Training Approach 1 (Direct) ---")
    target_direct = 'Electricity_balance_not_controllable'
    
    X_train_dir = train_df[base_features]
    y_train_dir = train_df[target_direct]
    
    X_val_dir = val_df[base_features]
    y_val_dir = val_df[target_direct]
    
    model_direct = HistGradientBoostingRegressor(random_state=42, max_iter=100)
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
    load_features = ['temperature', 'tod', 'month', 'day_type_week', 'day_type_jf', 'period_holiday']
    model_load = HistGradientBoostingRegressor(random_state=42, max_iter=100)
    model_load.fit(train_df[load_features], train_df['Load'])
    preds_load = model_load.predict(val_df[load_features])
    
    # Sub-model 2: Solar (depends heavily on nebulosity, tod)
    solar_features = ['nebulosity', 'nebulosity_by_solar_power_weights', 'tod', 'month']
    model_solar = HistGradientBoostingRegressor(random_state=42, max_iter=100)
    model_solar.fit(train_df[solar_features], train_df['Solar_power'])
    preds_solar = model_solar.predict(val_df[solar_features])
    
    # Sub-model 3: Wind (depends heavily on wind and its polynomials)
    wind_features = ['wind', 'wind_by_wind_power_weights', 'tod', 'month', 'wind_sq', 'wind_cube']
    model_wind = LinearRegression()
    model_wind.fit(train_df[wind_features], train_df['Wind_power'])
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

if __name__ == "__main__":
    train_and_compare()
