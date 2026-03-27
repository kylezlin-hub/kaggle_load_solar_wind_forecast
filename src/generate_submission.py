import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from data_pipeline import clean_and_interpolate, create_features

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def generate_submission():
    train_path = r'c:\kaggle_load_solar_wind_forecast\data\train.csv'
    test_path = r'c:\kaggle_load_solar_wind_forecast\data\test.csv'
    out_path = r'c:\kaggle_load_solar_wind_forecast\data\submission.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_ids = test_df['Id'].copy()
    
    print("Cleaning and interpolating...")
    train_clean = clean_and_interpolate(train_df, is_train=True)
    test_clean = clean_and_interpolate(test_df, is_train=False)
    
    print("Engineering advanced time-series features...")
    # Concatenate so that historical lag features generate correctly for the start of test.csv
    combined_df = pd.concat([train_clean, test_clean], ignore_index=True)
    combined_df = combined_df.sort_values('date')
    combined_df = create_features(combined_df)
    
    # Split back into train and test masks
    target_col = 'Electricity_balance_not_controllable'
    is_test_mask = combined_df['Id'].isin(test_ids)
    
    train_features_df = combined_df[~is_test_mask]
    test_features_df = combined_df[is_test_mask]
    
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
        'temp_x_hour', 'wind_sq', 'wind_cube'
    ]
    
    X_train = train_features_df[base_features]
    y_train = train_features_df[target_col]
    X_test = test_features_df[base_features]
    
    print("Training Direct Model (HistGradientBoosting)...")
    model = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    
    print("Generating predictions...")
    preds = model.predict(X_test)
    
    submission = pd.DataFrame({
        'Id': test_features_df['Id'].astype(int),
        'Predicted': preds
    }).sort_values('Id')
    
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}")

if __name__ == "__main__":
    generate_submission()
