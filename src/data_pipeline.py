import pandas as pd
import numpy as np

HALF_HOURS_PER_DAY = 48
HALF_HOURS_PER_WEEK = 7 * HALF_HOURS_PER_DAY


def clean_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and interpolates missing/invalid data in the load/solar/wind dataset.
    """
    df = df.copy()
    
    # 1. Parse dates and expose missing time steps / intervals
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    
    # Resample to strictly 30-min frequencies to expose gaps
    df = df.resample('30min').asfreq()
    
    if 'Id' in df.columns:
        df['Id'] = df['Id'].interpolate(method='linear').round()

    # 2. Fix invalid zeros (Load should not be 0)
    if 'Load' in df.columns:
        invalid_load_mask = df['Load'] <= 0
        df.loc[invalid_load_mask, 'Load'] = np.nan
            
    # 3. Interpolate targets
    targets_to_interpolate = [
        'Load', 'Solar_power', 'Wind_power', 
        'temperature', 'wind', 'nebulosity', 
        'Electricity_balance_not_controllable'
    ]
    
    cols_to_fill = [c for c in targets_to_interpolate if c in df.columns]
    
    for col in cols_to_fill:
        # Step A: Linear interpolation for short gaps
        df[col] = df[col].interpolate(method='linear', limit=8)
        
        # Step B: 1 week shift fallback
        df[col] = df[col].fillna(df[col].shift(HALF_HOURS_PER_WEEK))
        df[col] = df[col].fillna(df[col].shift(-HALF_HOURS_PER_WEEK))
        
        # Step C: Median fallback
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    df['month'] = df.index.month
    df['year'] = df.index.year
    df['Date'] = df.index.strftime('%Y%m%d').astype(int)
    
    return df.reset_index()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features such as cyclical time representations,
    lags, rolling statistics, and basic interaction terms.
    """
    df = df.copy()
    
    # 1. Cyclical Time Features; these are typical calendar features
    df['tod_sin'] = np.sin(2 * np.pi * df['tod'] / 48.0)
    df['tod_cos'] = np.cos(2 * np.pi * df['tod'] / 48.0)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 2. Lag Features (Weather features only)
    # 30-minute data frequency -> 1 day = 48 steps, 2 days = 96 steps, 1 week = 336 steps.
    for col in ['temperature', 'nebulosity', 'wind']:
        df[f'{col}_lag_1d'] = df[col].shift(HALF_HOURS_PER_DAY)
        df[f'{col}_lag_2d'] = df[col].shift(2 * HALF_HOURS_PER_DAY)
        df[f'{col}_lag_1w'] = df[col].shift(HALF_HOURS_PER_WEEK)
        
    # 3. Rolling Window Statistics (6 Hours = 12 half-hours)
    for col in ['temperature', 'wind']:
        df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=12, min_periods=1).mean()
        df[f'{col}_rolling_std_6h'] = df[col].rolling(window=12, min_periods=1).std()
        
    # 4. Interaction Features
    df['temp_x_hour'] = df['temperature'] * (df['tod'] / 2.0)
    
    # 5. Polynomial Features
    df['wind_sq'] = df['wind'] ** 2
    df['wind_cube'] = df['wind'] ** 3
    
    # Backfill the NaNs created by lagging/rolling at the very start of the dataset
    df = df.bfill()
    
    return df


def get_base_features() -> list[str]:
    """Return the canonical feature list used for model training/inference."""
    return [
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
