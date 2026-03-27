import pandas as pd
import numpy as np

def clean_and_interpolate(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
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
        df[col] = df[col].fillna(df[col].shift(336))
        df[col] = df[col].fillna(df[col].shift(-336))
        
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
    
    # 2. Lag Features (Weather features only); We can add more lag features here. Since we have weather forecast, technically we can also try lead features
    for col in ['temperature', 'nebulosity', 'wind']:
        df[f'{col}_lag_1d'] = df[col].shift(24)
        df[f'{col}_lag_2d'] = df[col].shift(48)
        df[f'{col}_lag_1w'] = df[col].shift(336)
        
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
    df = df.fillna(method='bfill')
    
    return df
