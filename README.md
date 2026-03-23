# Kaggle Load Solar Wind Forecast 

This is a self-study project based on data from the Kaggle competition “Not Controllable Electricity Balance Forecast.” The goal is to strengthen my machine learning skills and explore how AI/ML can be applied in the energy sector.

This repository contains an end-to-end data processing and machine learning solution for forecasting `Electricity_balance_not_controllable` based on weather conditions (temperature, nebulosity, wind) and calendar data. 

## Project Structure

* **`data_pipeline.py`** 
  Contains the `clean_and_interpolate()` and `create_features()` functions. This script ensures chronological continuity, applies localized interpolation for temporary data gaps, and constructs advanced predictive features:
  * **Cyclical Encodings:** Transforms hour-of-day and month into sine/cosine embeddings.
  * **Historical Lags:** Autoregressive lookbacks at exactly 1 day and 1 week increments.
  * **Rolling Windows:** Smoothing factors and rolling standard deviations over recent 6-hour windows.
  * **Polynomial Features:** Encodes the physical $v^2$ and $v^3$ relationship for Wind interactions.

* **`baseline_models.py`**
  Empirically compares two distinct forecasting architectures against an out-of-time validation set:
  * **Approach 1 (Direct Forecasting):** Trains a single `HistGradientBoostingRegressor` directly against the target balance.
  * **Approach 2 (Component Forecasting):** Trains separated specialized models for Load, Solar, and Wind individually (using Linear & Tree models), then mathematically subtracts them to derive the target balance.

* **`generate_submission.py`**
  The final execution script. Trains the optimal winning model (Approach 1) on 100% of the training data and outputs predictions for the 17,520 rows in `test.csv`.

## Results
`baseline_models.py` empirically demonstrates **Approach 1 (Direct Forecasting)** to be substantially superior. 

While Component forecasting is physically intuitive, it suffers from heavy *compounding errors*. Furthermore, sophisticated Tree models in the Direct approach easily capture dynamic real-world caps—such as wind turbine "cut-out" speeds where energy production drops rapidly to $0$ at very high wind velocities—which sub-models struggle to isolate securely.

## How to Run

1. **Test out-of-bag validation metrics:**
   ```bash
   python baseline_models.py
   ```

2. **Generate Kaggle Submission output:**
   ```bash
   python generate_submission.py
   ```
   *The script will output `submission.csv` inside the `/data` folder, exactly formatted with `Id,Predicted`.*
