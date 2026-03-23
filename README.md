# Kaggle Load Solar Wind Forecast 

This is a self-study project based on data from the Kaggle competition “Not Controllable Electricity Balance Forecast.” The goal is to strengthen my machine learning skills and explore how AI/ML can be applied in the energy sector.

This repository contains an end-to-end data processing and machine learning solution for forecasting `Electricity_balance_not_controllable` based on weather conditions (temperature, nebulosity, wind) and calendar data. 

I tried two different approaches:
  * **Approach 1 (Direct Forecasting):** Trains a single `HistGradientBoostingRegressor` directly against the target balance.
  * **Approach 2 (Component Forecasting):** Trains separated specialized models for Load, Solar, and Wind individually (using Linear & Tree models), then mathematically subtracts them to derive the target balance.

Results: `baseline_models.py` empirically demonstrates **Approach 1 (Direct Forecasting)** to be substantially superior. 

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
