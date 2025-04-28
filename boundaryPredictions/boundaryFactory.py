import pandas as pd
import numpy as np
import warnings
from tkinter import messagebox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from openpyxl import load_workbook

# --- Product Class Definition (Ensure this matches your actual Product class) ---
class Product:
    """Represents a product with its data and forecast/boundaries."""
    def __init__(self, productID):
        self.productID = productID
        self.boundaries = []
        self.dates = []
        self.demand = []
        self.forecast = []
        self.forecastDates = []
        self.backtestMetrics = {} # To store MAPE results
        self.bestParams = {}


# --- XGBoost Helper Functions (Copied from Xgboost.py) ---
# [Keep all helper functions calculateForecastAccuracy, detectAndHandleOutliers, etc. as they were]
# --- (Helper functions omitted for brevity in this response, assume they are present) ---
def calculateForecastAccuracy(actual, predicted):
    if not isinstance(actual, np.ndarray): actual = np.array(actual)
    if not isinstance(predicted, np.ndarray): predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("Arrays must be same length/shape")
    if len(actual) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'mpe': np.nan}

    results = {}
    results['mae'] = mean_absolute_error(actual, predicted)
    results['rmse'] = np.sqrt(mean_squared_error(actual, predicted))

    nonZeroIdx = actual != 0
    if np.sum(nonZeroIdx) > 0:
        # Ensure division is safe (avoid dividing by zero if actual is zero but predicted isn't)
        actual_safe = actual[nonZeroIdx]
        predicted_safe = predicted[nonZeroIdx]
        percentage_errors = (actual_safe - predicted_safe) / actual_safe
        results['mape'] = np.mean(np.abs(percentage_errors)) * 100
        results['mpe'] = np.mean(percentage_errors) * 100
    else:
        results['mape'] = np.nan
        results['mpe'] = np.nan
    return results

def detectAndHandleOutliers(series, threshold=2.5, productID=None):
    if not isinstance(series, pd.Series): series = pd.Series(series)
    if len(series) < 12: return series # Not enough data for robust detection

    try:
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce').dropna()
            if len(series) < 12: return series
    except Exception as e:
        print(f"Error converting series for outlier detection: {e}")
        return series

    adjusted = series.copy()
    median = adjusted.median()
    # Use nanmedian for MAD calculation to handle potential NaNs introduced by coercion
    mad = np.nanmedian(np.abs(adjusted - median))
    madScale = 1.4826 # Standard factor for MAD to estimate std dev

    if mad > 1e-6: # Avoid division by zero or near-zero
        # Calculate robust Z-scores using median and MAD
        zScores = np.abs(adjusted - median) / (mad * madScale)
        outliers = zScores > threshold

        if outliers.any(): # Use .any() here
            # Cap outliers at median +/- threshold * MAD * scale
            upperBound = median + (threshold * mad * madScale)
            lowerBound = median - (threshold * mad * madScale)
            adjusted[zScores > threshold] = np.clip(adjusted[zScores > threshold], lowerBound, upperBound)
            # Alternative: Replace with median or NaN
            # adjusted[outliers] = median
    # If MAD is zero (e.g., constant series), no outliers are detected by this method
    return adjusted


def create_features(df, target_col='Value', include_lags=True, productID=None):
    """Create enhanced features for time series forecasting."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        warnings.warn("DataFrame index is not DatetimeIndex. Date features may be inaccurate.")
        # Attempt to convert if possible, otherwise skip date features
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            print("Could not convert index to DatetimeIndex.")
            # Create basic time features even without datetime index
            df['time_idx'] = np.arange(len(df))
            df['time_idx_sq'] = df['time_idx']**2
            # Add lag features if requested and possible
            if include_lags and target_col in df.columns:
                min_len_for_lag = {1: 2, 2: 3, 4: 5} # Min length needed for lag
                for lag, min_len in min_len_for_lag.items():
                    if len(df) >= min_len:
                        df[f'lag_{lag}'] = df[target_col].shift(lag)
            return df

    # Basic date features
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week.astype(int)
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['weekofyear'] = df.index.isocalendar().week.astype(int) # Same as 'week'
    df['year'] = df.index.year

    # Add lag features if possible and requested
    if include_lags and target_col in df.columns:
        min_len_for_lag = {1: 2, 2: 3, 4: 5, 8: 9, 13: 14} # Min length needed for lag
        for lag, min_len in min_len_for_lag.items():
            if len(df) >= min_len:
                df[f'lag_{lag}'] = df[target_col].shift(lag)

        # Rolling statistics (use median for robustness)
        rolling_windows = [4, 8]
        for window in rolling_windows:
             if len(df) >= window:
                 df[f'rolling_median_{window}'] = df[target_col].rolling(window=window, min_periods=1).median()
                 df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=max(2, window // 2)).std() # Need at least 2 points for std

        # Exponentially weighted features
        if len(df) > 4: # Need a few points for EWM
            df['ewm_mean_8'] = df[target_col].ewm(span=8, min_periods=1).mean()
            df['ewm_std_8'] = df[target_col].ewm(span=8, min_periods=max(2, 4)).std() # Need points for std

    return df

def get_adaptive_blend_weight(forecast_idx, historical_data_len):
    """Calculate adaptive blend weights based on forecast horizon and data length."""
    # Base decay factor - rely less on XGBoost for longer forecasts
    base_weight = max(0.8 - (forecast_idx * 0.05), 0.3) # Start high, decay to 0.3

    # If we have limited historical data, rely less on complex XGBoost model
    if historical_data_len < 26: # Less than half a year
        base_weight = max(0.6 - (forecast_idx * 0.08), 0.2)
    elif historical_data_len < 12: # Less than a quarter
        base_weight = max(0.4 - (forecast_idx * 0.1), 0.1) # Rely heavily on seasonal/simple model

    return base_weight

def create_seasonal_model(historical_df, target_col='Value'):
    """Create a simple seasonal model (e.g., average of historical periods)."""
    if historical_df.empty or target_col not in historical_df.columns:
        return {'mean': 0, 'seasonal_factors': {}} # Return default if no data

    # Simple model: Use overall mean and potentially period-of-year factors
    # Assuming the index represents periods (like quarters)
    model = {}
    model['mean'] = historical_df[target_col].mean()

    # If index is DatetimeIndex, try extracting seasonal component (e.g., quarter)
    if isinstance(historical_df.index, pd.DatetimeIndex):
        historical_df['period'] = historical_df.index.quarter # Or month, week etc.
        period_means = historical_df.groupby('period')[target_col].mean()
        if model['mean'] > 1e-6:
             model['seasonal_factors'] = (period_means / model['mean']).to_dict()
        else:
             model['seasonal_factors'] = {p: 1.0 for p in period_means.index}
    else:
        # If not datetime index, maybe use modulo for seasonality if index is numeric
        try:
            # Assuming 4 periods per cycle (like quarters)
            historical_df['period'] = (historical_df.index % 4) + 1
            period_means = historical_df.groupby('period')[target_col].mean()
            if model['mean'] > 1e-6:
                 model['seasonal_factors'] = (period_means / model['mean']).to_dict()
            else:
                 model['seasonal_factors'] = {p: 1.0 for p in period_means.index}
        except TypeError:
             # Cannot determine seasonality easily
             model['seasonal_factors'] = {}

    return model

def get_seasonal_prediction(seasonal_model, date_or_period):
    """Get prediction from seasonal model for a specific date or period index."""
    if not seasonal_model or 'mean' not in seasonal_model:
        return 0

    base_prediction = seasonal_model['mean']
    factor = 1.0

    if 'seasonal_factors' in seasonal_model and seasonal_model['seasonal_factors']:
        period = None
        if isinstance(date_or_period, (pd.Timestamp, np.datetime64)):
            period = pd.Timestamp(date_or_period).quarter # Assuming quarterly seasonality
        elif isinstance(date_or_period, (int, np.integer)):
             # Assuming period index corresponds to keys in seasonal_factors
             # Need to know the cycle length (e.g., 4 for quarters)
             cycle_length = 4 # Default assumption
             period_keys = list(seasonal_model['seasonal_factors'].keys())
             if period_keys:
                 # Infer cycle length if possible, otherwise default to 4
                 if all(isinstance(k, int) for k in period_keys):
                     cycle_length = max(period_keys) # Assumes keys are 1, 2, 3, 4...
             period = (date_or_period % cycle_length)
             # Adjust if keys are 1-based
             if 0 not in period_keys and 1 in period_keys:
                 period += 1

        if period is not None:
            factor = seasonal_model['seasonal_factors'].get(period, 1.0)

    prediction = base_prediction * factor
    return prediction

# --- NEW Bias Function ---
def apply_regression_bias(prediction, historical_values, sensitivity=0.5):
    """
    Applies a regression-to-mean bias to predictions based on historical statistics.

    Args:
        prediction (float or array-like): The raw prediction value(s) to adjust
        historical_values (array-like): Historical data for the product
        sensitivity (float): Controls how strongly to apply the bias (0.0-1.0)
                            Higher values create stronger regression to mean

    Returns:
        float or array-like: The adjusted prediction value(s)
    """
    # If prediction is an array, handle it recursively
    if isinstance(prediction, (np.ndarray, pd.Series)) and len(prediction) > 0:
        # Apply the function to each element if it's an array
        if isinstance(prediction, pd.Series):
            return prediction.apply(lambda x: apply_regression_bias(x, historical_values, sensitivity))
        else:
            # Ensure historical_values is treated as a single list/array for each call
            return np.array([apply_regression_bias(p, historical_values, sensitivity) for p in prediction])

    # --- Original function logic for scalar prediction ---
    if not isinstance(historical_values, (list, np.ndarray, pd.Series)) or len(historical_values) < 2:
        return prediction  # Not enough data to calculate statistics

    # Filter out NaN values and ensure we have numeric data
    historical_clean = np.array([x for x in historical_values if pd.notna(x) and isinstance(x, (int, float))])
    if len(historical_clean) < 2:
        return prediction

    # Calculate statistics
    hist_mean = np.mean(historical_clean)
    hist_std = np.std(historical_clean)

    if hist_std < 1e-6:  # Avoid division by zero for constant series
        return prediction

    # Calculate z-score of prediction (how many standard deviations from mean)
    # Ensure prediction is numeric before calculation
    if not isinstance(prediction, (int, float)):
        try:
            prediction = float(prediction)
        except (ValueError, TypeError):
            return prediction # Return original if cannot convert

    z_score = (prediction - hist_mean) / hist_std

    # Calculate bias factor - stronger bias for predictions far from mean
    # The bias pulls prediction toward the mean proportionally to z-score and sensitivity
    # Using tanh for smooth scaling of bias factor
    bias_factor = 1.0 - (sensitivity * np.tanh(abs(z_score) / 2)) # Divide z_score to control steepness

    # Apply bias: move prediction toward mean by reducing the distance
    adjusted_prediction = hist_mean + (prediction - hist_mean) * bias_factor

    return adjusted_prediction


# --- XGBoost Forecasting Implementation ---
def calculateXGBoostForecast(dataSeries, forecastDatesIndex, params, productID=None):
    """Generate forecasts using XGBoost model with adaptive ensemble."""
    target_col = 'Value' # Generic name for the value being forecast

    if not isinstance(dataSeries, pd.Series):
        raise ValueError("dataSeries must be a pandas Series")

    # Ensure dataSeries has enough non-NaN values to train
    if dataSeries.dropna().empty or len(dataSeries.dropna()) < params.get('min_train_size', 5):
        warnings.warn(f"Not enough valid data points in dataSeries for {productID} ({len(dataSeries.dropna())}). Returning simple mean forecast.")
        mean_val = dataSeries.mean() # Mean of whatever is there (might be NaN)
        return [max(0, mean_val if pd.notna(mean_val) else 0.0)] * len(forecastDatesIndex)

    if len(forecastDatesIndex) == 0:
        warnings.warn(f"Zero forecast length requested for {productID}. Returning empty list.")
        return []


    # Ensure index is DatetimeIndex for feature engineering
    if not isinstance(dataSeries.index, pd.DatetimeIndex):
         warnings.warn(f"Index for {productID} is not DatetimeIndex. Creating one assuming sequential periods.")
         # Assuming quarterly data based on original column names like 'Q3 95'
         try:
             # Try to infer frequency if possible, default to Quarter start 'QS'
             inferred_freq = pd.infer_freq(dataSeries.index) if isinstance(dataSeries.index, pd.RangeIndex) else 'Q' # Use Quarter End
             if inferred_freq is None: inferred_freq = 'Q' # Default if inference fails
             start_date = pd.to_datetime('2020-01-01') # Arbitrary start date
             dataSeries.index = pd.date_range(start=start_date, periods=len(dataSeries), freq=inferred_freq)
             print(f"  Created DatetimeIndex for {productID} with frequency '{dataSeries.index.freqstr}'")
         except Exception as e:
             print(f"Error creating DatetimeIndex for {productID}: {e}. Proceeding with numerical index.")
             dataSeries.index = pd.RangeIndex(start=0, stop=len(dataSeries), step=1)


    # Prepare historical data DataFrame
    historical_df = pd.DataFrame({target_col: dataSeries})

    # Handle potential outliers
    historical_df[target_col] = detectAndHandleOutliers(
        historical_df[target_col],
        threshold=params.get('spikeThreshold', 2.5),
        productID=productID
    )

    # Create features for historical data
    historical_features = create_features(historical_df, target_col=target_col, productID=productID)

    # Drop rows with NaN target *before* selecting features
    historical_features_clean = historical_features.dropna(subset=[target_col])
    # Select features used for training (X) and the target (y)
    feature_cols = [col for col in historical_features_clean.columns if col != target_col]
    # Further drop rows where any *feature* is NaN AFTER target check
    historical_features_clean = historical_features_clean.dropna(subset=feature_cols)


    if historical_features_clean.empty or len(historical_features_clean) < params.get('min_train_size', 5): # Need some data to train
        warnings.warn(f"Not enough clean historical data for {productID} after feature creation ({len(historical_features_clean)}). Using simple mean forecast.")
        mean_val = historical_df[target_col].mean()
        return [max(0, mean_val if pd.notna(mean_val) else 0.0)] * len(forecastDatesIndex)

    y_train = historical_features_clean[target_col]
    X_train = historical_features_clean[feature_cols]

    # Scale features
    scaler = StandardScaler()
    try:
        # Ensure all columns are numeric before scaling
        X_train_numeric = X_train.select_dtypes(include=np.number).fillna(0) # Select only numeric, fill remaining NaNs
        if X_train_numeric.empty:
             warnings.warn(f"No numeric features found for scaling for {productID}. Proceeding without scaling.")
             X_train_scaled = X_train_numeric # Will be empty, handled below
        else:
             X_train_scaled = pd.DataFrame(
                 scaler.fit_transform(X_train_numeric),
                 columns=X_train_numeric.columns, # Use columns from numeric subset
                 index=X_train_numeric.index
             )
    except Exception as e:
        warnings.warn(f"Error scaling features for {productID}: {e}. Proceeding without scaling.")
        X_train_scaled = X_train.select_dtypes(include=np.number).fillna(0)


    # Define XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=params.get('n_estimators', 100),
        learning_rate=params.get('learning_rate', 0.05),
        max_depth=params.get('max_depth', 3),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        min_child_weight=params.get('min_child_weight', 1),
        gamma=params.get('gamma', 0),
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1 # Use all available cores
    )

    # Train the model only if we have scaled features
    model_trained = False
    if not X_train_scaled.empty:
        try:
            model.fit(X_train_scaled, y_train)
            model_trained = True
        except Exception as e:
             warnings.warn(f"Error training XGBoost model for {productID}: {e}. Using simple mean forecast.")
    else:
        warnings.warn(f"No features available for training XGBoost model for {productID}. Using simple mean forecast.")

    if not model_trained:
         mean_val = historical_df[target_col].mean()
         return [max(0, mean_val if pd.notna(mean_val) else 0.0)] * len(forecastDatesIndex)


    # Create a simple seasonal model for blending
    seasonal_model = create_seasonal_model(historical_df, target_col=target_col)

    # Prepare forecast dataframe structure
    forecast_df = pd.DataFrame(index=forecastDatesIndex)
    forecast_df[target_col] = np.nan

    # Combine historical and forecast structure for iterative feature creation
    combined_df = pd.concat([historical_df[[target_col]], forecast_df]) # Only need target col for lags

    # Iteratively predict each future point
    forecasts = []
    current_data_for_features = combined_df.copy() # Start with historical + empty forecast slots

    # Get the columns used for scaling (subset of original features)
    scaled_feature_cols = X_train_scaled.columns

    for i, forecast_date in enumerate(forecastDatesIndex):
        # Create features for the *entire* series up to the point *before* the current forecast date
        features_for_forecast_step = create_features(current_data_for_features, target_col=target_col, productID=productID)

        # Get the feature row for the current forecast date
        try:
            forecast_features_row = features_for_forecast_step.loc[[forecast_date]].copy()
        except KeyError:
             warnings.warn(f"Could not locate forecast date {forecast_date} in feature set for {productID}. Using seasonal.")
             prediction = get_seasonal_prediction(seasonal_model, forecast_date)
             # Apply bias even to seasonal fallback
             historical_values = historical_df[target_col].values
             prediction = apply_regression_bias(prediction, historical_values, sensitivity=0.3) # Apply bias
             prediction = max(0, prediction) # Ensure non-negative
             forecasts.append(prediction)
             current_data_for_features.loc[forecast_date, target_col] = prediction
             continue

        # Align features with those used in training (especially the scaled ones)
        forecast_features_aligned = forecast_features_row[scaled_feature_cols] # Select only scaled columns

        # Handle potential NaNs in the forecast feature row
        for col in scaled_feature_cols:
            if forecast_features_aligned[col].isna().any(): # Use .any() for Series check
                # Impute using mean from the original training feature set (before scaling)
                if col in X_train.columns and pd.notna(X_train[col].mean()):
                    fill_value = X_train[col].mean()
                else:
                    fill_value = 0 # Fallback if mean is NaN or col not in original X_train
                forecast_features_aligned[col] = forecast_features_aligned[col].fillna(fill_value)

        # Scale the forecast features using the *fitted* scaler
        try:
            # Ensure numeric types before scaling
            forecast_features_numeric = forecast_features_aligned.select_dtypes(include=np.number).fillna(0)
            if forecast_features_numeric.empty:
                 raise ValueError("No numeric features to scale for forecast step.")

            forecast_features_scaled = pd.DataFrame(
                scaler.transform(forecast_features_numeric), # Use transform
                columns=forecast_features_numeric.columns,
                index=forecast_features_numeric.index
            )
            # Make prediction with XGBoost
            xgb_prediction = model.predict(forecast_features_scaled)[0]

        except (NotFittedError, ValueError) as e:
             warnings.warn(f"Scaler/Prediction issue for {productID} at step {i}: {e}. Using unscaled data if possible.")
             # Try predicting with unscaled data as a fallback
             try:
                 # Use original X_train columns for alignment if predicting unscaled
                 forecast_features_unscaled_aligned = forecast_features_row[X_train.columns]
                 # Impute NaNs in unscaled features
                 for col in X_train.columns:
                     if forecast_features_unscaled_aligned[col].isna().any(): # Use .any()
                         fill_value = X_train[col].mean() if pd.notna(X_train[col].mean()) else 0
                         forecast_features_unscaled_aligned[col] = forecast_features_unscaled_aligned[col].fillna(fill_value)

                 forecast_features_numeric_unscaled = forecast_features_unscaled_aligned.select_dtypes(include=np.number).fillna(0)
                 if forecast_features_numeric_unscaled.empty: raise ValueError("No numeric features for unscaled prediction.")

                 # Re-train a model on unscaled data (less ideal, but fallback)
                 unscaled_model = xgb.XGBRegressor(**model.get_params()) # Get params from original model
                 unscaled_model.fit(X_train.select_dtypes(include=np.number).fillna(0), y_train)
                 xgb_prediction = unscaled_model.predict(forecast_features_numeric_unscaled)[0]

             except Exception as fallback_e:
                 warnings.warn(f"Error predicting with unscaled data for {productID}: {fallback_e}. Using seasonal.")
                 xgb_prediction = get_seasonal_prediction(seasonal_model, forecast_date) # Final fallback

        except Exception as e:
            warnings.warn(f"General error scaling/predicting features for {productID} at step {i}: {e}. Using seasonal.")
            xgb_prediction = get_seasonal_prediction(seasonal_model, forecast_date) # Final fallback


        # Get seasonal prediction for blending
        seasonal_prediction = get_seasonal_prediction(seasonal_model, forecast_date)

        # Use adaptive blending
        blend_weight = get_adaptive_blend_weight(
            forecast_idx=i,
            historical_data_len=len(historical_df.dropna(subset=[target_col])) # Use length of valid historical points
        )

        prediction = blend_weight * xgb_prediction + (1 - blend_weight) * seasonal_prediction

        # --- Apply regression-to-mean bias ---
        historical_values = historical_df[target_col].values
        prediction = apply_regression_bias(prediction, historical_values, sensitivity=0.3) # Adjust sensitivity as needed

        # Ensure non-negative forecast
        prediction = max(0, prediction)

        # Store prediction
        forecasts.append(prediction)

        # Update the combined dataframe with the prediction for the next step's lag calculation
        current_data_for_features.loc[forecast_date, target_col] = prediction

    return forecasts


# --- Main Boundary Prediction and Forecasting Function ---
def makeBoundaryPredictions(boundaries, productsWithForecast, weeksToForecast):
    """
    Extracts historical boundaries, forecasts future boundaries using XGBoost,
    calculates backtest MAPE, and updates Product objects with integer forecasts.
    """

    if not isinstance(boundaries, pd.DataFrame):
        messagebox.showerror("Error", "Input 'boundaries' must be a pandas DataFrame.")
        return
    if not isinstance(productsWithForecast, list):
        messagebox.showerror("Error", "Input 'productsWithForecast' must be a list of Product objects.")
        return
    if not isinstance(weeksToForecast, int) or weeksToForecast < 0:
         messagebox.showerror("Error", "Input 'weeksToForecast' must be a non-negative integer.")
         return

    # Rename columns
    boundaries = boundaries.rename(columns={'Unnamed: 0': 'Product ID',
                                            'Unnamed: 1': 'Attribute'})

    # --- Filter based on 'Q3 95' (as per original code) ---
    filter_column = 'Q3 95'
    if filter_column not in boundaries.columns:
         messagebox.showerror("Error", f"Filter column '{filter_column}' not found in boundaries DataFrame.")
         return

    notNanMask = boundaries[filter_column].notna()
    notNanDF = boundaries[notNanMask].copy()

    if notNanDF.empty:
        messagebox.showwarning("Warning", f"No rows remaining after filtering NaNs in '{filter_column}'. Cannot proceed.")
        return

    # Deselecting specific string rows like 'Product ID' or 'Total'
    stringsToExclude = ['Product ID', 'Total']
    notStringsMask = ~notNanDF['Product ID'].isin(stringsToExclude)
    notStringsDF = notNanDF[notStringsMask].copy()

    if notStringsDF.empty:
        messagebox.showwarning("Warning", "No product rows remaining after excluding 'Product ID' and 'Total'. Cannot proceed.")
        return

    # --- Convert potential boundary columns to numeric ---
    print("\n--- Data types BEFORE conversion ---")
    # Use .head() to avoid printing huge info for many columns
    print(notStringsDF.head().info())

    potential_cols_to_convert = [
        col for col in notStringsDF.columns
        if col not in ['Product ID', 'Attribute']
    ]
    print(f"\nAttempting conversion for columns: {potential_cols_to_convert}")

    for col in potential_cols_to_convert:
        if col in notStringsDF.columns:
            original_dtype = str(notStringsDF[col].dtype)
            try:
                notStringsDF[col] = pd.to_numeric(notStringsDF[col], errors='coerce')
                new_dtype = str(notStringsDF[col].dtype)
                if original_dtype != new_dtype:
                    print(f"  Converted '{col}' from {original_dtype} to {new_dtype}")

                # Fill NaNs resulting from coercion with the mean of the column
                if notStringsDF[col].isnull().any(): # Use .any()
                    mean_val = notStringsDF[col].mean()
                    if pd.notna(mean_val):
                        # Fill with mean first (keeps it float for now)
                        notStringsDF[col] = notStringsDF[col].fillna(mean_val)
                        print(f"  Filled NaNs in '{col}' with mean ({mean_val:.2f}).")
                    else:
                        # Fill with 0 if mean is NaN
                        notStringsDF[col] = notStringsDF[col].fillna(0)
                        print(f"  Mean for '{col}' is NaN. Filled NaNs with 0.")
                    # We will cast to int later, after extracting the row

            except Exception as e:
                print(f"  Could not convert column '{col}' to numeric: {e}")

    print("\n--- Data types AFTER conversion (and potential NaN fill) ---")
    print(notStringsDF.head().info())

    # Now, identify numeric boundaries columns (should be float or int now)
    potential_boundaries_cols = [
        col for col in notStringsDF.columns
        if col not in ['Product ID', 'Attribute'] and pd.api.types.is_numeric_dtype(notStringsDF[col])
    ]

    if not potential_boundaries_cols:
         messagebox.showerror("Error", "Still no numeric boundaries columns found after attempting conversion.")
         return

    print(f"\nIdentified potential boundaries columns: {potential_boundaries_cols}")

    # Set 'Product ID' as index for efficient lookup
    try:
        boundaries_indexed = notStringsDF.set_index('Product ID')
    except KeyError:
        messagebox.showerror("Error", "'Product ID' column not found after renaming/filtering.")
        return

    # --- Process each product ---
    for product in productsWithForecast:
        if not hasattr(product, 'productID') or not hasattr(product, 'boundaries'):
             print(f"Warning: Skipping item in productsWithForecast list as it doesn't seem like a valid Product object: {product}")
             continue

        # Clear previous boundaries and metrics
        product.boundaries = []
        product.backtestMetrics = {} # Initialize/clear backtest metrics
        historical_boundaries = []

        try:
            # Find the row(s) for the current product using its ID as the index
            product_data = boundaries_indexed.loc[[product.productID]] # Use double brackets to always get DataFrame

            if product_data.empty:
                raise KeyError # Product ID not found after all filtering

            # If multiple rows match, use the first one (or add logic to handle duplicates)
            if len(product_data) > 1:
                product_row = product_data.iloc[0]
                warnings.warn(f"Multiple rows found for Product ID {product.productID}. Using the first row.")
            else:
                product_row = product_data.iloc[0] # Get the Series from the single-row DataFrame

            # Extract historical boundaries values from the identified columns
            for column in potential_boundaries_cols:
                try:
                    boundaries_value = product_row[column]

                    # Check if boundaries_value is somehow still an array/Series (shouldn't be with .iloc[0])
                    if isinstance(boundaries_value, (np.ndarray, pd.Series)):
                        # This case is less likely now but kept as safeguard
                        boundaries_value = boundaries_value.iloc[0] if isinstance(boundaries_value, pd.Series) else boundaries_value[0]
                        warnings.warn(f"Unexpected array/Series found for {product.productID}, column {column}. Using first value.")

                    # Check for NaN with scalar value
                    if pd.isna(boundaries_value):
                        warnings.warn(f"Unexpected NaN found in column '{column}' for product {product.productID} after fillna. Replacing with 0.")
                        historical_boundaries.append(0.0)
                    else:
                        # Append as float initially
                        historical_boundaries.append(float(boundaries_value))
                except Exception as e:
                    print(f"Error extracting value from column {column} for product {product.productID}: {e}")
                    historical_boundaries.append(0.0) # Default to 0 on error

            # Assign historical boundaries (as floats for now)
            print(f"Product {product.productID}: Found {len(historical_boundaries)} historical boundaries.")

            # --- Forecast future boundaries if requested and possible ---
            if weeksToForecast >= 0 and len(historical_boundaries) > 0: # Allow weeksToForecast=0 for backtest only

                # Create a pandas Series with a DatetimeIndex for XGBoost functions
                try:
                    start_date = pd.to_datetime('2020-01-01') # Arbitrary start
                    # Assuming quarterly frequency based on column names like 'Q3 95'
                    historical_dates = pd.date_range(start=start_date, periods=len(historical_boundaries), freq='Q') # Use Quarter End
                    boundaries_series = pd.Series(historical_boundaries, index=historical_dates, dtype=float)

                    # Define default parameters for XGBoost boundaries forecast
                    params = {
                        'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3,
                        'min_child_weight': 1, 'gamma': 0, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'spikeThreshold': 2.5,
                        'min_train_size': 5 # Min data points needed to train XGBoost
                    }

                    # --- Backtesting for MAPE ---
                    backtest_periods = min(4, len(historical_boundaries) // 3) # e.g., 4 periods, or 1/3 of data if less
                    if backtest_periods > 0 and len(historical_boundaries) >= params['min_train_size'] + backtest_periods:
                        print(f"  Running backtest ({backtest_periods} periods) for {product.productID}...")
                        train_series = boundaries_series.iloc[:-backtest_periods]
                        test_series = boundaries_series.iloc[-backtest_periods:]

                        # --- Add check for dimensionality ---
                        if train_series.ndim > 1:
                            warnings.warn(f"Backtest train_series has multiple dimensions for {product.productID}. Using first column.")
                            train_series = train_series.iloc[:, 0] if isinstance(train_series, pd.DataFrame) else train_series.flatten()
                        if test_series.ndim > 1:
                            warnings.warn(f"Backtest test_series has multiple dimensions for {product.productID}. Using first column.")
                            test_series = test_series.iloc[:, 0] if isinstance(test_series, pd.DataFrame) else test_series.flatten()
                        # --- End check ---

                        backtest_forecast_values = calculateXGBoostForecast(
                            dataSeries=train_series,
                            forecastDatesIndex=test_series.index,
                            params=params,
                            productID=product.productID
                        )

                        if len(test_series.values) == len(backtest_forecast_values):
                             product.backtestMetrics = calculateForecastAccuracy(
                                 test_series.values,
                                 backtest_forecast_values
                             )
                             mape = product.backtestMetrics.get('mape', np.nan)
                             print(f"    Backtest MAPE: {mape:.2f}%" if pd.notna(mape) else "    Backtest MAPE: N/A")
                        else:
                             print(f"    Backtest error: Length mismatch between actuals ({len(test_series.values)}) and forecast ({len(backtest_forecast_values)}).")
                             product.backtestMetrics = {'error': 'Length mismatch'}
                    else:
                        print(f"  Skipping backtest for {product.productID}: Insufficient historical data ({len(historical_boundaries)} points) or backtest periods=0.")
                        product.backtestMetrics = {'error': 'Insufficient data for backtest'}


                    # --- Generate Actual Future Forecast (if requested) ---
                    forecasted_values = [] # Initialize empty list
                    if weeksToForecast > 0:
                        print(f"  Attempting to forecast {weeksToForecast} future boundaries...")
                        last_historical_date = historical_dates[-1]
                        # Ensure frequency is correctly inferred or set for date offset
                        freq_offset = pd.infer_freq(historical_dates)
                        if freq_offset is None: freq_offset = 'Q' # Default if inference fails

                        forecast_dates_index = pd.date_range(
                            start=last_historical_date + pd.tseries.frequencies.to_offset(freq_offset), # Use offset based on inferred/set freq
                            periods=weeksToForecast,
                            freq=freq_offset
                        )

                        # Calculate the forecast using ALL historical data
                        forecasted_values = calculateXGBoostForecast(
                            dataSeries=boundaries_series, # Use the full series
                            forecastDatesIndex=forecast_dates_index,
                            params=params,
                            productID=product.productID
                        )
                        print(f"  Generated {len(forecasted_values)} forecast values.")

                    # --- Combine historical and forecast, then cast to int ---
                    # Combine historical (already float) and forecasted (float)
                    combined_boundaries_float = historical_boundaries + forecasted_values

                    # Cast all combined values to integers (truncates decimals)
                    # Ensure non-negative before casting, handle potential NaNs in forecast output
                    product.boundaries = [int(max(0, val)) if pd.notna(val) else 0 for val in combined_boundaries_float]

                    print(f"  Stored {len(product.boundaries)} total boundaries (historical + forecast) as integers.")


                except Exception as e:
                    # Display error in messagebox AND print it
                    error_msg = f"Error processing/forecasting boundaries for {product.productID}: {e}"
                    messagebox.showerror("Forecast/Processing Error", error_msg)
                    print(f"  {error_msg}") # Also print to console for logging
                    # Store historical as int, append NaNs for forecast part
                    product.boundaries = [int(max(0, val)) if pd.notna(val) else 0 for val in historical_boundaries]
                    product.boundaries.extend([np.nan] * weeksToForecast) # Append NaNs for failed forecast


            elif weeksToForecast == 0:
                 print(f"  weeksToForecast is 0, skipping future forecast (backtest may have run).")
                 # Store historical boundaries as integers
                 product.boundaries = [int(max(0, val)) if pd.notna(val) else 0 for val in historical_boundaries]

            else: # No historical boundaries found
                 print(f"  No valid historical boundaries found, cannot forecast or backtest.")
                 product.boundaries = [] # Empty list if no history
                 product.backtestMetrics = {'error': 'No historical data'}


        except KeyError:
            # Handle cases where a product ID from productsWithForecast is not found
            msg = f'Product ID {product.productID} not found in boundaries data after filtering. Cannot extract/forecast boundaries.'
            # messagebox.showwarning("Warning", msg) # Avoid too many popups
            print(f"Warning: {msg}")
            product.boundaries = [] # Empty list
            product.backtestMetrics = {'error': 'Product ID not found'}
        except Exception as e:
             # Catch any other unexpected errors during product processing
             error_msg = f"Unexpected error processing product {product.productID}: {e}"
             messagebox.showerror("Product Processing Error", error_msg)
             print(f"ERROR: {error_msg}")
             product.boundaries = [int(max(0, val)) if pd.notna(val) else 0 for val in historical_boundaries] # Store historical if possible
             product.boundaries.extend([np.nan] * weeksToForecast)
             product.backtestMetrics = {'error': f'Unexpected error: {e}'}


    # Print tail end of boundaries for verification
    if weeksToForecast > 0:
        print("\n--- Final Boundary Tails (Last Forecasted Periods) ---")
        for product in productsWithForecast:
            if hasattr(product, 'productID') and hasattr(product, 'boundaries') and product.boundaries:
                 print(f'Product {product.productID}: tail: {product.boundaries[-weeksToForecast:]}')
            elif hasattr(product, 'productID'):
                 print(f'Product {product.productID}: tail: No boundaries generated.')

    return # Function implicitly returns None, no need to return products list as it's modified in-place

