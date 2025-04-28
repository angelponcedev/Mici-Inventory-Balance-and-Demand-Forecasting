import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset
import xgboost as xgb
from forecast.customModel import Product
from sklearn.preprocessing import StandardScaler


# --- Product Class Definition ---
class Product:
    """Represents a product with its weekly demand data and forecast."""
    def __init__(self, productID, dates, demand):
        self.productID = productID
        self.dates = dates
        self.demand = demand
        self.forecast = []
        self.forecastDates = []
        self.backtestMetrics = {}
        self.bestParams = {}
        self.boundaries = []


# --- Helper Functions ---
def calculateForecastAccuracy(actual, predicted):
    if len(actual) != len(predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        if actual.shape != predicted.shape: raise ValueError("Arrays must be same length")
    results = {}
    actual = np.array(actual); predicted = np.array(predicted)
    if len(actual) == 0: return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'mpe': np.nan}
    results['mae'] = mean_absolute_error(actual, predicted)
    results['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
    nonZeroIdx = actual != 0
    if sum(nonZeroIdx) > 0:
        percentage_errors = (actual[nonZeroIdx] - predicted[nonZeroIdx]) / actual[nonZeroIdx]
        results['mape'] = np.mean(np.abs(percentage_errors)) * 100
        results['mpe'] = np.mean(percentage_errors) * 100
    else: results['mape'] = np.nan; results['mpe'] = np.nan
    return results

def detectAndHandleOutliers(series, threshold=2.5, productID=None):
    if len(series) < 12: return series
    try:
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce').dropna()
            if len(series) < 12: return series
    except Exception as e: print(f"Error converting series: {e}"); return series
    
    adjusted = series.copy()
    
    # Adaptive threshold based on coefficient of variation
    cv = adjusted.std() / adjusted.mean() if adjusted.mean() > 0 else 0
    
    # For highly volatile series (like 22B appears to be)
    if cv > 0.5 or (productID and '22B' in productID):
        # Use rolling median for more robust outlier detection
        rolling_median = adjusted.rolling(window=5, center=True, min_periods=2).median()
        
        rolling_median = rolling_median.ffill().bfill()
        
        # Calculate rolling MAD (Median Absolute Deviation)
        rolling_mad = (adjusted - rolling_median).abs().rolling(window=5, center=True, min_periods=2).median()
        
        rolling_mad = rolling_mad.ffill().bfill()
        
        # Identify outliers
        if rolling_mad.min() > 0:
            z_scores = 0.6745 * np.abs(adjusted - rolling_median) / rolling_mad
            outliers = z_scores > threshold
            
            if outliers.any():
                # Replace with rolling median + threshold * MAD (more conservative)
                adjusted.loc[outliers] = rolling_median.loc[outliers] + threshold * rolling_mad.loc[outliers] * np.sign(adjusted.loc[outliers] - rolling_median.loc[outliers])
    else:
        # Original approach for less volatile series
        median = adjusted.median()
        mad = np.median(np.abs(adjusted - median))
        madScale = 3
        
        if mad > 0:
            zScores = madScale * np.abs(adjusted - median) / mad
            outliers = zScores > threshold
            
            if outliers.any():
                upperBound = median + (threshold * mad * madScale)
                adjusted[outliers] = upperBound
    
    return adjusted


def processHistoricalData(rawData, productID):
    productData = rawData[rawData['Product ID'] == productID].copy()
    if productData.empty: return pd.DataFrame(columns=['Demand'])
    try: productData['Date'] = pd.to_datetime(productData['Date'])
    except Exception as e: warnings.warn(f"Could not parse 'Date' for {productID}: {e}"); return pd.DataFrame(columns=['Demand'])
    productData['Weekly Demand'] = pd.to_numeric(productData['Weekly Demand'], errors='coerce')
    productData = productData.dropna(subset=['Date', 'Weekly Demand'])
    if productData.empty: return pd.DataFrame(columns=['Demand'])
    ts = pd.Series(productData['Weekly Demand'].values, index=productData['Date']).sort_index()
    ts = ts[~ts.index.duplicated(keep='last')]
    ts_resampled = ts.resample('W').last()
    nonZeroIndex = ts_resampled[ts_resampled > 0].index
    if not nonZeroIndex.empty: ts_resampled = ts_resampled.loc[nonZeroIndex[0]:]
    return pd.DataFrame({'Demand': ts_resampled})

# --- Enhanced Feature Engineering for XGBoost ---
def create_features(df, include_lags=True, productID=None):
    """Create enhanced features with better handling of volatile patterns"""
    df = df.copy()
    
    # Basic date features
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week.astype(int)
    df['week_of_quarter'] = ((df.index.isocalendar().week - 1) % 13) + 1
    df['is_quarter_end'] = df['week_of_quarter'].isin([12, 13]).astype(int)
    df['year'] = df.index.year
    
    # Add enhanced lag features if possible and requested
    if include_lags and 'Demand' in df.columns:
        # Basic lags
        if len(df) > 1: df['lag_1'] = df['Demand'].shift(1)
        if len(df) > 2: df['lag_2'] = df['Demand'].shift(2)
        if len(df) > 4: df['lag_4'] = df['Demand'].shift(4)
        if len(df) > 13: df['lag_13'] = df['Demand'].shift(13)
        if len(df) > 52: df['lag_52'] = df['Demand'].shift(52)
        
        # For products like 22B that may have more complex patterns
        # Add specific quarter-to-quarter change features
        if len(df) > 13:
            df['q_q_change'] = df['Demand'] / df['lag_13'].replace(0, np.nan)
            df['q_q_change'] = df['q_q_change'].fillna(1)
            # Cap extreme growth/decline values
            df['q_q_change'] = df['q_q_change'].clip(0.1, 10)
        
        # Robust rolling statistics using median 
        if len(df) > 4:
            df['rolling_median_4'] = df['Demand'].rolling(window=4, min_periods=1).median()
            # Directional change indicators (improving/declining)
            if len(df) > 8:
                recent_med = df['Demand'].rolling(window=4, min_periods=1).median()
                older_med = df['Demand'].shift(4).rolling(window=4, min_periods=1).median()
                df['direction'] = (recent_med > older_med).astype(int)
        
        # Exponentially weighted features (more weight to recent observations)
        if len(df) > 8:
            df['ewm_mean'] = df['Demand'].ewm(span=8, min_periods=1).mean()
            df['ewm_std'] = df['Demand'].ewm(span=8, min_periods=1).std()
            # Coefficient of variation as volatility indicator
            df['volatility'] = df['ewm_std'] / df['ewm_mean']
            df['volatility'] = df['volatility'].fillna(0).clip(0, 2)
    
    # Interactions and polynomial features
    df['quarter_week_interaction'] = df['quarter'] * df['week_of_quarter']
    df['quarter_squared'] = df['quarter'] ** 2
    
    return df

# --- Improved Get Adaptive Blend Weight Function ---
def get_adaptive_blend_weight(forecast_idx, historical_data, productID=None):
    """Calculate adaptive blend weights based on data characteristics"""
    # Base decay factor
    base_weight = max(0.8 - (forecast_idx * 0.05), 0.4)
    
    # If we have limited historical data, rely more on seasonal patterns
    if len(historical_data) < 52:
        base_weight = max(0.7 - (forecast_idx * 0.08), 0.3)
    
    # Check data volatility using coefficient of variation
    if len(historical_data) > 0 and historical_data.mean() > 0:
        cv = historical_data.std() / historical_data.mean()
        
        # For highly volatile series like 22B, use more seasonal influence
        if cv > 0.5 or (productID and '22B' in productID):
            base_weight = max(0.6 - (forecast_idx * 0.1), 0.25)
            
            # For long-term forecasts with volatile data, rely even more on seasonal patterns
            if forecast_idx > 8:
                base_weight = max(0.4 - ((forecast_idx-8) * 0.05), 0.2)
    
    return base_weight

# --- Enhanced Create Seasonal Model ---
def create_seasonal_model(historical_df, productID=None):
    """Create an enhanced seasonal model with product-specific adaptations."""
    if historical_df.empty:
        return {}
    
    # Add quarter and week-of-quarter columns
    df = historical_df.copy()
    df['quarter'] = df.index.quarter
    df['week_of_quarter'] = ((df.index.isocalendar().week - 1) % 13) + 1
    
    # Create seasonal model
    seasonal_model = {}
    
    # Overall mean
    seasonal_model['overall_mean'] = df['Demand'].mean()
    
    # Quarter means
    quarter_means = df.groupby('quarter')['Demand'].mean()
    seasonal_model['quarter_means'] = quarter_means.to_dict()
    
    # Quarter factors (relative to overall mean)
    seasonal_model['quarter_factors'] = {
        q: (v / seasonal_model['overall_mean']) if seasonal_model['overall_mean'] > 0 else 1.0
        for q, v in seasonal_model['quarter_means'].items()
    }
    
    # Week-of-quarter patterns
    week_of_quarter_means = df.groupby('week_of_quarter')['Demand'].mean()
    seasonal_model['week_of_quarter_means'] = week_of_quarter_means.to_dict()
    
    # Week-of-quarter factors (relative to overall mean)
    seasonal_model['week_of_quarter_factors'] = {
        w: (v / seasonal_model['overall_mean']) if seasonal_model['overall_mean'] > 0 else 1.0
        for w, v in seasonal_model['week_of_quarter_means'].items()
    }
    
    # Combined quarter-week patterns
    combined_patterns = df.groupby(['quarter', 'week_of_quarter'])['Demand'].mean()
    seasonal_model['combined_patterns'] = combined_patterns.to_dict()
    
    # For 22B, use more recent data for trend calculation
    if productID and '22B' in productID:
        # Focus more on recent trends for volatile products
        recent_window = min(13, len(df) // 2) 
        recent_mean = df['Demand'].iloc[-recent_window:].mean()
        seasonal_model['recent_mean'] = recent_mean
        
        # Calculate trend based on weighted quarters
        if len(df) >= 52:
            quarters = df.groupby(pd.Grouper(freq='Q'))['Demand'].mean()
            if len(quarters) >= 4:
                # Use exponential weighting to emphasize recent quarters
                weights = np.exp(np.linspace(-2, 0, min(8, len(quarters))))[-min(8, len(quarters)):]
                weighted_quarters = quarters.iloc[-len(weights):].values * weights
                trend_factor = weighted_quarters.sum() / (weights.sum() * seasonal_model['overall_mean']) if seasonal_model['overall_mean'] > 0 else 1.0
                seasonal_model['trend_factor'] = trend_factor
            else:
                seasonal_model['trend_factor'] = recent_mean / seasonal_model['overall_mean'] if seasonal_model['overall_mean'] > 0 else 1.0
        else:
            seasonal_model['trend_factor'] = recent_mean / seasonal_model['overall_mean'] if seasonal_model['overall_mean'] > 0 else 1.0
    else:
        # Original approach for other products
        if len(df) >= 26:
            recent_mean = df['Demand'].iloc[-26:].mean()
            seasonal_model['recent_mean'] = recent_mean
            seasonal_model['trend_factor'] = recent_mean / seasonal_model['overall_mean'] if seasonal_model['overall_mean'] > 0 else 1.0
        else:
            seasonal_model['recent_mean'] = seasonal_model['overall_mean']
            seasonal_model['trend_factor'] = 1.0
    
    return seasonal_model

def get_seasonal_prediction(seasonal_model, date):
    """Get prediction from seasonal model for a specific date."""
    if not seasonal_model:
        return 0
    
    quarter = date.quarter
    week_of_quarter = ((date.isocalendar().week - 1) % 13) + 1
    
    # Try to get combined quarter-week pattern
    combined_key = (quarter, week_of_quarter)
    if combined_key in seasonal_model['combined_patterns']:
        base_prediction = seasonal_model['combined_patterns'][combined_key]
    else:
        # Fall back to separate quarter and week-of-quarter factors
        quarter_factor = seasonal_model['quarter_factors'].get(quarter, 1.0)
        week_factor = seasonal_model['week_of_quarter_factors'].get(week_of_quarter, 1.0)
        base_prediction = seasonal_model['overall_mean'] * quarter_factor * week_factor
    
    # Apply trend adjustment
    prediction = base_prediction * seasonal_model['trend_factor']
    
    return prediction

# --- XGBoost Forecasting Implementation with Ensemble Approach ---
def calculateXGBoostForecast(dataSeries, forecastDatesIndex, params, productID=None):
    """Generate forecasts using XGBoost model with enhanced ensemble approach."""
    if dataSeries.empty or len(forecastDatesIndex) == 0:
        return [0] * len(forecastDatesIndex)
    
    # Prepare historical data
    historical_df = pd.DataFrame({'Demand': dataSeries})
    
    # Handle outliers with product-specific adaptation
    historical_df['Demand'] = detectAndHandleOutliers(
        historical_df['Demand'], 
        threshold=params.get('spikeThreshold', 2.5),
        productID=productID
    )
    
    # Create enhanced features
    historical_features = create_features(historical_df, productID=productID)
    
    # Drop rows with NaN (from lag features)
    historical_features_clean = historical_features.dropna()
    
    if historical_features_clean.empty:
        return [0] * len(forecastDatesIndex)
    
    # Prepare X and y for training
    y_train = historical_features_clean['Demand']
    X_train = historical_features_clean.drop('Demand', axis=1)
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Enhanced XGBoost model with product-specific tweaks
    learning_rate = params.get('learning_rate', 0.05)
    if productID and '22B' in productID:
        # Slower learning rate and more trees for 22B to reduce overfitting
        learning_rate = learning_rate / 2
        n_estimators = params.get('n_estimators', 200) * 1.5
    else:
        n_estimators = params.get('n_estimators', 200)
    
    model = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=params.get('max_depth', 5),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        min_child_weight=params.get('min_child_weight', 3),
        gamma=params.get('gamma', 0),
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Create a seasonal model for blending
    seasonal_model = create_seasonal_model(historical_df, productID)
    
    # Prepare forecast dataframe
    forecast_df = pd.DataFrame(index=forecastDatesIndex)
    forecast_df['Demand'] = np.nan
    
    # Create a combined dataframe for iterative forecasting
    combined_df = pd.concat([historical_df, forecast_df])
    
    # Iteratively predict each point
    forecasts = []
    
    for i, forecast_date in enumerate(forecastDatesIndex):
        # Update features for the entire dataset
        features_df = create_features(combined_df, productID=productID)
        
        # Get the row for the current forecast date
        forecast_features = features_df.loc[[forecast_date]].copy()
        
        # If we have missing features, use seasonal model as fallback
        if forecast_features.isna().any(axis=1).iloc[0]:
            # Use seasonal model
            prediction = get_seasonal_prediction(seasonal_model, forecast_date)
        else:
            # Fill any remaining NaN values with mean
            for col in forecast_features.columns:
                if forecast_features[col].isna().any():
                    if col in X_train.columns:
                        forecast_features[col] = X_train[col].mean()
                    else:
                        forecast_features[col] = 0
            
            # Keep only columns that were in the training data
            forecast_features = forecast_features[X_train.columns]
            
            # Scale features
            forecast_features_scaled = pd.DataFrame(
                scaler.transform(forecast_features),
                columns=forecast_features.columns,
                index=forecast_features.index
            )
            
            # Make prediction with XGBoost
            xgb_prediction = model.predict(forecast_features_scaled)[0]
            
            # Get seasonal prediction for blending
            seasonal_prediction = get_seasonal_prediction(seasonal_model, forecast_date)
            
            # Use enhanced adaptive blending
            blend_weight = get_adaptive_blend_weight(
                forecast_idx=i, 
                historical_data=historical_df['Demand'],
                productID=productID
            )
            
            prediction = blend_weight * xgb_prediction + (1 - blend_weight) * seasonal_prediction
        
        # Ensure non-negative forecast
        prediction = max(0, prediction)
        
        # Store prediction
        forecasts.append(prediction)
        
        # Update the combined dataframe with the prediction
        combined_df.loc[forecast_date, 'Demand'] = prediction
    
    return forecasts

# --- Backtesting and Parameter Optimization ---
def backtest(dataFrame, params, backtestPeriods=13):
    """Run backtest with specified parameters on weekly data."""
    if len(dataFrame) <= backtestPeriods: 
        return {"error": "Not enough data"}
    
    train = dataFrame.iloc[:-backtestPeriods].copy()
    test = dataFrame.iloc[-backtestPeriods:].copy()
    actualValues = test['Demand'].values
    valid_indices = ~np.isnan(actualValues)
    
    if not np.any(valid_indices): 
        return {"error": "No valid actual values"}
    
    actualValues_valid = actualValues[valid_indices]
    test_index_valid = test.index[valid_indices]

    forecastValues_all = calculateXGBoostForecast(
        dataSeries=train['Demand'],
        forecastDatesIndex=test.index,
        params=params,
        productID=None  # No specific product ID during backtesting
    )
    
    forecastValues_valid = np.array(forecastValues_all)[valid_indices]
    metrics = calculateForecastAccuracy(actualValues_valid, forecastValues_valid)
    
    return {
        'metrics': metrics, 
        'actual': actualValues_valid, 
        'forecast': forecastValues_valid, 
        'periods': test_index_valid.tolist()
    }

def optimizeParameters(dataFrame, paramsGrid=None):
    """Optimize XGBoost parameters using grid search."""
    if paramsGrid is None:
        paramsGrid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'spikeThreshold': [2.5]
        }
    
    if len(dataFrame) < 52 + 13:
        warnings.warn("Not enough data for optimization. Using defaults.")
        return {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'spikeThreshold': 2.5
        }
    
    bestMape = float('inf')
    bestParams = None
    backtestPeriods = min(13, len(dataFrame) // 4)
    
    if backtestPeriods < 4:
        warnings.warn("Short backtest period.")
        return {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'spikeThreshold': 2.5
        }
    
    # Simplified grid search to avoid excessive computation
    for n_estimators in paramsGrid['n_estimators']:
        for learning_rate in paramsGrid['learning_rate']:
            for max_depth in paramsGrid['max_depth']:
                for min_child_weight in paramsGrid['min_child_weight']:
                    currentParams = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'min_child_weight': min_child_weight,
                        'gamma': paramsGrid['gamma'][0],
                        'subsample': paramsGrid['subsample'][0],
                        'colsample_bytree': paramsGrid['colsample_bytree'][0],
                        'spikeThreshold': paramsGrid['spikeThreshold'][0]
                    }
                    
                    backtestResults = backtest(dataFrame, params=currentParams, backtestPeriods=backtestPeriods)
                    
                    if 'error' not in backtestResults:
                        currentMape = backtestResults['metrics'].get('mape', float('inf'))
                        if pd.notna(currentMape) and currentMape < bestMape:
                            bestMape = currentMape
                            bestParams = currentParams
    
    if bestParams:
        print(f"Optimal params: {bestParams} (MAPE: {bestMape:.2f}%)")
        return bestParams
    else:
        warnings.warn("Optimization failed. Using defaults.")
        return {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'spikeThreshold': 2.5
        }

# --- Main Forecasting Function ---
def makeForecast(productsToForecast, forecastWeeks, weeklyData, runBacktest=True, spikeThreshold=2.5):
    """Generate weekly forecasts for specified products using XGBoost."""
    productList = []
    
    for productID in productsToForecast:
        print(f"\n--- Processing {productID} (Weekly XGBoost) ---")
        productTimeSeries = processHistoricalData(weeklyData, productID)
        
        if productTimeSeries.empty or productTimeSeries['Demand'].isnull().all():
            print(f"No valid weekly data for {productID}")
            product = Product(productID, [], [])
            product.forecast = [0] * forecastWeeks
            productList.append(product)
            continue
        
        product = Product(
            productID=productID, 
            dates=productTimeSeries.index.tolist(), 
            demand=productTimeSeries['Demand'].tolist()
        )
        
        # Product-specific parameter adjustments
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'spikeThreshold': spikeThreshold
        }
        
        # Specific adjustments for product 22B
        if productID == '22B':
            default_params.update({
                'learning_rate': 0.01,  # Lower learning rate
                'max_depth': 3,         # Less complex trees to avoid overfitting
                'min_child_weight': 5,  # More conservative to reduce overfitting
                'gamma': 0.1,           # More regularization
                'spikeThreshold': 2.0   # More aggressive outlier handling
            })
        
        if runBacktest and len(productTimeSeries) >= 52 + 13:
            print(f"Running optimization for {productID}")
            
            # For 22B, use more conservative parameter grid
            if productID == '22B':
                params_grid = {
                    'n_estimators': [200, 300, 400],
                    'learning_rate': [0.005, 0.01, 0.02],
                    'max_depth': [2, 3, 4],
                    'min_child_weight': [3, 5, 7],
                    'gamma': [0.1, 0.2],
                    'subsample': [0.7, 0.8],
                    'colsample_bytree': [0.7, 0.8],
                    'spikeThreshold': [2.0]
                }
                product.bestParams = optimizeParameters(productTimeSeries, params_grid)
            else:
                product.bestParams = optimizeParameters(productTimeSeries)
            
            backtestResults = backtest(
                productTimeSeries, 
                params=product.bestParams, 
                backtestPeriods=min(13, len(productTimeSeries) // 4)
            )
            
            if 'error' not in backtestResults:
                product.backtestMetrics = backtestResults['metrics']
                print(f"  Backtest MAPE: {product.backtestMetrics.get('mape', 'N/A'):.2f}%")
            else:
                print(f"  Backtest error: {backtestResults['error']}")
                
            if not product.bestParams:
                product.bestParams = default_params
        else:
            product.bestParams = default_params
            if len(productTimeSeries) < 52 + 13:
                print("  Skipping backtest/optimization: insufficient data.")
        
        try:
            lastDate = productTimeSeries.index[-1]
            forecastIndex = pd.date_range(
                start=lastDate + pd.Timedelta(weeks=1), 
                periods=forecastWeeks, 
                freq='W'
            )
            product.forecastDates = forecastIndex.tolist()

            product.forecast = calculateXGBoostForecast(
                dataSeries=productTimeSeries['Demand'],
                forecastDatesIndex=forecastIndex,
                params=product.bestParams,
                productID=productID
            )
            
            print(f"  Generated {len(product.forecast)}-week forecast.")
        except Exception as e:
            print(f"Error generating forecast for {productID}: {e}")
            import traceback
            traceback.print_exc()
            product.forecast = [0] * forecastWeeks
            product.forecastDates = [pd.NaT] * forecastWeeks
        
        productList.append(product)
    
    return productList

def formatResults(productList):
    results = []
    for product in productList:
        formattedDates = [d.strftime('%Y-%m-%d') if pd.notna(d) else 'N/A' for d in product.forecastDates]
        productResult = {
            "productID": product.productID, 
            "forecast": product.forecast, 
            "forecastDates": formattedDates, 
            "parameters": product.bestParams, 
            "backtestMetrics": product.backtestMetrics
        }
        results.append(productResult)
    return results

def generateDemandForecast(productIDs, forecastHorizonWeeks, historicalWeeklyData, spikeThreshold=2.5):
    if not isinstance(productIDs, list): 
        productIDs = [productIDs]
    if not isinstance(forecastHorizonWeeks, int) or forecastHorizonWeeks < 1: 
        forecastHorizonWeeks = 13
    
    productResultsList = makeForecast(
        productsToForecast=productIDs, 
        forecastWeeks=forecastHorizonWeeks, 
        weeklyData=historicalWeeklyData, 
        runBacktest=True, 
        spikeThreshold=spikeThreshold
    )
    
    return productResultsList  # Return list of Product objects
