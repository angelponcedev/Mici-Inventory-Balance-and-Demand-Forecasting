import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset

# --- Product Class Definition ---
# (Keep Product class as is)
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
# (Keep calculateForecastAccuracy, detectAndHandleOutliers, detectTrend,
#  calculateWeeklySeasonalFactors, processHistoricalData as they are
#  from the previous correct weekly version)
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

def detectAndHandleOutliers(series, threshold=2.5):
    if len(series) < 12: return series
    try:
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce').dropna()
            if len(series) < 12: return series
    except Exception as e: print(f"Error converting series: {e}"); return series
    adjusted = series.copy(); median = adjusted.median()
    mad = np.median(np.abs(adjusted - median)); madScale = 3
    if mad > 0:
        zScores = madScale * np.abs(adjusted - median) / mad
        outliers = zScores > threshold
        if outliers.any():
            upperBound = median + (threshold * mad * madScale)
            adjusted[outliers] = upperBound
    return adjusted

def detectTrend(series):
    if len(series) < 12: return 'none'
    midpoint = len(series) // 2
    firstHalf = series.iloc[:midpoint].mean(); secondHalf = series.iloc[midpoint:].mean()
    if firstHalf == 0: return 'none'
    percentChange = (secondHalf - firstHalf) / firstHalf * 100
    if percentChange > 20: return 'up'
    elif percentChange < -20: return 'down'
    else: return 'none'

def calculateWeeklySeasonalFactors(series):
    MIN_YEARS_FOR_WEEKLY_SEASONALITY = 2
    if len(series) < 52 * MIN_YEARS_FOR_WEEKLY_SEASONALITY:
        warnings.warn(f"Not enough data ({len(series)} weeks) for reliable weekly seasonality. Returning flat factors.")
        return pd.Series(1.0, index=range(1, 54))
    if not isinstance(series.index, pd.DatetimeIndex): raise ValueError("Index must be DatetimeIndex")
    weekNums = series.index.isocalendar().week.astype(int)
    weeklyValues = {w: series[weekNums == w].median() for w in range(1, 54) if not series[weekNums == w].empty}
    overallMedian = series.median()
    if overallMedian == 0: warnings.warn("Overall median is zero..."); return pd.Series(1.0, index=range(1, 54))
    for w in range(1, 54): weeklyValues.setdefault(w, overallMedian)
    factors = pd.Series({w: val / overallMedian for w, val in weeklyValues.items()})
    factors = factors.clip(0.6, 1.8)
    num_weeks = len(factors)
    if factors.sum() > 0: factors = factors * (num_weeks / factors.sum())
    else: factors = pd.Series(1.0, index=range(1, 54))
    return factors

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
    ts_resampled = ts.resample('W').last() # Use 'W' or 'W-SUN' etc.
    nonZeroIndex = ts_resampled[ts_resampled > 0].index
    if not nonZeroIndex.empty: ts_resampled = ts_resampled.loc[nonZeroIndex[0]:]
    return pd.DataFrame({'Demand': ts_resampled})

# --- Core Forecasting Algorithm (Weekly) ---
def calculateCustomForecast(dataSeries, forecastDatesIndex, params):
    """Enhanced weekly forecasting algorithm."""
    if dataSeries.empty:
        return [0] * len(forecastDatesIndex)

    # Ensure dataSeries is numeric
    try:
        numericSeries = pd.to_numeric(dataSeries, errors='coerce').dropna()
        if numericSeries.empty:
            print("Warning: No numeric values found in data series.")
            return [0] * len(forecastDatesIndex)
        dataSeries = numericSeries
    except Exception as e:
        print(f"Error ensuring numeric series: {e}")
        return [0] * len(forecastDatesIndex)

    # Extract parameters
    numRecentWeeks = params.get('numRecentWeeks', 26)
    blendAlpha = params.get('blendAlpha', 0.6)
    spikeThreshold = params.get('spikeThreshold', 2.5)

    # Prepare data
    adjustedSeries = detectAndHandleOutliers(dataSeries, threshold=spikeThreshold)
    trendPattern = detectTrend(adjustedSeries)
    seasonalFactors = calculateWeeklySeasonalFactors(adjustedSeries)

    # Adjust parameters based on trend
    if trendPattern == 'down':
        numRecentWeeks = min(numRecentWeeks, 13)
        blendAlpha = 0.8
    elif trendPattern == 'up':
        numRecentWeeks = min(numRecentWeeks, 52)
        blendAlpha = 0.6

    # Limit recent weeks
    nRecent = min(numRecentWeeks, len(adjustedSeries))
    recentData = adjustedSeries.iloc[-nRecent:]

    # Check for recent spike
    recentValues = adjustedSeries.iloc[-4:] if len(adjustedSeries) >= 4 else adjustedSeries
    isRecentSpike = False
    previousMedian = np.nan # Initialize previousMedian
    if not recentValues.empty:
        mostRecent = recentValues.iloc[-1]
        if len(recentValues) > 1:
            previousMedian = recentValues.iloc[:-1].median()
            if pd.notna(previousMedian): # Check if median calculation was successful
                 isRecentSpike = mostRecent > (previousMedian * (1 + (spikeThreshold / 5)))
        # If only one recent value, cannot determine spike relative to previous
        # else: previousMedian remains NaN

    forecasts = []
    weightedAvg = 0
    if nRecent > 0:
        weights = np.exp(np.linspace(-1.5, 0, nRecent))
        weights /= weights.sum()
        weightedAvg = (recentData * weights).sum()

    # --- Forecasting Loop ---
    for i, date in enumerate(forecastDatesIndex):
        # Ensure date is a Timestamp if it's not already (might be needed if index type changes)
        if not isinstance(date, pd.Timestamp):
             date = pd.Timestamp(date)

        weekNum = date.isocalendar().week
        seasonalFactor = seasonalFactors.get(weekNum, 1.0)
        calculatedValue = weightedAvg * seasonalFactor

        # Calculate date for same week last year
        dateLastYear = date - pd.DateOffset(weeks=52)
        lastYearValue = None

        # *** FIXED LINE: Use 'not in' instead of '.contains()' ***
        if dateLastYear not in dataSeries.index:
            # If exact date not found, look for nearest within tolerance
            nearest_idx = dataSeries.index.get_indexer([dateLastYear], method='nearest', tolerance=pd.Timedelta('3 days'))
            valid_nearest = nearest_idx[nearest_idx != -1]
            if len(valid_nearest) > 0:
                lastYearValue = dataSeries.iloc[valid_nearest[0]]
        else:
            # Exact date found
            lastYearValue = dataSeries[dateLastYear]

        # Blend calculated value with last year's value
        if lastYearValue is not None and pd.notna(lastYearValue):
            forecastValue = blendAlpha * calculatedValue + (1 - blendAlpha) * lastYearValue
        else:
            # Use only calculated value if no valid data from last year
            forecastValue = calculatedValue

        # Adjust forecast down if immediately following a detected spike
        if isRecentSpike and i < 2 and pd.notna(previousMedian): # Check previousMedian is valid
            spikeRecoveryFactor = 0.6
            # Adjust towards the *previous* median level (times seasonality)
            adjustedValue = (forecastValue * (1 - spikeRecoveryFactor) +
                           previousMedian * seasonalFactor * spikeRecoveryFactor)
            forecastValue = adjustedValue

        # Ensure forecast is non-negative
        forecasts.append(max(0, forecastValue))

    return forecasts

# --- Backtesting and Parameter Optimization (Weekly) ---
def backtest(dataFrame, params, backtestPeriods=13):
    """Run backtest with specified parameters on weekly data."""
    if len(dataFrame) <= backtestPeriods: return {"error": "Not enough data"}
    train = dataFrame.iloc[:-backtestPeriods].copy()
    test = dataFrame.iloc[-backtestPeriods:].copy()
    actualValues = test['Demand'].values
    valid_indices = ~np.isnan(actualValues)
    if not np.any(valid_indices): return {"error": "No valid actual values"}
    actualValues_valid = actualValues[valid_indices]
    test_index_valid = test.index[valid_indices]

    # *** CORRECT THE ARGUMENT NAME HERE ***
    forecastValues_all = calculateCustomForecast(
        dataSeries=train['Demand'],
        forecastDatesIndex=test.index, # Use forecastDatesIndex
        params=params
    )
    forecastValues_valid = np.array(forecastValues_all)[valid_indices]
    metrics = calculateForecastAccuracy(actualValues_valid, forecastValues_valid)
    return {'metrics': metrics, 'actual': actualValues_valid, 'forecast': forecastValues_valid, 'periods': test_index_valid.tolist()}

# (Keep optimizeParameters as is - it correctly calls the fixed backtest function)
def optimizeParameters(dataFrame, paramsGrid=None):
    if paramsGrid is None:
        paramsGrid = {'numRecentWeeks': [13, 26, 52], 'blendAlpha': [0.4, 0.6, 0.8], 'spikeThreshold': [2.0, 2.5, 3.0]}
    if len(dataFrame) < 52 + 13:
        warnings.warn("Not enough data for optimization. Using defaults.")
        return {'numRecentWeeks': 26, 'blendAlpha': 0.6, 'spikeThreshold': 2.5}
    bestMape = float('inf'); bestParams = None
    backtestPeriods = min(13, len(dataFrame) // 4)
    if backtestPeriods < 4:
         warnings.warn("Short backtest period."); return {'numRecentWeeks': 26, 'blendAlpha': 0.6, 'spikeThreshold': 2.5}
    for numRecent in paramsGrid['numRecentWeeks']:
        for blendAlpha in paramsGrid['blendAlpha']:
            for spikeThreshold in paramsGrid['spikeThreshold']:
                if numRecent >= len(dataFrame) - backtestPeriods: continue
                currentParams = {'numRecentWeeks': numRecent, 'blendAlpha': blendAlpha, 'spikeThreshold': spikeThreshold}
                backtestResults = backtest(dataFrame, params=currentParams, backtestPeriods=backtestPeriods)
                if 'error' not in backtestResults:
                    currentMape = backtestResults['metrics'].get('mape', float('inf'))
                    if pd.notna(currentMape) and currentMape < bestMape: bestMape = currentMape; bestParams = currentParams
    if bestParams: print(f"Optimal params: {bestParams} (MAPE: {bestMape:.2f}%)"); return bestParams
    else: warnings.warn("Optimization failed. Using defaults."); return {'numRecentWeeks': 26, 'blendAlpha': 0.6, 'spikeThreshold': 2.5}


# --- Main Forecasting Function (Weekly) ---
def makeForecast(productsToForecast, forecastWeeks, weeklyData, runBacktest=True, spikeThreshold=2.5):
    """Generate weekly forecasts for specified products."""
    productList = []
    for productID in productsToForecast:
        print(f"\n--- Processing {productID} (Weekly) ---")
        productTimeSeries = processHistoricalData(weeklyData, productID)
        if productTimeSeries.empty or productTimeSeries['Demand'].isnull().all():
            print(f"No valid weekly data for {productID}"); product = Product(productID, [], []); product.forecast = [0] * forecastWeeks; productList.append(product); continue
        product = Product(productID=productID, dates=productTimeSeries.index.tolist(), demand=productTimeSeries['Demand'].tolist())
        if runBacktest and len(productTimeSeries) >= 52 + 13:
            print(f"Running optimization for {productID}"); product.bestParams = optimizeParameters(productTimeSeries)
            backtestResults = backtest(productTimeSeries, params=product.bestParams, backtestPeriods=min(13, len(productTimeSeries) // 4))
            if 'error' not in backtestResults: product.backtestMetrics = backtestResults['metrics']; print(f"  Backtest MAPE: {product.backtestMetrics.get('mape', 'N/A'):.2f}%")
            else: print(f"  Backtest error: {backtestResults['error']}")
            if not product.bestParams: product.bestParams = {'numRecentWeeks': 26, 'blendAlpha': 0.6, 'spikeThreshold': spikeThreshold}
        else:
            product.bestParams = {'numRecentWeeks': 26, 'blendAlpha': 0.6, 'spikeThreshold': spikeThreshold}
            if len(productTimeSeries) < 52 + 13: print("  Skipping backtest/optimization: insufficient data.")
        try:
            lastDate = productTimeSeries.index[-1]
            forecastIndex = pd.date_range(start=lastDate + pd.Timedelta(weeks=1), periods=forecastWeeks, freq='W')
            product.forecastDates = forecastIndex.tolist()

            # *** CORRECT THE ARGUMENT NAME HERE ***
            product.forecast = calculateCustomForecast(
                dataSeries=productTimeSeries['Demand'],
                forecastDatesIndex=forecastIndex, # Use forecastDatesIndex
                params=product.bestParams
            )
            print(f"  Generated {len(product.forecast)}-week forecast.")
        except Exception as e:
            print(f"Error generating forecast for {productID}: {e}"); import traceback; traceback.print_exc()
            product.forecast = [0] * forecastWeeks; product.forecastDates = [pd.NaT] * forecastWeeks
        productList.append(product)
    return productList

# (Keep formatResults as is)
def formatResults(productList):
    results = []
    for product in productList:
        formattedDates = [d.strftime('%Y-%m-%d') if pd.notna(d) else 'N/A' for d in product.forecastDates]
        productResult = {"productID": product.productID, "forecast": product.forecast, "forecastDates": formattedDates, "parameters": product.bestParams, "backtestMetrics": product.backtestMetrics}
        results.append(productResult)
    return results

# (Keep generateCustomDemandForecast interface as is - it calls the fixed makeForecast)
def generateCustomDemandForecast(productIDs, forecastHorizonWeeks, historicalWeeklyData, spikeThreshold=2.5):
    if not isinstance(productIDs, list): productIDs = [productIDs]
    if not isinstance(forecastHorizonWeeks, int) or forecastHorizonWeeks < 1: forecastHorizonWeeks = 13
    productResultsList = makeForecast(productsToForecast=productIDs, forecastWeeks=forecastHorizonWeeks, weeklyData=historicalWeeklyData, runBacktest=True, spikeThreshold=spikeThreshold)
    return productResultsList # Return list of Product objects
