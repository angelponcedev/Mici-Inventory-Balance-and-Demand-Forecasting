import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

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


def makeForecast(productsToForecast, periodsToForecast, periodDemand):

    """
    Generate forecasts for specified products with enhanced spike detection.
    
    Args:
        productsToForecast: List of product codes to forecast
        periodsToForecast: Number of quarters to forecast
        periodDemand: DataFrame with historical demand data
        runBacktest: Whether to perform backtesting
        spikeThreshold: Threshold for spike detection (higher = less sensitive)
        
    Returns:
        List of Product objects with forecasts
    """
    
    productList = []
    
    # Process each product
    for productID in productsToForecast:
        print(f"\n--- Processing {productID} ---")
        
        # Process data for this product
        productData = processHistoricalData(periodDemand, productID)
        
        if productData.empty:
            print(f"No valid data found for {productID}")
            # Create empty product
            product = Product(productID, [], [])
            product.forecast = [0] * periodsToForecast
            productList.append(product)
            continue
        
        # Store historical data
        product = Product(
            productID=productID,
            quarter=productData.index.tolist(),
            demand=productData['Demand'].tolist()
        )
        
        productList.append(product)
        
    for product in productList:
        print(f"Product ID:{product.productID}\nQuarter:{product.quarter}\nDemand{product.demand}\n")
        
    return productList

def processHistoricalData(rawData, productID):
    """Process historical data with improved type handling."""
    if 'Demand Value' not in rawData.columns and 'Effective Demand Attribute' in rawData.columns:
        rawData['Demand Value'] = rawData['Effective Demand Attribute']
    
    # Filter to this product
    productData = rawData[rawData['Product ID'] == productID].copy()
    
    if productData.empty:
        return pd.DataFrame(columns=['Demand'])
    
    # Parse quarters
    productData['Quarter Period'] = productData['Quarter'].apply(parseCustomQuarter)
    productData = productData.dropna(subset=['Quarter Period'])
    
    # Convert to numeric explicitly, coercing errors to NaN
    productData['Demand Value'] = pd.to_numeric(productData['Demand Value'], errors='coerce')
    
    # Drop rows with NaN demand values
    productData = productData.dropna(subset=['Demand Value'])
    
    # Create demand time series
    ts = pd.Series(
        productData['Demand Value'].values, 
        index=pd.PeriodIndex(productData['Quarter Period'], freq='Q')
    )
    ts = ts.sort_index()
    
    # Remove initial zeros
    nonZeroIndex = ts[ts > 0].index
    if not len(nonZeroIndex) == 0:
        firstNonZero = nonZeroIndex[0]
        ts = ts.loc[firstNonZero:]
        
    # Handle duplicate periods by taking the latest value
    ts = ts[~ts.index.duplicated(keep='last')]
    
    return pd.DataFrame({'Demand': ts})

def parseCustomQuarter(qStr):
    """Parses quarter strings like 'Q2 95' into pandas Period objects."""
    try:
        qStr = str(qStr).strip()
        parts = qStr.split()
        if len(parts) != 2 or not parts[0].startswith('Q') or not parts[0][1:].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Invalid format: '{qStr}'")
        
        quarterNum = parts[0][1:]
        yearShort = int(parts[1])
        year = 1900 + yearShort if yearShort >= 50 else 2000 + yearShort
        
        return pd.Period(f"{year}Q{quarterNum}", freq='Q')
    except Exception as e:
        warnings.warn(f"Could not parse quarter string '{qStr}': {e}. Returning NaT.")
        return pd.NaT