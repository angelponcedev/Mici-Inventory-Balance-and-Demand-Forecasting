import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset # Import DateOffset

# --- (Keep the get_week_start_date function as defined previously) ---
def get_week_start_date(year, quarter_num, week_num):
    """Calculates the start date of a specific week within a quarter."""
    if pd.isna(year) or pd.isna(quarter_num) or pd.isna(week_num):
        return pd.NaT # Return Not-a-Time for invalid inputs
    try:
        year = int(year)
        quarter_num = int(quarter_num)
        week_num = int(week_num)
        if not 1 <= quarter_num <= 4: raise ValueError("Quarter must be 1-4")
        if not 1 <= week_num <= 13: raise ValueError("Week must be 1-13")
        month = (quarter_num - 1) * 3 + 1
        quarter_start_date = pd.Timestamp(year=year, month=month, day=1)
        week_start_date = quarter_start_date + DateOffset(weeks=(week_num - 1))
        return week_start_date
    except Exception as e:
        print(f"Error calculating date for Y={year}, Q={quarter_num}, Wk={week_num}: {e}")
        return pd.NaT

def processDataset(input_data, weeklyDemandRatioInput):
    """
    Processes quarterly demand data, converts it to weekly demand based on
    a provided ratio, adds week start dates, filters out zero-demand quarters,
    and returns a sorted weekly demand DataFrame.

    Args:
        input_data: DataFrame, path, or dict with demand data.
        weeklyDemandRatioInput: DataFrame (with ratios in row 0, cols Wk1-Wk13),
                                list, or array-like with 13 weekly ratios.

    Returns:
        pd.DataFrame: Long-format DataFrame with 'Product ID', 'Quarter',
                      'Week', 'Date', 'Weekly Demand'.
        None: If errors occur.
    """

    # --- Process and Validate weeklyDemandRatio ---
    # (Keep the existing validation logic from the previous version)
    weeklyDemandRatio_np = None
    try:
        temp_ratio = weeklyDemandRatioInput
        if isinstance(temp_ratio, pd.DataFrame):
            print("Input ratio is a DataFrame, extracting first row (Wk1-Wk13)...")
            week_columns = [f'Wk{i}' for i in range(1, 14)]
            if not all(col in temp_ratio.columns for col in week_columns):
                 raise ValueError(f"Input DataFrame ratio missing one or more columns: {week_columns}")
            if temp_ratio.empty: raise ValueError("Input DataFrame ratio is empty.")
            temp_ratio = temp_ratio.loc[temp_ratio.index[0], week_columns]
            print("Ratio extracted as Series.")
        if not hasattr(temp_ratio, '__len__') or len(temp_ratio) != 13:
            raise ValueError("Weekly ratio must have exactly 13 elements.")
        weeklyDemandRatio_np = np.array(temp_ratio, dtype=float)
        if np.isnan(weeklyDemandRatio_np).any():
             raise ValueError("Weekly ratio contains non-numeric or NaN values.")
        if not np.isclose(weeklyDemandRatio_np.sum(), 1.0, atol=1e-6):
             print(f"Warning: Sum of ratios ({weeklyDemandRatio_np.sum():.4f}) is not close to 1.0.")
    except Exception as e:
        print(f"Error processing or validating weeklyDemandRatio: {e}")
        return None

    # --- Load Data ---
    # (Keep the existing loading logic)
    try:
        if isinstance(input_data, pd.DataFrame): df = input_data.copy()
        elif isinstance(input_data, str):
            if input_data.endswith('.xlsx'): df = pd.read_excel(input_data)
            elif input_data.endswith('.csv'): df = pd.read_csv(input_data)
            else: raise ValueError("Unsupported file type.")
        else: df = pd.DataFrame(input_data)
    except Exception as e:
        print(f"Error loading/creating DataFrame: {e}")
        return None

    # --- Filtering Rows ---
    # (Keep the existing filtering logic for 'EffectiveDemand')
    target_attribute_col = 'Attribute'
    target_id_col = 'Product ID'
    try:
        if target_attribute_col not in df.columns or target_id_col not in df.columns:
            missing_cols = [col for col in [target_attribute_col, target_id_col] if col not in df.columns]
            raise KeyError(f"Missing essential columns: {missing_cols}")
        filteredDF = df[df[target_attribute_col] == "EffectiveDemand"].copy()
        if filteredDF.empty:
            print("DataFrame is empty after filtering for 'EffectiveDemand'.")
            return None
    except KeyError as e:
        print(f"Error during filtering: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during filtering: {e}")
        return None

    # --- Reshaping (Melting) ---
    # (Keep the existing melting logic)
    id_vars = [target_id_col, target_attribute_col]
    value_vars = [col for col in filteredDF.columns if col not in id_vars]
    if not value_vars:
        print("Error: No value columns (Quarters) found to melt.")
        return None
    try:
        long_demand_df = pd.melt(
            filteredDF, id_vars=id_vars, value_vars=value_vars,
            var_name='Quarter', value_name='Quarterly Demand'
        )
    except Exception as e:
        print(f"Error during melting: {e}")
        return None

    # --- Prepare for Sorting (Extract Year/Quarter Num) ---
    # (Keep the existing extraction and sorting logic)
        # --- Prepare for Sorting & Dates (Extract Year/Quarter Num) ---
    parse_errors = False # Flag to track if parsing issues occurred
    try:

        # --- NEW Regex for "Qx yy" format ---
        # Extract Quarter Number (the digit right after 'Q')
        long_demand_df['Quarter_Str'] = long_demand_df['Quarter'].str.extract(r'Q(\d)')
        # Extract Year Number (the two digits after the space at the end)
        long_demand_df['Year_Str'] = long_demand_df['Quarter'].str.extract(r' (\d{2})$')

        # Check for initial parsing failures (NaNs after extract)
        initial_parse_failed = long_demand_df['Year_Str'].isnull() | long_demand_df['Quarter_Str'].isnull()
        if initial_parse_failed.any():
            # Get examples of failed quarter strings for better debugging
            failed_examples = long_demand_df.loc[initial_parse_failed, 'Quarter'].unique()
            print(f"Warning: Could not parse Year/Quarter for some rows. Check 'Quarter' column format. Expected 'Qx yy'. Examples of failed formats: {failed_examples[:5]}") # Show first 5 unique failed formats
            parse_errors = True
            # Fill NaNs resulting from extraction failure before numeric conversion
            long_demand_df.loc[initial_parse_failed, ['Year_Str', 'Quarter_Str']] = '-1' # Use .loc for safety

        # Convert extracted strings to numeric
        long_demand_df['Year'] = pd.to_numeric(long_demand_df['Year_Str'], errors='coerce')
        long_demand_df['Quarter_Num'] = pd.to_numeric(long_demand_df['Quarter_Str'], errors='coerce')

        # Check for numeric conversion failures (NaNs after to_numeric)
        numeric_conversion_failed = long_demand_df['Year'].isnull() | long_demand_df['Quarter_Num'].isnull()
        # Avoid double warning if initial parse already failed and filled -1
        rows_failed_numeric_only = numeric_conversion_failed & (~initial_parse_failed)
        if rows_failed_numeric_only.any():
             if not parse_errors: # Only print this warning if the initial parse seemed okay
                 print("Warning: Could not convert extracted Year/Quarter strings to numbers (unexpected characters?).")
             parse_errors = True
             # Fill NaNs resulting from numeric conversion failure
             long_demand_df.loc[rows_failed_numeric_only, ['Year', 'Quarter_Num']] = -1 # Use .loc

        # Calculate Full_Year (handles 2-digit and 4-digit years)
        pivot_year = 50 # Assumes yy >= 50 is 19yy, yy < 50 is 20yy
        long_demand_df['Full_Year'] = np.select(
            [
                (long_demand_df['Year'] >= pivot_year) & (long_demand_df['Year'] < 100),
                (long_demand_df['Year'] < pivot_year) & (long_demand_df['Year'] >= 0)
            ],
            [
                1900 + long_demand_df['Year'],
                2000 + long_demand_df['Year']
            ],
            default=long_demand_df['Year'] # Assumes 4-digit or -1
        )
        # Ensure Quarter_Num is integer, keeping -1 as is
        # Use nullable integer type Int64 to handle potential NaNs before fillna
        long_demand_df['Quarter_Num'] = long_demand_df['Quarter_Num'].astype('Int64').fillna(-1)


        # Sort
        long_demand_df_sorted = long_demand_df.sort_values(
            by=[target_id_col, 'Full_Year', 'Quarter_Num']
        )

        # Drop intermediate columns
        long_demand_df_sorted = long_demand_df_sorted.drop(
            columns=['Year', 'Year_Str', 'Quarter_Str', 'Attribute']
        )
        long_demand_df_sorted = long_demand_df_sorted.reset_index(drop=True)

    except Exception as e:
        print(f"Error during sorting or year/quarter extraction: {e}")
        # Optionally re-raise or print traceback for more detail
        # import traceback
        # traceback.print_exc()
        return None

    # --- *** NEW STEP: Filter out rows with zero quarterly demand *** ---
    print(f"Rows before filtering zero demand: {len(long_demand_df_sorted)}")
    # Also handle potential NaN values in 'Quarterly Demand' just in case
    long_demand_df_filtered = long_demand_df_sorted[
        (long_demand_df_sorted['Quarterly Demand'].notna()) &
        (long_demand_df_sorted['Quarterly Demand'] != 0)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning later
    print(f"Rows after filtering zero demand: {len(long_demand_df_filtered)}")

    if long_demand_df_filtered.empty:
        print("DataFrame is empty after filtering out zero demand quarters.")
        return None
    # --- End of New Step ---

    # --- Converting Quarter Demand to Weekly Demand ---
    # Now iterate over the *filtered* DataFrame
    weekly_data_list = []
    num_weeks = len(weeklyDemandRatio_np)

    print("Converting quarterly demand to weekly demand...")
    try:
        # *** Use long_demand_df_filtered here ***
        for index, row in long_demand_df_filtered.iterrows():
            product_id = row[target_id_col]
            quarter_str = row['Quarter']
            quarterly_demand = row['Quarterly Demand'] # Will not be 0 or NaN here
            year = row['Full_Year']
            q_num = row['Quarter_Num']

            # Simplified check as we've already filtered NaNs and 0s for demand
            if pd.isna(year) or year == -1 or pd.isna(q_num) or q_num == -1:
                 print(f"Warning: Skipping row {index} due to invalid Year/Quarter: Year={year}, Q={q_num}")
                 continue

            for week_num in range(1, num_weeks + 1):
                ratio_for_week = weeklyDemandRatio_np[week_num - 1]
                weekly_demand_value = quarterly_demand * ratio_for_week
                week_start_dt = get_week_start_date(year, q_num, week_num)

                weekly_data_list.append({
                    'Product ID': product_id,
                    'Quarter': quarter_str,
                    'Week': week_num,
                    'Date': week_start_dt,
                    'Weekly Demand': weekly_demand_value,
                })

        if not weekly_data_list:
             print("No weekly data generated after filtering. Check input data.")
             # This might happen if all quarterly demands were zero
             return None

        weekly_demand_df = pd.DataFrame(weekly_data_list)
        weekly_demand_df['Date'] = pd.to_datetime(weekly_demand_df['Date'])

    except Exception as e:
        print(f"Error during weekly demand conversion or date calculation: {e}")
        return None

    print("Weekly demand conversion and date calculation complete.")
    return weekly_demand_df