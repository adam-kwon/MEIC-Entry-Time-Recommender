import logging
import warnings

# FORCE silence before ANY other imports happen
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Prevent the "streamlit run" suggestion
import streamlit.runtime.scriptrunner_utils.script_run_context as src
logging.getLogger(src.__name__).setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from other.helpers import parse_currency_column, parse_dollar

def load_and_parse_csv(uploaded_files_list, date_range=None):
    all_records_dfs = []
    all_collected_dates = []
    successfully_processed_files_count = 0
    
    try:
        results = []
        for uploaded_file_obj in uploaded_files_list:
            r = _parse_single_csv_to_records_df(uploaded_file_obj.name, uploaded_file_obj)
            results.append(r)

        for i, records_df_single in enumerate(results):
            filename = uploaded_files_list[i].name
            
            if records_df_single is not None and not records_df_single.empty:
                records_df_single['FileName'] = filename
                all_records_dfs.append(records_df_single)
                successfully_processed_files_count += 1
            else:
                print(f"No valid records extracted from '{filename}'. It will be excluded.")
    
    except Exception as e:
        print(f"Threaded processing failed: {e}. Falling back to sequential processing.")        
    if not all_records_dfs:
        print("No CSV files were successfully processed to provide records for merging.")
        return None, list(set(all_collected_dates)), {}, None, [], [], []

    # Process consolidated data (existing logic - unchanged)
    combined_records_df = pd.concat(all_records_dfs, ignore_index=True)
    if combined_records_df.empty:
        print("Combined records from CSV files are empty before final aggregation.")
        return None, list(set(all_collected_dates)), {}, None

    combined_records_df = combined_records_df.sort_values(by=['Date', 'Timestamp'])
    
    if combined_records_df.empty:
        print("Combined records are empty prior to daily aggregation.")
        return None, list(set(all_collected_dates)), {}, None

    # CRITICAL: Store the raw trade-level data BEFORE any date filtering for Monte Carlo resampling
    # This must happen BEFORE date range filter is applied
    raw_trades_df = combined_records_df.copy()

    # Apply date range filtering to consolidated data AFTER storing unfiltered raw trades
    # This filtering only affects the DISPLAY/AGGREGATION, not Monte Carlo
    if date_range is not None and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date is not None and end_date is not None:
            combined_records_df = combined_records_df[
                (combined_records_df['Date'] >= pd.to_datetime(start_date)) &
                (combined_records_df['Date'] <= pd.to_datetime(end_date))
            ]

            if combined_records_df.empty:
                print(f"No records found in the specified date range: {start_date} to {end_date}.")
                return None, list(set(all_collected_dates)), {}, raw_trades_df
            
    return list(set(all_collected_dates)), raw_trades_df
    
def _parse_single_csv_to_records_df(filename, _uploaded_file_obj):
    df_temp = None
    try:
        _uploaded_file_obj.seek(0)
        df_temp = pd.read_csv(_uploaded_file_obj)

        df_temp.rename(columns={'Date Opened': 'Original_Date',
                                'Date Closed': 'Date_Closed_Original',
                                'Funds at Close': 'NetLiquidity_Raw',
                                'P/L': 'PL_Raw'}, inplace=True)
        df_temp['Date'] = pd.to_datetime(df_temp['Original_Date'], errors='coerce')
        time_col_for_sort = 'Time Opened'
        if 'Time Closed' in df_temp.columns and df_temp['Time Closed'].notna().any():
            time_col_for_sort = 'Time Closed'

        if df_temp['Date'].isna().all():
            print(f"File '{filename}': Date column could not be parsed or is all empty.")
            return None, []

        # OPTIMIZED: Parse P/L using helper function
        pl_col_name = 'PL_Raw'
        if pl_col_name in df_temp.columns:
            df_temp['P/L_Parsed'] = parse_currency_column(df_temp[pl_col_name])
        else:
            df_temp['P/L_Parsed'] = np.nan

        # OPTIMIZED: Parse Net Liquidity using helper function
        nl_col_name = 'NetLiquidity_Raw'
        if nl_col_name in df_temp.columns:
            df_temp['Net Liquidity_Parsed'] = parse_currency_column(df_temp[nl_col_name])
        else:
            df_temp['Net Liquidity_Parsed'] = np.nan

        # OPTIMIZED: Create timestamp more efficiently
        df_temp['Time_str_for_sort'] = df_temp[time_col_for_sort].astype(str).fillna('00:00:00')
        # Use str slicing instead of dt.strftime for speed
        date_str = df_temp['Date'].astype(str).str[:10]
        df_temp['Timestamp'] = pd.to_datetime(
            date_str + ' ' + df_temp['Time_str_for_sort'],
            errors='coerce'
        )
        
        # Fill NaN timestamps with date
        df_temp['Timestamp'] = df_temp['Timestamp'].fillna(df_temp['Date'])

        # Build columns to keep
        columns_to_keep = ['Date', 'Timestamp', 'P/L_Parsed', 'Net Liquidity_Parsed']
        
        # Always preserve Time Opened if it exists
        if 'Time Opened' in df_temp.columns:
            columns_to_keep.append('Time Opened')
        
        # Add scatter plot columns if they exist
        if 'Max Loss' in df_temp.columns:
            df_temp['Max Loss_Parsed'] = pd.to_numeric(df_temp['Max Loss'], errors='coerce')
            columns_to_keep.append('Max Loss_Parsed')
        
        if 'Legs' in df_temp.columns:
            columns_to_keep.append('Legs')

        # Preserve Strategy column for portfolio grouping
        if 'Strategy' in df_temp.columns:
            columns_to_keep.append('Strategy')

        # Preserve Margin Req. for worst-case calculation
        if 'Margin Req.' in df_temp.columns:
            columns_to_keep.append('Margin Req.')

        # Handle Date Opened column
        df_temp['Date Opened_Parsed'] = pd.to_datetime(df_temp['Original_Date'], errors='coerce')
        columns_to_keep.append('Date Opened_Parsed')

        # Select and rename columns
        final_df = df_temp[columns_to_keep].copy()
        
        rename_dict = {
            'P/L_Parsed': 'P/L',
            'Net Liquidity_Parsed': 'Net Liquidity'
        }
        
        if 'Max Loss_Parsed' in final_df.columns:
            rename_dict['Max Loss_Parsed'] = 'Max Loss'
        if 'Date Opened_Parsed' in final_df.columns:
            rename_dict['Date Opened_Parsed'] = 'Date Opened'
        
        final_df.rename(columns=rename_dict, inplace=True)
        
        # Drop rows with invalid dates
        final_df = final_df.dropna(subset=['Date'])
        
        return final_df
        
    except Exception as e:
        print(f"Critical error processing records from file {filename}: {e}")
        salvaged_dates = []
        if df_temp is not None and 'Date' in df_temp.columns:
            try:
                salvaged_dates = pd.to_datetime(df_temp['Date'], errors='coerce').dropna().dt.date.unique().tolist()
            except:
                pass
        return None