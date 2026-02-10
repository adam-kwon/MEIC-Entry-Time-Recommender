import logging
import warnings

logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import streamlit.runtime.scriptrunner_utils.script_run_context as src
logging.getLogger(src.__name__).setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

def update_config(key, value):
    """Store config value in session state"""
    st.session_state[key] = value

def get_config(key, default=None):
    """Retrieve config value from session state"""
    return st.session_state.get(key, default)

def parse_dollar(value):
    """Convert a string like ($955.16) or $531.80 to float"""
    if not isinstance(value, str):
        if pd.is_numeric(value):
            return float(value)
        return np.nan
    value = value.replace('$', '').replace(',', '').strip()
    if not value: return np.nan
    try:
        if value.startswith('(') and value.endswith(')'): return -float(value[1:-1])
        return float(value)
    except ValueError: return np.nan

def parse_currency_column(series):
    """
    Fast parser for currency values like $1,234.56 or ($1,234.56)
    Combines regex operations into a single optimized chain.
    """
    return (series
            .astype(str)
            .str.replace(r'[\$,]', '', regex=True)  # Remove $ and ,
            .str.replace(r'^\((.*)\)$', r'-\1', regex=True)  # Handle (123) -> -123
            .pipe(pd.to_numeric, errors='coerce'))

def get_date_range_from_csv(file_path: str) -> Optional[tuple]:
    """
    Get the date range (min, max) from a CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (min_date, max_date) or None if error
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['Date Opened'] = pd.to_datetime(df['Date Opened'])
        return (df['Date Opened'].min(), df['Date Opened'].max())
    except Exception as e:
        print(f"Error reading date range from {file_path}: {e}")
        return None