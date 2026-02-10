import logging
import warnings

# FORCE silence before ANY other imports happen
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Prevent the "streamlit run" suggestion
import streamlit.runtime.scriptrunner_utils.script_run_context as src
logging.getLogger(src.__name__).setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
from other.config import KEY_RECOMMENDER_OPTIMIZER, KEY_FORCE_COUNT, KEY_TOP_N_CANDIDATES, KEY_MARGINAL_THRESHOLD, KEY_CORRELATION_PENALTY, KEY_MAX_ENTRY_TIMES, KEY_MIN_ENTRY_DISTANCE
from other.helpers import update_config, get_config
from data.trade_log_processing import load_csv_data
from analysis.entry_time_recommender import run_entry_time_analysis, render_entry_time_results
from rendering.ui_components import setup_ui_components

if get_config(KEY_RECOMMENDER_OPTIMIZER) is None:
    update_config(KEY_RECOMMENDER_OPTIMIZER, 'MAR')
if get_config(KEY_FORCE_COUNT) is None:
    update_config(KEY_FORCE_COUNT, 0)
if get_config(KEY_MAX_ENTRY_TIMES) is None:
    update_config(KEY_MAX_ENTRY_TIMES, 300)
if get_config(KEY_TOP_N_CANDIDATES) is None:
    update_config(KEY_TOP_N_CANDIDATES, 300)
if get_config(KEY_CORRELATION_PENALTY) is None:
    update_config(KEY_CORRELATION_PENALTY, 1.5)
if get_config(KEY_MARGINAL_THRESHOLD) is None:
    update_config(KEY_MARGINAL_THRESHOLD, 0.05)
if get_config(KEY_MIN_ENTRY_DISTANCE) is None:
    update_config(KEY_MIN_ENTRY_DISTANCE, 20)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'app_layout': "wide",
        'start_date': pd.to_datetime('2022-05-16').date(),  # Default start date
        'end_date': pd.to_datetime('today').date(),  # Default end date
        'run_entry_time_optimizer': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_entry_time_optimizer(processed_dfs, start_date=None, end_date=None):
    # """Handle entry time optimization analysis"""
    # if not st.session_state.get('run_entry_time_optimizer', False):
    #     return
    
    # st.session_state.run_entry_time_optimizer = False  # Reset flag
    
    raw_trades_list = st.session_state.get('raw_trades_list', [])
    
    if not raw_trades_list or all(df is None or df.empty for df in raw_trades_list):
        st.warning("No trade data available for entry time optimization.")
        return
    
    st.header("Results")
    
    with st.spinner("Running entry time optimization..."):
        optimization_method = get_config(KEY_RECOMMENDER_OPTIMIZER)
        analysis_results = run_entry_time_analysis(
            raw_trades_list=raw_trades_list,
            start_date=start_date,
            end_date=end_date,
            max_entry_times=st.session_state.get('opt_max_entry_times', 200),
            force_size=st.session_state.get('opt_force_size', 0),
            top_n_candidates=st.session_state.get('opt_top_n_candidates', 300),
            correlation_penalty=st.session_state.get('opt_correlation_penalty', 1.5),
            min_time_distance_minutes=st.session_state.get('opt_min_time_distance', 20),
            optimization_metric=optimization_method,
            marginal_threshold=st.session_state.get('opt_marginal_threshold', 0.05),
            starting_capital=10000,  # Could make this configurable
            periods_per_year=52  # Weekly data by default
        )
    
    render_entry_time_results(analysis_results)

def main():
    initialize_session_state()
    st.set_page_config(page_title="MEIC Entry Time Recommender", layout=st.session_state.app_layout)
    
    uploaded_files = setup_ui_components()

    if uploaded_files:
        (processed_dfs, raw_trades_list) = load_csv_data(uploaded_files)

        st.session_state.raw_trades_list = raw_trades_list
        handle_entry_time_optimizer(processed_dfs)

if __name__ == "__main__":
    main()
