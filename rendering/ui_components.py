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
from data.strategy_browser import setup_strategy_browser_ui
from other.config import KEY_FORCE_COUNT, KEY_RECOMMENDER_OPTIMIZER, KEY_MAX_ENTRY_TIMES, KEY_CORRELATION_PENALTY, KEY_MARGINAL_THRESHOLD, KEY_TOP_N_CANDIDATES, KEY_MIN_ENTRY_DISTANCE
from other.helpers import get_config, update_config

def create_config_handler(config_key, widget_key, default_value, value_type=None):
    """
    Create config value and on_change handler for a widget.
    
    Args:
        config_key: Configuration key to load/save
        widget_key: Widget's session_state key
        default_value: Default value if config is None
        value_type: Type to convert loaded value to (e.g., int, float, str)
    
    Returns:
        (current_value, on_change_handler)
    """
    config = get_config(config_key)
    
    if config is None:
        current_value = default_value
    else:
        current_value = value_type(config) if value_type else config
    
    def on_change_handler():
        new_value = st.session_state[widget_key]
        update_config(config_key, new_value)
    
    return current_value, on_change_handler

def render_recommender_ui():
    st.title("MEIC Entry Time Curve Fitter")
    st.caption("Make sure to sanitize the trade log and remove entry times you do not want considered.")

    cols = st.columns([0.2, 0.2, 0.2, 0.2])

    with cols[0]:
        entry_times, handler = create_config_handler(KEY_MAX_ENTRY_TIMES, 'opt_max_entry_times', 200, int)
        st.number_input("No. Entries", 1, 500, entry_times, 10, key='opt_max_entry_times', on_change=handler, help="Maximum number of entry times to test. It will often return less than this if conditions are not met. To force it to return a specific number of entry times, use the 'Force' setting.")

        top_n_candidates, handler = create_config_handler(KEY_TOP_N_CANDIDATES, 'opt_top_n_candidates', 300, int)
        st.number_input("Top", 10, 1000, top_n_candidates, 10, key='opt_top_n_candidates', on_change=handler, help="Consider top N profitable times for optimization.")

    with cols[1]:
        correl_penalty, handler = create_config_handler(KEY_CORRELATION_PENALTY, 'opt_correlation_penalty', 1.5, float)
        st.number_input("Corr. Penalty", 0.0, 50.0, correl_penalty, 0.1, key='opt_correlation_penalty', format="%.1f", on_change=handler, help="A penalty of 1.5 - 2.0 usually prevents 'over-stacking' correlated times.")

        min_distance, handler = create_config_handler(KEY_MIN_ENTRY_DISTANCE, 'opt_min_time_distance', 15, int)
        st.number_input("Min Time Dist.", 5, 120, min_distance, 5, key='opt_min_time_distance', on_change=handler, help="Minimum time distance (in minutes) between selected entry times. Set to 0 to disable this constraint.")

    with cols[2]:
        method, handler = create_config_handler(KEY_RECOMMENDER_OPTIMIZER, 'opt_metric', 'PROFIT', None)

        if method is None:
            optimizer_idx = 1            
        elif method == 'SHARPE':
            optimizer_idx = 0
        elif method == 'SORTINO':
            optimizer_idx = 1
        elif method == 'MAR':
            optimizer_idx = 2
        elif method == 'PROFIT':
            optimizer_idx = 3
                
        st.selectbox("Optimizer", ['SHARPE', 'SORTINO', 'MAR', 'PROFIT'], optimizer_idx, key='opt_metric', on_change=handler, help="Entry times will be selected based on this optimizer.")

        marginal_threshold, handler = create_config_handler(KEY_MARGINAL_THRESHOLD, 'opt_marginal_threshold', 0.05, float)
        st.number_input("Marginal Threshold", 0.0, 1.0, marginal_threshold, 0.01, key='opt_marginal_threshold', format="%.2f", on_change=handler, help="Minimum meaningful change in the optimization metric when adding an entry. Below this threshold adds marginal benefit.")

    with cols[3]:
        force_count, handler = create_config_handler(KEY_FORCE_COUNT, 'opt_force_size', 0, int)
        st.number_input("Force", 0, 350, force_count, 0, key='opt_force_size', on_change=handler, help="Set to 0 for automatic optimization, or set to a number (e.g., 10) to force that many entries.")

        # st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        # if st.button("RECOMMEND"):
        #     st.session_state.run_entry_time_optimizer = True
        #     st.rerun(scope="app")

    
def show_page_layout_toggle():
    layout_options = ("wide", "centered")
    try:
        current_layout_index = layout_options.index(st.session_state.get('app_layout', 'wide'))
    except ValueError:
        current_layout_index = 0
        st.session_state.app_layout = "wide"

    chosen_layout = st.sidebar.radio(
        "Page Layout",
        options=layout_options,
        index=0,
        help="Choose overall page layout"
    )

    if chosen_layout != st.session_state.app_layout:
        st.session_state.app_layout = chosen_layout
        st.rerun()

def show_google_sheets_toggle():
    st.sidebar.markdown("### Google Sheets")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Load Trades"):
            st.session_state.load_google_sheet = True
    with col2:
        if st.button("Preprocess"):
            st.session_state.preprocess_scratch = True

def filter_df_by_date_range(df, date_range):
    if df is None or df.empty or 'Date' not in df.columns:
        return df  # Do nothing if invalid

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            try:
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                        (df['Date'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))]
            except Exception as e:
                st.warning(f"Date filtering error: {e}")
    return df

def setup_ui_components():
    uploaded_files = setup_strategy_browser_ui()            
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    elif 'uploaded_files' in st.session_state:
        uploaded_files = st.session_state.uploaded_files
    else:
        uploaded_files = None

    render_recommender_ui()

    return uploaded_files
