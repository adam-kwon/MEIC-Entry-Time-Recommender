# strategy_browser.py
import logging
import warnings

# Silence streamlit warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
from typing import Optional, List


def setup_strategy_browser_ui() -> Optional[List]:
    uploaded_files = st.sidebar.file_uploader(
        "",
        type=['csv'],
        accept_multiple_files=True,
        key="strategy_file_uploader",
        label_visibility="hidden"
    )
    
    if uploaded_files:
        return uploaded_files
    
    return None