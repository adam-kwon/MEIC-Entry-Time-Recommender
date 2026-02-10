import logging
import warnings

# FORCE silence before ANY other imports happen
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Prevent the "streamlit run" suggestion
import streamlit.runtime.scriptrunner_utils.script_run_context as src
logging.getLogger(src.__name__).setLevel(logging.ERROR)

from concurrent.futures import ThreadPoolExecutor, as_completed
from data.data_utils import load_and_parse_csv

def load_csv_data(uploaded_files):
    return process_individual_trade_logs_threaded(uploaded_files)
    
def _process_single_file_wrapper(uploaded_file):
    df_item, raw_trades = load_and_parse_csv([uploaded_file])
    return df_item, raw_trades

def process_individual_trade_logs_threaded(uploaded_files):
    ExecutorClass = ThreadPoolExecutor    
    try:
        processed_dfs = []
        raw_trades_list = []

        with ExecutorClass() as executor:
            future_to_file = {
                executor.submit(_process_single_file_wrapper, uploaded_file): uploaded_file 
                for uploaded_file in uploaded_files
            }
            
            for future in as_completed(future_to_file):
                uploaded_file = future_to_file[future]
                
                try:
                    df_item, raw_trades = future.result()
                                            
                    processed_dfs.append(df_item)
                    raw_trades_list.append(raw_trades)
                except Exception as e:
                    for f in future_to_file:
                        f.cancel()
                    raise Exception(f"Failed to process {uploaded_file.name}: {str(e)}")        
    except Exception as e:
        print(f"ERROR: Processing failed - {str(e)}")
        raise

    return processed_dfs, raw_trades_list
