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
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple


def calculate_cagr(cumulative_pl, starting_capital, num_periods, periods_per_year):
    ending_value = starting_capital + cumulative_pl
    years = num_periods / periods_per_year
    
    if years == 0 or starting_capital <= 0:
        return 0
    
    cagr = (ending_value / starting_capital) ** (1 / years) - 1
    return cagr


def calculate_max_drawdown(series):
    cumulative = series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)  # Return positive value


def calculate_mar(series, starting_capital, periods_per_year):
    cumulative_pl = series.sum()
    num_periods = len(series)
    
    cagr = calculate_cagr(cumulative_pl, starting_capital, num_periods, periods_per_year)
    max_dd = calculate_max_drawdown(series)
    
    # Handle zero drawdown case (perfect equity curve)
    if max_dd == 0:
        if cagr > 0:
            return 999.0  # Very high MAR for perfect positive returns
        elif cagr < 0:
            return -999.0  # Very low MAR for losses with no drawdown (unusual case)
        else:
            return 0  # No returns, no drawdown
    
    mar = cagr / (max_dd / starting_capital)  # Normalize drawdown as percentage
    return mar


def calculate_sortino(series, periods_per_year, target_return=0):
    avg_return = series.mean()
    
    # Only calculate deviation for returns below target (usually 0)
    downside_returns = series[series < target_return]
    
    if len(downside_returns) == 0:
        # No downside - perfect!
        if avg_return > 0:
            return 999.0  # Perfect upside, no downside
        else:
            return 0  # No returns at all
    
    downside_deviation = downside_returns.std()
    
    if downside_deviation == 0:
        return 0
    
    sortino = (avg_return / downside_deviation) * np.sqrt(periods_per_year)
    return sortino


def calculate_metrics(df_daily, starting_capital, periods_per_year):
    metrics = {}
    for entry_time in df_daily.columns:
        series = df_daily[entry_time]
        total_pl = series.sum()
        avg_pl = series.mean()
        std_pl = series.std()
        sharpe = (avg_pl / std_pl) * np.sqrt(periods_per_year) if std_pl > 0 else 0
        
        # Calculate Sortino
        sortino = calculate_sortino(series, periods_per_year)
        
        # Calculate MAR
        mar = calculate_mar(series, starting_capital, periods_per_year)
        
        # Calculate CAGR and Max DD for display
        cumulative_pl = series.sum()
        num_periods = len(series)
        cagr = calculate_cagr(cumulative_pl, starting_capital, num_periods, periods_per_year)
        max_dd = calculate_max_drawdown(series)
        
        metrics[entry_time] = {
            'total_pl': total_pl,
            'avg_pl': avg_pl,
            'sharpe': sharpe,
            'sortino': sortino,
            'mar': mar,
            'cagr': cagr,
            'max_dd': max_dd
        }
    
    return metrics


def select_next_entry_greedy(
    df_daily, 
    selected_times, 
    candidates, 
    corr_matrix, 
    correlation_penalty,
    optimization_metric,
    starting_capital,
    periods_per_year,
    min_time_distance_minutes=0
):
    if optimization_metric == 'MAR':
        metric_key = 'mar'
    elif optimization_metric == 'SORTINO':
        metric_key = 'sortino'
    elif optimization_metric == 'PROFIT':
        metric_key = 'avg_pl'
    else:  # SHARPE
        metric_key = 'sharpe'
    
    best_time = None
    best_adjusted_metric = -999999
    
    for entry_time in candidates:
        if entry_time in selected_times:
            continue
        
        # Check minimum time distance constraint
        if min_time_distance_minutes > 0 and selected_times:
            # Convert time strings to minutes since midnight
            def time_to_minutes(time_str):
                h, m = map(int, time_str.split(':'))
                return h * 60 + m
            
            candidate_minutes = time_to_minutes(entry_time)
            too_close = False
            
            for selected_time in selected_times:
                selected_minutes = time_to_minutes(selected_time)
                time_diff = abs(candidate_minutes - selected_minutes)
                
                if time_diff < min_time_distance_minutes:
                    too_close = True
                    break
            
            if too_close:
                continue
        
        # Calculate portfolio performance with this candidate added
        test_times = selected_times + [entry_time]
        portfolio_daily = df_daily[test_times].sum(axis=1)
        
        avg_pl = portfolio_daily.mean()
        std_pl = portfolio_daily.std()
        
        if optimization_metric == 'SHARPE':
            raw_metric = (avg_pl / std_pl) * np.sqrt(periods_per_year) if std_pl > 0 else 0
        elif optimization_metric == 'SORTINO':
            raw_metric = calculate_sortino(portfolio_daily, periods_per_year)
        elif optimization_metric == 'PROFIT':
            raw_metric = avg_pl
        else:  # MAR
            raw_metric = calculate_mar(portfolio_daily, starting_capital, periods_per_year)
        
        # Calculate average correlation with existing selections
        if selected_times:
            corrs = [corr_matrix.loc[entry_time, st] for st in selected_times]
            avg_corr = np.mean(corrs)
        else:
            avg_corr = 0
        
        # Apply correlation penalty
        adjusted_metric = raw_metric - (correlation_penalty * avg_corr)
        
        if adjusted_metric > best_adjusted_metric:
            best_adjusted_metric = adjusted_metric
            best_time = entry_time
    
    return best_time


def optimize_all_sizes(
    df_daily, 
    metrics, 
    top_n_candidates,
    max_entry_times,
    correlation_penalty,
    optimization_metric,
    starting_capital,
    periods_per_year,
    min_time_distance_minutes=0
):
    """
    Build optimal portfolio by adding entries one at a time using greedy optimization
    """
    if optimization_metric == 'MAR':
        metric_key = 'mar'
    elif optimization_metric == 'SORTINO':
        metric_key = 'sortino'
    elif optimization_metric == 'PROFIT':
        metric_key = 'avg_pl'
    else:  # SHARPE
        metric_key = 'sharpe'
    
    # Get top N most profitable times
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['total_pl'], reverse=True)
    candidates = [t for t, _ in sorted_metrics[:top_n_candidates]]
    
    # Calculate correlation matrix
    corr_matrix = df_daily[candidates].corr()
    
    # Build sequence greedily
    master_sequence = []
    for _ in range(min(max_entry_times, len(candidates))):
        next_time = select_next_entry_greedy(
            df_daily,
            master_sequence,
            candidates,
            corr_matrix,
            correlation_penalty,
            optimization_metric,
            starting_capital,
            periods_per_year,
            min_time_distance_minutes
        )
        
        if next_time is None:
            break
        
        master_sequence.append(next_time)
    
    # Evaluate performance at each portfolio size
    results = {}
    for size in range(1, len(master_sequence) + 1):
        selected_times = master_sequence[:size]
        
        portfolio_daily = df_daily[selected_times].sum(axis=1)
        total_pl = portfolio_daily.sum()
        avg_pl = portfolio_daily.mean()
        std_pl = portfolio_daily.std()
        
        # Annualized Sharpe
        sharpe = (avg_pl / std_pl) * np.sqrt(periods_per_year) if std_pl > 0 else 0
        
        # Calculate Sortino for portfolio
        sortino = calculate_sortino(portfolio_daily, periods_per_year)
        
        # Calculate MAR for portfolio
        mar = calculate_mar(portfolio_daily, starting_capital, periods_per_year)
        
        # Calculate CAGR and Max DD
        cumulative_pl = portfolio_daily.sum()
        num_periods = len(portfolio_daily)
        cagr = calculate_cagr(cumulative_pl, starting_capital, num_periods, periods_per_year)
        max_dd = calculate_max_drawdown(portfolio_daily)
        
        if size > 1:
            corrs = [corr_matrix.loc[selected_times[i], selected_times[j]] 
                     for i in range(len(selected_times)) 
                     for j in range(i+1, len(selected_times))]
            avg_correlation = np.mean(corrs)
        else:
            avg_correlation = 0
            
        results[size] = {
            'times': selected_times.copy(),
            'total_pl': total_pl,
            'avg_pl': avg_pl,
            'sharpe': sharpe,
            'sortino': sortino,
            'mar': mar,
            'cagr': cagr,
            'max_dd': max_dd,
            'avg_corr': avg_correlation,
            'pl_per_entry': total_pl / size,
            'sharpe_per_entry': sharpe / np.sqrt(size),
            'sortino_per_entry': sortino / np.sqrt(size),
            'mar_per_entry': mar / np.sqrt(size)
        }
    
    return results


def recommend_optimal_size(results, force_size, marginal_threshold, optimization_metric):
    """Determine optimal portfolio size"""
    if optimization_metric == 'MAR':
        metric_key = 'mar'
    elif optimization_metric == 'SORTINO':
        metric_key = 'sortino'
    elif optimization_metric == 'PROFIT':
        metric_key = 'avg_pl'
    else:  # SHARPE
        metric_key = 'sharpe'
    
    if force_size is not None:
        if force_size != 0 and force_size in results:
            return force_size
    
    # Identify the absolute best size based on chosen metric
    best_metric_size = max(results.keys(), key=lambda s: results[s][metric_key])
    
    # Check if max size degrades performance
    if best_metric_size < max(results.keys()):
        return best_metric_size
    
    # Look for elbow in metric curve
    optimal_size = best_metric_size
    for size in range(2, best_metric_size + 1):
        metric_gain = results[size][metric_key] - results[size-1][metric_key]
        if metric_gain < marginal_threshold:
            optimal_size = size - 1
            break
    
    return optimal_size


def run_entry_time_analysis(
    raw_trades_list: List[pd.DataFrame],
    start_date,
    end_date,
    max_entry_times: int = 200,
    force_size: int = None,
    top_n_candidates: int = 300,
    correlation_penalty: float = 1.5,
    min_time_distance_minutes: int = 20,
    optimization_metric: str = 'SORTINO',
    marginal_threshold: float = 0.05,
    starting_capital: float = 10000,
    periods_per_year: int = 52
) -> Dict:
    """
    Main analysis function for entry time optimization.
    
    Args:
        raw_trades_list: List of DataFrames with raw trade data
        start_date: Start date for filtering
        end_date: End date for filtering
        max_entry_times: Maximum number of entry times to test
        force_size: Force specific number of entries (None for automatic)
        top_n_candidates: Consider top N profitable times
        correlation_penalty: Penalty for correlation (1.5-2.0 recommended)
        min_time_distance_minutes: Minimum minutes between entries
        optimization_metric: 'SHARPE', 'SORTINO', or 'MAR'
        marginal_threshold: Minimum improvement threshold
        starting_capital: Starting capital for CAGR calculation
        periods_per_year: 52 for weekly, 252 for daily
        
    Returns:
        Dictionary with analysis results
    """
    if not raw_trades_list or all(df is None or df.empty for df in raw_trades_list):
        return {"error": "No trade data available"}
    
    # Combine all dataframes
    all_dfs = [df for df in raw_trades_list if df is not None and not df.empty]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Check required columns
    required_columns = ['Date Opened', 'Time Opened', 'P/L']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        return {"error": f"Missing required columns: {missing_columns}"}
    
    # Apply date filter
    combined_df['Date Opened'] = pd.to_datetime(combined_df['Date Opened'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Date Opened'])
    
    if start_date and end_date:
        combined_df = combined_df[
            (combined_df['Date Opened'] >= pd.to_datetime(start_date)) &
            (combined_df['Date Opened'] <= pd.to_datetime(end_date))
        ]
    
    if combined_df.empty:
        return {"error": "No trades found in selected date range"}
    
    # Extract entry time (HH:MM)
    combined_df['Entry Time'] = combined_df['Time Opened'].str[:5]
    
    # Build daily P/L matrix
    daily_pl_by_time = {}
    for entry_time in sorted(combined_df['Entry Time'].unique()):
        time_trades = combined_df[combined_df['Entry Time'] == entry_time]
        daily_pl = time_trades.groupby('Date Opened')['P/L'].sum()
        daily_pl_by_time[entry_time] = daily_pl
    
    df_daily = pd.DataFrame(daily_pl_by_time)
    
    if df_daily.empty:
        return {"error": "Could not create daily P/L matrix"}
    
    # Calculate metrics for each entry time
    metrics = calculate_metrics(df_daily, starting_capital, periods_per_year)
    
    # Optimize portfolio sizes
    results = optimize_all_sizes(
        df_daily,
        metrics,
        top_n_candidates,
        max_entry_times,
        correlation_penalty,
        optimization_metric,
        starting_capital,
        periods_per_year,
        min_time_distance_minutes
    )
    
    # Determine optimal size
    optimal_size = recommend_optimal_size(results, force_size, marginal_threshold, optimization_metric)
    
    return {
        'combined_df': combined_df,
        'df_daily': df_daily,
        'metrics': metrics,
        'results': results,
        'optimal_size': optimal_size,
        'config': {
            'max_entry_times': max_entry_times,
            'force_size': force_size,
            'top_n_candidates': top_n_candidates,
            'correlation_penalty': correlation_penalty,
            'min_time_distance_minutes': min_time_distance_minutes,
            'optimization_metric': optimization_metric,
            'marginal_threshold': marginal_threshold,
            'starting_capital': starting_capital,
            'periods_per_year': periods_per_year
        }
    }

def render_entry_time_results(analysis_results: Dict):
    """Render the entry time optimization results with text and visualizations"""
    if 'error' in analysis_results:
        st.error(analysis_results['error'])
        return
    
    metrics = analysis_results['metrics']
    results = analysis_results['results']
    optimal_size = analysis_results['optimal_size']
    config = analysis_results['config']
    combined_df = analysis_results['combined_df']
    
    # Map optimization metric to dictionary key
    if config['optimization_metric'] == 'MAR':
        metric_key = 'mar'
    elif config['optimization_metric'] == 'SORTINO':
        metric_key = 'sortino'
    elif config['optimization_metric'] == 'PROFIT':
        metric_key = 'avg_pl'
    else:  # SHARPE
        metric_key = 'sharpe'

    # Header with key info
    st.markdown(f"""
    **Total Trades Analyzed**: {len(combined_df):,}  
    **Unique Entry Times**: {len(metrics)}  
    **Optimization Metric**: {config['optimization_metric']}  
    **Optimal Portfolio Size**: {optimal_size} entries
    """)
    
    r = results[optimal_size]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total P/L", f"${r['total_pl']:,.0f}")
    with col2:
        st.metric(f"{config['optimization_metric']} Ratio (Normalized)", f"{r[metric_key]:.2f}")
    with col3:
        st.metric("CAGR", f"{r['cagr']*100:.1f}%")
    with col4:
        st.metric("Avg Correlation", f"{r['avg_corr']:.3f}")

    st.markdown("#### Your Optimal Entry Times")
    optimal_times = results[optimal_size]['times']

    if optimal_times:
        optimal_times = sorted(results[optimal_size]['times'])
        
        time_data = []
        for entry_time in optimal_times:
            m = metrics[entry_time]
            time_data.append({
                'Time': entry_time,
                'Total P/L': f"${m['total_pl']:,.0f}",
                'Sharpe': f"{m['sharpe']:.2f}",
                'Sortino': f"{m['sortino']:.2f}",
                'MAR': f"{m['mar']:.2f}",
                'CAGR': f"{m['cagr']*100:.1f}%",
                'Max DD': f"${m['max_dd']:,.0f}"
            })
        
        time_df = pd.DataFrame(time_data)
        st.dataframe(time_df, width='stretch')


    # 2. Portfolio Size Comparison Chart
    st.markdown("#### Portfolio Performance vs Size")
    
    sizes = sorted(results.keys())
    metric_values = [results[s][metric_key] for s in sizes]
    total_pls = [results[s]['total_pl'] for s in sizes]
    correlations = [results[s]['avg_corr'] for s in sizes]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{config["optimization_metric"]} Ratio by Portfolio Size', 
                       'Total P/L by Portfolio Size'),
        vertical_spacing=0.15
    )
    
    # Metric chart
    fig.add_trace(
        go.Scatter(
            x=sizes, y=metric_values,
            mode='lines+markers',
            name=config['optimization_metric'],
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Highlight optimal size
    fig.add_vline(
        x=optimal_size,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal: {optimal_size}",
        row=1, col=1
    )
    
    # Total P/L chart
    fig.add_trace(
        go.Bar(
            x=sizes, y=total_pls,
            name='Total P/L',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Number of Entry Times", row=2, col=1)
    fig.update_yaxes(title_text=f"{config['optimization_metric']} Ratio (Normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Total P/L ($)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    
    # 3. Detailed Size Comparison Table
    with st.expander("Detailed Portfolio Size Comparison", expanded=False):
        comparison_data = []
        for size in sizes:
            r = results[size]
            comparison_data.append({
                'Size': size,
                'Total P/L': f"${r['total_pl']:,.0f}",
                'Sharpe': f"{r['sharpe']:.2f}",
                'Sortino': f"{r['sortino']:.2f}",
                'MAR': f"{r['mar']:.2f}",
                'CAGR': f"{r['cagr']*100:.1f}%",
                'Max DD': f"${r['max_dd']:,.0f}",
                'Avg Corr': f"{r['avg_corr']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width='stretch')
    
    # 5. Top/Bottom Entry Times
    with st.expander("Top & Bottom Entry Times (All)", expanded=False):
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['total_pl'], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Most Profitable:**")
            top_10_data = []
            for entry_time, m in sorted_metrics[:10]:
                top_10_data.append({
                    'Time': entry_time,
                    'Total P/L': f"${m['total_pl']:,.0f}",
                    f"{config['optimization_metric']}": f"{m[metric_key]:.2f}"
                })
            st.dataframe(pd.DataFrame(top_10_data), width='stretch')
        
        with col2:
            st.markdown("**Bottom 10 Least Profitable:**")
            bottom_10_data = []
            for entry_time, m in sorted_metrics[-10:]:
                bottom_10_data.append({
                    'Time': entry_time,
                    'Total P/L': f"${m['total_pl']:,.0f}",
                    f"{config['optimization_metric']}": f"{m[metric_key]:.2f}"
                })
            st.dataframe(pd.DataFrame(bottom_10_data), width='stretch')
    
    # 6. Marginal Benefit Analysis
    with st.expander("Marginal Benefit Analysis", expanded=False):
        st.markdown("Shows how much each additional entry improves performance:")
        
        marginal_data = []
        for size in range(2, len(results) + 1):
            if size not in results or size - 1 not in results:
                continue
            
            prev = results[size-1]
            curr = results[size]
            marginal_metric = curr[metric_key] - prev[metric_key]
            
            status = "Good" if marginal_metric >= config['marginal_threshold'] else "Marginal"
            
            marginal_data.append({
                'Adding Entry': f"#{size}",
                f'{config["optimization_metric"]} Gain': f"{marginal_metric:.3f}",
                'New Correlation': f"{curr['avg_corr']:.3f}",
                'Status': status
            })
        
        marginal_df = pd.DataFrame(marginal_data)
        st.dataframe(marginal_df, width='stretch')