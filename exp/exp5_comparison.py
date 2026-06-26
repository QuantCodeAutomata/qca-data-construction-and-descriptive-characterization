import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib
matplotlib.use('Agg')
import os
import logging
import pandas as pd
import numpy as np

from src.data_loader import load_or_download_data
from src.performance_comparison import (
    run_comparison_analysis, plot_rolling_sharpe_comparison, plot_cost_sensitivity
)

RESULTS_DIR = Path(__file__).parent.parent / 'results'

def run_exp5() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv('MASSIVE_TOKEN')
    _, _, rf_daily, _ = load_or_download_data(api_key)
    
    # Load historical
    hist_returns_df = pd.read_parquet(RESULTS_DIR / 'exp2_strategy_returns.parquet')
    hist_turnover_df = pd.read_parquet(RESULTS_DIR / 'exp2_turnover.parquet')
    hist_returns = {c: hist_returns_df[c].dropna() for c in hist_returns_df.columns}
    hist_turnover = {c: hist_turnover_df[c].dropna() for c in hist_turnover_df.columns}
    
    # Load dynamic
    dyn_returns_df = pd.read_parquet(RESULTS_DIR / 'exp4_dynamic_returns_varA.parquet')
    dyn_turnover_df = pd.read_parquet(RESULTS_DIR / 'exp4_dynamic_turnover_varA.parquet')
    dyn_returns = {c: dyn_returns_df[c].dropna() for c in dyn_returns_df.columns}
    dyn_turnover = {c: dyn_turnover_df[c].dropna() for c in dyn_turnover_df.columns}
    
    gross_df, net_df, bootstrap_dict = run_comparison_analysis(
        hist_returns, dyn_returns, hist_turnover, dyn_turnover, rf_daily
    )
    
    gross_df.to_csv(RESULTS_DIR / 'exp5_gross_metrics.csv')
    net_df.to_csv(RESULTS_DIR / 'exp5_net_metrics.csv')
    
    # Bootstrap results table
    boot_rows = []
    for k, v in bootstrap_dict.items():
        row = {'pair_metric': k}
        row.update(v)
        boot_rows.append(row)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(RESULTS_DIR / 'exp5_bootstrap_results.csv', index=False)
    
    strategies = [s for s in ['BHP','LO_MVP','LO_C95','LO_TVP'] if s in hist_returns and s in dyn_returns]
    plot_rolling_sharpe_comparison(hist_returns, dyn_returns, rf_daily, strategies, RESULTS_DIR/'exp5_rolling_sharpe.png')
    plot_cost_sensitivity(gross_df, net_df, RESULTS_DIR/'exp5_cost_sensitivity.png')
    
    return {'gross_strategies': len(gross_df), 'bootstrap_tests': len(bootstrap_dict)}

if __name__ == '__main__':
    run_exp5()
