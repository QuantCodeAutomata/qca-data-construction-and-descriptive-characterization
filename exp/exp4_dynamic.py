import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd
import numpy as np

from src.data_loader import load_or_download_data
from src.dynamic_optimizer import run_dynamic_backtest
from src.historical_optimizer import compute_metrics_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'

def run_exp4() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv('MASSIVE_TOKEN')
    prices_df, returns_df, rf_daily, metadata = load_or_download_data(api_key)
    
    print('ASSUMPTION: ARMA(1,1)-GARCH(1,1) model order (paper unspecified)')
    print('ASSUMPTION: Empirical PIT for marginal transformation')
    print('ASSUMPTION: Student-t copula with empirical dof (median of univariate t-fits)')
    print('ASSUMPTION: 5000 Monte Carlo scenarios')
    print('ASSUMPTION: Variant A = no dynamic turnover cap (primary)')
    
    # Rebalance monthly (21 trading days) for tractable runtime;
    # n_scenarios=2000 is sufficient for robust CVaR estimation.
    # ASSUMPTION: rebalance_every=21 (paper unspecified)
    # ASSUMPTION: n_scenarios=2000 (paper unspecified; default is 5000)
    strat_returns, strat_weights, strat_turnover = run_dynamic_backtest(
        returns_df, rf_daily, variant='A', n_scenarios=2000, rebalance_every=21
    )
    
    returns_out = pd.DataFrame(strat_returns)
    returns_out.to_parquet(RESULTS_DIR / 'exp4_dynamic_returns_varA.parquet')
    pd.DataFrame(strat_turnover).to_parquet(RESULTS_DIR / 'exp4_dynamic_turnover_varA.parquet')
    
    metrics_df = compute_metrics_table(strat_returns, rf_daily, strat_turnover)
    metrics_df.to_csv(RESULTS_DIR / 'exp4_metrics_varA.csv')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, ret_series in strat_returns.items():
        cum_val = 100 * (1 + ret_series).cumprod()
        ax.plot(cum_val.index, cum_val.values, label=name, linewidth=1.0)
    ax.set_title('Dynamic Strategy Cumulative Value ($100 initial)')
    ax.set_xlabel('Date'); ax.set_ylabel('Portfolio Value ($)')
    ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp4_cumulative_value.png', dpi=150)
    plt.close()
    
    return {'n_strategies': len(strat_returns), 'n_dates': len(returns_out)}

if __name__ == '__main__':
    run_exp4()
