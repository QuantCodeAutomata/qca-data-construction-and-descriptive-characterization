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
from src.historical_optimizer import run_historical_backtest, compute_metrics_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def run_exp2() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv('MASSIVE_TOKEN')
    prices_df, returns_df, rf_daily, metadata = load_or_download_data(api_key)

    strategy_returns, strategy_weights, strategy_turnover = run_historical_backtest(returns_df, rf_daily)

    # Save returns
    returns_df_out = pd.DataFrame(strategy_returns)
    returns_df_out.to_parquet(RESULTS_DIR / 'exp2_strategy_returns.parquet')

    # Save turnover
    turnover_df = pd.DataFrame(strategy_turnover)
    turnover_df.to_parquet(RESULTS_DIR / 'exp2_turnover.parquet')

    # Metrics table
    metrics_df = compute_metrics_table(strategy_returns, rf_daily, strategy_turnover)
    metrics_df.to_csv(RESULTS_DIR / 'exp2_metrics_gross.csv')

    # Plot cumulative value
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, ret_series in strategy_returns.items():
        cum_val = 100 * (1 + ret_series).cumprod()
        ax.plot(cum_val.index, cum_val.values, label=name, linewidth=1.0)
    ax.set_title('Historical Strategy Cumulative Value ($100 initial)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp2_cumulative_value.png', dpi=150)
    plt.close()

    print('ASSUMPTION: Initial weights = 1/30 for all strategies')
    print('ASSUMPTION: Turnover cap C_TO = 1/(30*252) per day')

    summary = {'n_strategies': len(strategy_returns), 'n_dates': len(returns_df_out)}
    return summary


if __name__ == '__main__':
    run_exp2()
