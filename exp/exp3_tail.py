import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib
matplotlib.use('Agg')
import os
import logging
import pandas as pd
import numpy as np

from src.tail_diagnostics import compute_all_tail_metrics, plot_hill_curves, plot_tail_metrics_comparison

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def run_exp3() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    # Load exp2 results
    returns_path = RESULTS_DIR / 'exp2_strategy_returns.parquet'
    if not returns_path.exists():
        logging.info('exp2 results not found, running exp2 first...')
        from exp.exp2_historical import run_exp2
        run_exp2()

    strategy_returns_df = pd.read_parquet(returns_path)
    strategy_returns = {col: strategy_returns_df[col].dropna() for col in strategy_returns_df.columns}

    # Compute tail metrics
    metrics_df = compute_all_tail_metrics(strategy_returns)
    metrics_df.to_csv(RESULTS_DIR / 'exp3_tail_metrics.csv')

    # Hill curves for subset
    subset = [s for s in ['BHP', 'LO_MVP', 'LO_C95', 'LO_TVP', 'LO_TC95'] if s in strategy_returns]
    plot_hill_curves(strategy_returns, RESULTS_DIR / 'exp3_hill_curves.png', strategies_subset=subset)
    plot_tail_metrics_comparison(metrics_df, RESULTS_DIR / 'exp3_tail_comparison.png')

    # Print findings
    if 'hill_alpha_mean' in metrics_df.columns:
        top = metrics_df['hill_alpha_mean'].nlargest(3)
        print(f'Thinnest tails (highest Hill alpha): {top.index.tolist()}')

    return metrics_df.to_dict()


if __name__ == '__main__':
    run_exp3()
