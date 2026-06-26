import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401 – imported for side-effect of backend selection

from src.data_loader import (
    load_or_download_data,
    ETF_CATEGORIES,
    STUDY_START,
    STUDY_END,
)
from src.descriptive_stats import (
    compute_summary_statistics,
    compute_correlation_matrix,
    normalize_prices,
    compute_cumulative_returns,
    plot_correlation_heatmap,
    plot_normalized_prices_by_category,
    plot_cumulative_returns_by_category,
    plot_return_distributions,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s – %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'

_ASSUMPTIONS = [
    "RF CONVENTION : simple daily rate = annualised T-bill yield (^IRX) / 252.",
    "RF FALLBACK   : if ^IRX download fails, a constant 5 % (0.05) annual rate is used.",
    "RETURNS TYPE  : arithmetic (simple) daily returns via pct_change().",
    "PANEL BALANCE : dates with any NaN across ETFs are dropped.",
    "STUDY PERIOD  : {} to {}.".format(STUDY_START, STUDY_END),
    "PRICES SOURCE : Polygon.io daily OHLCV aggregates (close price).",
    "KURTOSIS      : excess kurtosis reported (Fisher=True, normal distribution = 0).",
    "NORMALISATION : price series rebased to 100 at first observation.",
]


def run_exp1() -> dict:
    """Execute Experiment 1: descriptive statistics for commodity ETF universe.

    Steps
    -----
    1. Load (or download) price and risk-free rate data.
    2. Compute summary statistics, correlation matrix, normalised prices, and
       cumulative returns.
    3. Persist all tabular outputs as CSV files in *results/*.
    4. Generate and save four publication-quality plots to *results/*.
    5. Print the assumption log to stdout.
    6. Return a high-level summary dictionary.

    Returns
    -------
    dict
        Keys: ``n_assets``, ``n_dates``, ``study_start``, ``study_end``,
        ``results_dir``, ``assets``.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    logger.info("Loading data …")
    prices_df, returns_df, rf_daily, metadata = load_or_download_data(
        os.getenv('MASSIVE_TOKEN')
    )
    categories = metadata.get('categories', ETF_CATEGORIES)

    logger.info(
        "Loaded %d assets × %d trading days.", len(returns_df.columns), len(returns_df)
    )

    # ------------------------------------------------------------------ #
    # 2. Compute statistics                                                #
    # ------------------------------------------------------------------ #
    logger.info("Computing summary statistics …")
    summary_stats = compute_summary_statistics(returns_df)

    logger.info("Computing correlation matrix …")
    corr_matrix = compute_correlation_matrix(returns_df)

    logger.info("Normalising prices …")
    norm_prices = normalize_prices(prices_df)

    logger.info("Computing cumulative returns …")
    cum_returns = compute_cumulative_returns(returns_df)

    # ------------------------------------------------------------------ #
    # 3. Save tabular outputs                                              #
    # ------------------------------------------------------------------ #
    summary_path = RESULTS_DIR / 'summary_statistics.csv'
    summary_stats.to_csv(summary_path)
    logger.info("Saved summary statistics → %s", summary_path)

    corr_path = RESULTS_DIR / 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_path)
    logger.info("Saved correlation matrix → %s", corr_path)

    norm_path = RESULTS_DIR / 'normalized_prices.csv'
    norm_prices.to_csv(norm_path)
    logger.info("Saved normalised prices → %s", norm_path)

    cum_path = RESULTS_DIR / 'cumulative_returns.csv'
    cum_returns.to_csv(cum_path)
    logger.info("Saved cumulative returns → %s", cum_path)

    # ------------------------------------------------------------------ #
    # 4. Generate plots                                                    #
    # ------------------------------------------------------------------ #
    logger.info("Plotting correlation heatmap …")
    plot_correlation_heatmap(
        corr_matrix,
        categories,
        RESULTS_DIR / 'correlation_heatmap.png',
    )

    logger.info("Plotting normalised prices by category …")
    plot_normalized_prices_by_category(
        norm_prices,
        categories,
        RESULTS_DIR / 'normalized_prices_by_category.png',
    )

    logger.info("Plotting cumulative returns by category …")
    plot_cumulative_returns_by_category(
        cum_returns,
        categories,
        RESULTS_DIR / 'cumulative_returns_by_category.png',
    )

    logger.info("Plotting return distributions …")
    plot_return_distributions(
        returns_df,
        RESULTS_DIR / 'return_distributions.png',
    )

    # ------------------------------------------------------------------ #
    # 5. Print assumption log                                              #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("ASSUMPTION LOG – Experiment 1: Descriptive Statistics")
    print("=" * 70)
    for assumption in _ASSUMPTIONS:
        print(f"  • {assumption}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------ #
    # 6. Build and return summary                                          #
    # ------------------------------------------------------------------ #
    summary = {
        'n_assets': len(returns_df.columns),
        'n_dates': len(returns_df),
        'study_start': str(returns_df.index[0].date()),
        'study_end': str(returns_df.index[-1].date()),
        'results_dir': str(RESULTS_DIR),
        'assets': list(returns_df.columns),
    }
    logger.info("Experiment 1 complete.  Results in %s.", RESULTS_DIR)
    return summary


if __name__ == '__main__':
    run_exp1()
