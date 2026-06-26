import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_CATEGORY_ORDER = ['Agriculture', 'Energy', 'Metals', 'Broad']


def compute_summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute per-asset descriptive statistics for a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns with assets as columns.

    Returns
    -------
    pd.DataFrame
        One row per asset with columns:
        ``mean_daily``, ``mean_annual``, ``std_daily``, ``std_annual``,
        ``skewness``, ``excess_kurtosis``.
    """
    stats = pd.DataFrame(index=returns.columns)
    stats['mean_daily'] = returns.mean()
    stats['mean_annual'] = returns.mean() * 252
    stats['std_daily'] = returns.std()
    stats['std_annual'] = returns.std() * np.sqrt(252)
    stats['skewness'] = returns.apply(lambda col: scipy.stats.skew(col.dropna()))
    stats['excess_kurtosis'] = returns.apply(
        lambda col: scipy.stats.kurtosis(col.dropna(), fisher=True)
    )
    return stats


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix of daily returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns with assets as columns.

    Returns
    -------
    pd.DataFrame
        Square Pearson correlation matrix.
    """
    return returns.corr(method='pearson')


def normalize_prices(prices: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    """Rebase all price series so they start at *base_value*.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide DataFrame of daily close prices with a DatetimeIndex.
    base_value : float
        Starting value for every series (default 100).

    Returns
    -------
    pd.DataFrame
        Normalised prices with the same shape as *prices*.
    """
    return prices / prices.iloc[0] * base_value


def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative total return paths from arithmetic daily returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Cumulative returns expressed as a fraction (0.10 = 10 %).
    """
    return (1 + returns).cumprod() - 1


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    categories: dict[str, str],
    output_path: str | Path,
) -> None:
    """Save a seaborn heatmap of the correlation matrix sorted by category.

    Tickers are sorted in the order Agriculture → Energy → Metals → Broad so
    that intra-category blocks are contiguous on the heat map.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square Pearson correlation matrix.
    categories : dict[str, str]
        Mapping ``{ticker: category}``.
    output_path : str or Path
        File path at which to save the figure (PNG).
    """
    def _sort_key(ticker: str) -> tuple[int, str]:
        cat = categories.get(ticker, 'Unknown')
        order = _CATEGORY_ORDER.index(cat) if cat in _CATEGORY_ORDER else len(_CATEGORY_ORDER)
        return order, ticker

    sorted_tickers = sorted(corr_matrix.columns, key=_sort_key)
    sorted_corr = corr_matrix.loc[sorted_tickers, sorted_tickers]

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        sorted_corr,
        ax=ax,
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 6},
        linewidths=0.3,
        linecolor='white',
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
    )
    ax.set_title('ETF Return Correlation Matrix (sorted by category)', fontsize=13, pad=12)
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s.", output_path)


def plot_normalized_prices_by_category(
    norm_prices: pd.DataFrame,
    categories: dict[str, str],
    output_path: str | Path,
) -> None:
    """Save a 2×2 figure of normalised price paths, one panel per category.

    Parameters
    ----------
    norm_prices : pd.DataFrame
        Normalised prices (base = 100) with a DatetimeIndex.
    categories : dict[str, str]
        Mapping ``{ticker: category}``.
    output_path : str or Path
        File path at which to save the figure (PNG).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    axes_flat = axes.flatten()

    for idx, cat in enumerate(_CATEGORY_ORDER):
        ax = axes_flat[idx]
        tickers = [t for t in norm_prices.columns if categories.get(t) == cat]
        for ticker in tickers:
            if ticker in norm_prices.columns:
                ax.plot(norm_prices.index, norm_prices[ticker], label=ticker, linewidth=1.2)
        ax.set_title(f'{cat} ETFs – Normalised Prices (base = 100)', fontsize=11)
        ax.set_ylabel('Index (base = 100)')
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', labelrotation=30)

    fig.suptitle('Commodity ETF Normalised Prices by Category', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved normalised prices plot to %s.", output_path)


def plot_cumulative_returns_by_category(
    cum_returns: pd.DataFrame,
    categories: dict[str, str],
    output_path: str | Path,
) -> None:
    """Save a 2×2 figure of cumulative return paths, one panel per category.

    Parameters
    ----------
    cum_returns : pd.DataFrame
        Cumulative return paths (fraction, e.g. 0.10 = 10 %) with a DatetimeIndex.
    categories : dict[str, str]
        Mapping ``{ticker: category}``.
    output_path : str or Path
        File path at which to save the figure (PNG).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    axes_flat = axes.flatten()

    for idx, cat in enumerate(_CATEGORY_ORDER):
        ax = axes_flat[idx]
        tickers = [t for t in cum_returns.columns if categories.get(t) == cat]
        for ticker in tickers:
            if ticker in cum_returns.columns:
                pct = cum_returns[ticker] * 100
                ax.plot(cum_returns.index, pct, label=ticker, linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'{cat} ETFs – Cumulative Returns', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', labelrotation=30)

    fig.suptitle('Commodity ETF Cumulative Returns by Category', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved cumulative returns plot to %s.", output_path)


def plot_return_distributions(
    returns: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a 6×5 grid of per-asset return histograms overlaid with a KDE.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns with assets as columns.
    output_path : str or Path
        File path at which to save the figure (PNG).
    """
    tickers = list(returns.columns)
    n_cols = 5
    n_rows = 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 22))
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i >= len(tickers):
            ax.set_visible(False)
            continue
        ticker = tickers[i]
        data = returns[ticker].dropna()
        ax.hist(data, bins=50, density=True, alpha=0.55, color='steelblue', edgecolor='none')

        kde_x = np.linspace(data.min(), data.max(), 300)
        kde = scipy.stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color='crimson', linewidth=1.4)

        ax.set_title(ticker, fontsize=9, pad=3)
        ax.set_xlabel('Daily Return', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Commodity ETF Daily Return Distributions', fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved return distributions plot to %s.", output_path)
