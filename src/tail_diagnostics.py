import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


def compute_loss_series(returns):
    """Convert a return series to a loss series.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio return series.

    Returns
    -------
    pd.Series
        Loss series ``-returns`` (positive values indicate losses), preserving
        the original index.
    """
    return -returns


def compute_empirical_var(losses, alpha):
    """Compute the empirical Value-at-Risk from a loss series.

    Parameters
    ----------
    losses : np.ndarray
        Array of loss observations (positive = loss).
    alpha : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    float
        VaR at confidence level ``alpha``.
    """
    return float(np.quantile(losses, alpha))


def compute_empirical_cvar(losses, alpha):
    """Compute the empirical CVaR (Expected Shortfall) from a loss series.

    CVaR is defined as the mean of losses that meet or exceed the VaR
    threshold.

    Parameters
    ----------
    losses : np.ndarray
        Array of loss observations (positive = loss).
    alpha : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    float
        CVaR at confidence level ``alpha``.
    """
    var = compute_empirical_var(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def compute_max_drawdown_from_returns(returns):
    """Compute maximum drawdown from a cumulative return path.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.

    Returns
    -------
    float
        Maximum drawdown expressed as a positive decimal (e.g. 0.25 = 25 %).
    """
    cumval = (1 + returns).cumprod()
    running_max = cumval.cummax()
    dd = (running_max - cumval) / running_max
    return float(dd.max())


def hill_estimator(losses, k_max_frac=0.05):
    """Compute the Hill tail-index estimator over a range of threshold values.

    For an ordered sample of positive losses ``X[0] >= X[1] >= ... >= X[m-1]``,
    the Hill estimator at order statistic ``k`` is:

        alpha_hat(k) = k / sum_{i=0}^{k-1} log(X[i] / X[k])

    Parameters
    ----------
    losses : np.ndarray
        Full loss series (may include non-positive values; these are filtered out).
    k_max_frac : float, optional
        Fraction of the positive-loss sample to use as the upper k limit
        (default 0.05, i.e. the top 5 %).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(k_array, alpha_hat_array)`` where each entry corresponds to the Hill
        estimate computed using the ``k`` largest order statistics.  Both arrays
        are empty if the sample is too small.
    """
    X = np.sort(losses[losses > 0])[::-1]
    m = len(X)
    if m < 10:
        return np.array([]), np.array([])

    m5 = int(math.floor(k_max_frac * m))
    k_min = int(max(5, math.ceil(0.01 * m)))

    k_arr = []
    alpha_arr = []

    for k in range(k_min, m5 + 1):
        if k >= m:
            break
        log_ratios = np.log(X[:k] / X[k])
        if np.any(log_ratios <= 0) or log_ratios.mean() == 0:
            continue
        alpha_hat = 1.0 / log_ratios.mean()
        k_arr.append(k)
        alpha_arr.append(alpha_hat)

    return np.array(k_arr), np.array(alpha_arr)


def compute_all_tail_metrics(strategy_returns):
    """Compute tail-risk metrics and Hill tail indices for all strategies.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Map from strategy name to daily return series.

    Returns
    -------
    pd.DataFrame
        One row per strategy with columns:
        ``VaR_95``, ``CVaR_95``, ``VaR_99``, ``CVaR_99``, ``MDD``,
        ``hill_alpha_mean``, ``hill_alpha_median`` (VaR/CVaR/MDD in percent).
    """
    rows = []
    for name, ret_series in strategy_returns.items():
        losses = compute_loss_series(ret_series)
        var_95 = compute_empirical_var(losses, 0.95)
        cvar_95 = compute_empirical_cvar(losses, 0.95)
        var_99 = compute_empirical_var(losses, 0.99)
        cvar_99 = compute_empirical_cvar(losses, 0.99)
        mdd = compute_max_drawdown_from_returns(ret_series)

        k_arr, alpha_arr = hill_estimator(losses)
        hill_mean = float(alpha_arr.mean()) if len(alpha_arr) > 0 else np.nan
        hill_median = float(np.median(alpha_arr)) if len(alpha_arr) > 0 else np.nan

        rows.append({
            'strategy': name,
            'VaR_95': var_95 * 100,
            'CVaR_95': cvar_95 * 100,
            'VaR_99': var_99 * 100,
            'CVaR_99': cvar_99 * 100,
            'MDD': mdd * 100,
            'hill_alpha_mean': hill_mean,
            'hill_alpha_median': hill_median,
        })

    return pd.DataFrame(rows).set_index('strategy')


def plot_hill_curves(strategy_returns, output_path, strategies_subset=None):
    """Plot Hill tail-index estimates as a function of the order statistic k.

    Each line shows how the estimated tail index ``alpha_hat`` varies with the
    number of upper order statistics ``k`` used.  A thicker tail corresponds to
    a lower Hill alpha.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Map from strategy name to daily return series.
    output_path : str or Path
        File path at which to save the figure (PNG).
    strategies_subset : list[str] or None, optional
        Subset of strategy names to plot.  All strategies are used if None.
    """
    names = strategies_subset if strategies_subset is not None else list(strategy_returns.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        if name not in strategy_returns:
            continue
        losses = compute_loss_series(strategy_returns[name])
        k_arr, alpha_arr = hill_estimator(losses)
        if len(k_arr) == 0:
            logger.warning("Hill estimator returned no values for strategy %s.", name)
            continue
        ax.plot(k_arr, alpha_arr, label=name, linewidth=1.5)

    ax.set_xlabel('k (number of upper order statistics)')
    ax.set_ylabel('Hill tail-index estimate α̂(k)')
    ax.set_title('Hill Tail-Index Estimates by Strategy')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved Hill curves to %s.", output_path)


def plot_tail_metrics_comparison(metrics_df, output_path):
    """Bar chart comparing VaR_95, CVaR_95, and MDD across strategies.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`compute_all_tail_metrics` (strategies as index).
    output_path : str or Path
        File path at which to save the figure (PNG).
    """
    cols = [c for c in ['VaR_95', 'CVaR_95', 'MDD'] if c in metrics_df.columns]
    plot_df = metrics_df[cols].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.25
    offsets = np.linspace(-(len(cols) - 1) * width / 2, (len(cols) - 1) * width / 2, len(cols))

    for offset, col in zip(offsets, cols):
        ax.bar(x + offset, plot_df[col].values, width, label=col, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Value (%)')
    ax.set_title('Tail Risk Metrics by Strategy: VaR₉₅, CVaR₉₅, MDD')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved tail metrics comparison to %s.", output_path)
