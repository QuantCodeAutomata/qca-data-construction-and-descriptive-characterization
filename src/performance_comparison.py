import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional

try:
    from arch.bootstrap import MovingBlockBootstrap
    HAS_ARCH_BOOTSTRAP = True
except ImportError:
    HAS_ARCH_BOOTSTRAP = False

logger = logging.getLogger(__name__)
BLOCK_SIZE = 20
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# Scalar performance metrics
# ---------------------------------------------------------------------------

def compute_sharpe(returns: pd.Series, rf: pd.Series) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    rf : pd.Series
        Daily risk-free rate (same or larger index than *returns*).

    Returns
    -------
    float
        ``sqrt(252) * mean(excess) / std(excess)``.
    """
    rf_aligned = rf.reindex(returns.index).ffill().fillna(0.0)
    excess = returns - rf_aligned
    std = excess.std()
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


def compute_calmar(returns: pd.Series) -> float:
    """Calmar ratio (CAGR / maximum drawdown).

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        CAGR divided by the absolute maximum drawdown.  Returns 0 if MDD is 0.
    """
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    mdd = float(abs(drawdown.min()))
    if mdd == 0.0 or np.isnan(mdd):
        return 0.0
    n_years = len(returns) / 252.0
    if n_years <= 0:
        return 0.0
    cagr = float(cumulative.iloc[-1]) ** (1.0 / n_years) - 1.0
    return cagr / mdd


def compute_starr(returns: pd.Series, rf: pd.Series, alpha: float = 0.95) -> float:
    """Stable Tail Adjusted Return Ratio (STARR).

    ``sqrt(252) * mean(excess) / CVaR_{alpha}(losses)``

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    rf : pd.Series
        Daily risk-free rate.
    alpha : float
        Confidence level for CVaR (default 0.95).

    Returns
    -------
    float
        Annualised STARR.  Returns 0 if CVaR ≤ 0.
    """
    rf_aligned = rf.reindex(returns.index).ffill().fillna(0.0)
    excess = returns - rf_aligned
    losses = -excess
    var_level = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_level]
    cvar = float(tail.mean()) if len(tail) > 0 else var_level
    if cvar <= 0.0 or np.isnan(cvar):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / cvar)


# ---------------------------------------------------------------------------
# Net-of-cost returns
# ---------------------------------------------------------------------------

def apply_transaction_costs(
    returns: pd.Series,
    turnover: pd.Series,
    cost_rate: float,
) -> pd.Series:
    """Deduct proportional transaction costs from gross returns.

    Parameters
    ----------
    returns : pd.Series
        Gross daily portfolio returns.
    turnover : pd.Series
        One-way portfolio turnover on each date (sum of absolute weight changes).
    cost_rate : float
        Per-unit round-trip transaction cost (e.g. 0.001 for 10 bps).

    Returns
    -------
    pd.Series
        Net-of-cost daily returns: ``R_net = R - cost_rate * turnover``.
    """
    to_aligned = turnover.reindex(returns.index).fillna(0.0)
    return returns - cost_rate * to_aligned


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------

def moving_block_bootstrap_difference(
    series1: pd.Series,
    series2: pd.Series,
    rf: pd.Series,
    metric: str,
    block_size: int = BLOCK_SIZE,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple:
    """Bootstrap confidence interval for the difference in a performance metric.

    Uses the non-overlapping moving-block bootstrap (MBB) implemented manually.
    Each bootstrap resample is constructed by sampling overlapping blocks of
    length *block_size* with replacement.

    Parameters
    ----------
    series1, series2 : pd.Series
        Daily return series for two strategies, aligned on their common dates.
    rf : pd.Series
        Daily risk-free rate.
    metric : str
        One of ``'sharpe'``, ``'calmar'``, or ``'starr'``.
    block_size : int
        MBB block length.
    n_bootstrap : int
        Number of bootstrap replications.

    Returns
    -------
    tuple
        ``(obs_diff, ci_lower, ci_upper, is_significant)``

        * **obs_diff** – observed metric(series1) − metric(series2).
        * **ci_lower / ci_upper** – 95 % percentile bootstrap CI.
        * **is_significant** – True if CI excludes zero.
    """
    common_idx = series1.index.intersection(series2.index)
    s1 = series1.loc[common_idx].values
    s2 = series2.loc[common_idx].values
    rf_vals = rf.reindex(common_idx).ffill().fillna(0.0).values
    n = len(s1)

    def _metric(arr: np.ndarray, rf_arr: np.ndarray) -> float:
        r = pd.Series(arr, index=common_idx)
        rf_s = pd.Series(rf_arr, index=common_idx)
        if metric == 'sharpe':
            return compute_sharpe(r, rf_s)
        if metric == 'calmar':
            return compute_calmar(r)
        if metric == 'starr':
            return compute_starr(r, rf_s)
        return float(np.mean(arr))

    obs_diff = _metric(s1, rf_vals) - _metric(s2, rf_vals)

    # Build all overlapping blocks of length block_size
    blocks = [np.arange(i, min(i + block_size, n)) for i in range(n - block_size + 1)]
    n_blocks = len(blocks)
    n_blocks_needed = int(np.ceil(n / block_size))

    rng = np.random.default_rng(42)
    boot_diffs: list[float] = []

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_blocks, size=n_blocks_needed)
        idx = np.concatenate([blocks[b] for b in sampled])[:n]
        rf_boot = rf_vals[idx]
        boot_diffs.append(_metric(s1[idx], rf_boot) - _metric(s2[idx], rf_boot))

    boot_arr = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    is_significant = bool((ci_lower > 0) or (ci_upper < 0))

    return obs_diff, ci_lower, ci_upper, is_significant


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

def compute_rolling_sharpe(
    returns: pd.Series,
    rf: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Annualised rolling Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    rf : pd.Series
        Daily risk-free rate.
    window : int
        Rolling window in trading days (default 252).

    Returns
    -------
    pd.Series
        Rolling annualised Sharpe ratio.
    """
    rf_aligned = rf.reindex(returns.index).ffill().fillna(0.0)
    excess = returns - rf_aligned
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    return np.sqrt(252) * rolling_mean / rolling_std


# ---------------------------------------------------------------------------
# Full comparison analysis
# ---------------------------------------------------------------------------

def run_comparison_analysis(
    hist_returns: dict,
    dyn_returns: dict,
    hist_turnover: dict,
    dyn_turnover: dict,
    rf_series: pd.Series,
) -> tuple:
    """Compute gross and net performance tables and bootstrap significance tests.

    Parameters
    ----------
    hist_returns, dyn_returns : dict
        Mapping ``{strategy_name: pd.Series}`` of daily gross returns.
    hist_turnover, dyn_turnover : dict
        Mapping ``{strategy_name: pd.Series}`` of daily one-way turnover.
    rf_series : pd.Series
        Daily risk-free rate.

    Returns
    -------
    tuple
        ``(gross_df, net_df, bootstrap_dict)``

        * **gross_df** – DataFrame of Sharpe, Calmar, STARR-95 for all strategies.
        * **net_df** – DataFrame of net Sharpe at three transaction cost levels.
        * **bootstrap_dict** – ``{key: {obs_diff, ci_lower, ci_upper, is_significant}}``.
    """
    # ------------------------------------------------------------------ #
    # Gross metrics                                                        #
    # ------------------------------------------------------------------ #
    gross_rows = []
    all_rets = {
        **{'HIST_' + k: v for k, v in hist_returns.items()},
        **{'DYN_' + k: v for k, v in dyn_returns.items()},
    }
    for name, r in all_rets.items():
        gross_rows.append({
            'strategy': name,
            'sharpe':  compute_sharpe(r, rf_series),
            'calmar':  compute_calmar(r),
            'starr95': compute_starr(r, rf_series, alpha=0.95),
        })
    gross_df = pd.DataFrame(gross_rows).set_index('strategy')

    # ------------------------------------------------------------------ #
    # Net metrics at three cost levels                                     #
    # ------------------------------------------------------------------ #
    cost_rates = [0.0005, 0.001, 0.002]  # 5 bps, 10 bps, 20 bps
    net_rows = []
    for cost_rate in cost_rates:
        for name, r in hist_returns.items():
            to = hist_turnover.get(name, pd.Series(0.0, index=r.index))
            r_net = apply_transaction_costs(r, to, cost_rate)
            net_rows.append({
                'strategy': 'HIST_' + name,
                'cost_rate': cost_rate,
                'sharpe': compute_sharpe(r_net, rf_series),
            })
        for name, r in dyn_returns.items():
            to = dyn_turnover.get(name, pd.Series(0.0, index=r.index))
            r_net = apply_transaction_costs(r, to, cost_rate)
            net_rows.append({
                'strategy': 'DYN_' + name,
                'cost_rate': cost_rate,
                'sharpe': compute_sharpe(r_net, rf_series),
            })
    net_df = pd.DataFrame(net_rows)

    # ------------------------------------------------------------------ #
    # Bootstrap significance: DYN vs HIST for shared strategy names       #
    # ------------------------------------------------------------------ #
    bootstrap_dict: dict[str, dict] = {}
    shared = set(hist_returns.keys()) & set(dyn_returns.keys())
    for name in sorted(shared):
        key = f'DYN_vs_HIST_{name}_sharpe'
        try:
            obs, lo, hi, sig = moving_block_bootstrap_difference(
                dyn_returns[name], hist_returns[name], rf_series, 'sharpe'
            )
            bootstrap_dict[key] = {
                'obs_diff': float(obs),
                'ci_lower': float(lo),
                'ci_upper': float(hi),
                'is_significant': bool(sig),
            }
        except Exception as exc:
            logger.warning("Bootstrap failed for %s: %s", key, exc)

    return gross_df, net_df, bootstrap_dict


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rolling_sharpe_comparison(
    hist_returns: dict,
    dyn_returns: dict,
    rf_series: pd.Series,
    strategies: list,
    output_path: Path,
) -> None:
    """Plot rolling 252-day Sharpe ratios for historical vs dynamic strategies.

    Parameters
    ----------
    hist_returns, dyn_returns : dict
        Mapping ``{strategy_name: pd.Series}`` of gross daily returns.
    rf_series : pd.Series
        Daily risk-free rate.
    strategies : list
        Strategy names to include (must appear in at least one of the dicts).
    output_path : Path
        Destination file path for the saved PNG.
    """
    n = max(len(strategies), 1)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        if strat in hist_returns:
            rs = compute_rolling_sharpe(hist_returns[strat], rf_series)
            ax.plot(rs.index, rs.values, label='Historical', linewidth=1.0)
        if strat in dyn_returns:
            rs = compute_rolling_sharpe(dyn_returns[strat], rf_series)
            ax.plot(rs.index, rs.values, label='Dynamic', linewidth=1.0, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_title(f'{strat}: Rolling Sharpe (252-day)', fontsize=10)
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved rolling Sharpe comparison → %s", output_path)


def plot_cost_sensitivity(
    gross_metrics: pd.DataFrame,
    net_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot net Sharpe ratio as a function of transaction cost for all strategies.

    Parameters
    ----------
    gross_metrics : pd.DataFrame
        Gross performance table (index = strategy names).
    net_metrics : pd.DataFrame
        Net performance table with columns ``['strategy', 'cost_rate', 'sharpe']``.
    output_path : Path
        Destination file path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    cost_rates = sorted(net_metrics['cost_rate'].unique())
    costs_bps = [c * 10_000 for c in cost_rates]

    for strat in net_metrics['strategy'].unique():
        sub = net_metrics[net_metrics['strategy'] == strat]
        sharpes = []
        for c in cost_rates:
            rows = sub[sub['cost_rate'] == c]
            sharpes.append(float(rows['sharpe'].iloc[0]) if len(rows) > 0 else np.nan)
        ax.plot(costs_bps[:len(sharpes)], sharpes, marker='o', label=strat, linewidth=1.0)

    ax.set_xlabel('Transaction Cost (bps)')
    ax.set_ylabel('Net Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Transaction Costs')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved cost sensitivity plot → %s", output_path)
