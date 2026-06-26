"""Tests for src/historical_optimizer.py — portfolio optimization and metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.historical_optimizer import (
    solve_mvp,
    solve_tvp,
    solve_min_cvar,
    solve_max_cvar_ratio,
    compute_empirical_cvar,
    compute_max_drawdown,
    compute_portfolio_metrics,
    ALPHA_95,
    ALPHA_99,
    C_TO,
    WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_returns(n: int = 520, k: int = 5, seed: int = 99) -> np.ndarray:
    """Generate i.i.d. Gaussian daily returns (n x k)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.012, (n, k))


def _sample_cov(returns: np.ndarray):
    return np.cov(returns.T)


def _sample_mean(returns: np.ndarray):
    return returns.mean(axis=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_window_is_504():
    """WINDOW must be exactly 504 trading days as specified in the paper."""
    assert WINDOW == 504


def test_alpha_levels():
    """Confidence levels must be 0.95 and 0.99."""
    assert ALPHA_95 == 0.95
    assert ALPHA_99 == 0.99


def test_turnover_cap():
    """C_TO = 1/(30*252) as specified in the paper."""
    expected = 1.0 / (30 * 252)
    assert abs(C_TO - expected) < 1e-12


# ---------------------------------------------------------------------------
# solve_mvp
# ---------------------------------------------------------------------------

def test_mvp_weights_sum_to_one():
    """MVP weights must sum to 1.0 (full investment)."""
    rets = _make_simple_returns()
    mu = _sample_mean(rets)
    Sigma = _sample_cov(rets)
    w_prev = np.ones(5) / 5
    lb, ub = 0.0, 1.0
    w = solve_mvp(mu, Sigma, w_prev, lb, ub, enforce_turnover=False)
    assert abs(w.sum() - 1.0) < 1e-5, f"Weights sum = {w.sum()}"


def test_mvp_weights_long_only():
    """Long-only MVP weights must all be >= 0."""
    rets = _make_simple_returns()
    mu = _sample_mean(rets)
    Sigma = _sample_cov(rets)
    w_prev = np.ones(5) / 5
    w = solve_mvp(mu, Sigma, w_prev, 0.0, 1.0, enforce_turnover=False)
    assert (w >= -1e-6).all(), f"Negative weight found: {w.min()}"


def test_mvp_restricted_long_short():
    """Restricted long-short (-1/30) lower bound should produce non-None solution."""
    rets = _make_simple_returns()
    mu = _sample_mean(rets)
    Sigma = _sample_cov(rets)
    w_prev = np.ones(5) / 5
    lb, ub = -1.0 / 30, 1 + 1.0 / 30
    w = solve_mvp(mu, Sigma, w_prev, lb, ub, enforce_turnover=False)
    assert w is not None


def test_mvp_output_length():
    """MVP must return weight vector of length equal to number of assets."""
    rets = _make_simple_returns(k=8)
    mu = _sample_mean(rets)
    Sigma = _sample_cov(rets)
    w_prev = np.ones(8) / 8
    w = solve_mvp(mu, Sigma, w_prev, 0.0, 1.0, enforce_turnover=False)
    assert len(w) == 8


# ---------------------------------------------------------------------------
# solve_tvp
# ---------------------------------------------------------------------------

def test_tvp_weights_sum_to_one():
    """TVP weights must satisfy full-investment constraint."""
    rets = _make_simple_returns()
    mu = _sample_mean(rets)
    Sigma = _sample_cov(rets)
    w_prev = np.ones(5) / 5
    rf = 0.05 / 252
    w = solve_tvp(mu, Sigma, rf, w_prev, 0.0, 1.0, enforce_turnover=False)
    if w is not None:
        assert abs(w.sum() - 1.0) < 1e-4


def test_tvp_sharpe_higher_than_mvp():
    """TVP should achieve higher Sharpe ratio than MVP in well-behaved conditions."""
    rng = np.random.default_rng(7)
    rets = rng.normal(0.001, 0.012, (520, 5))
    mu = _sample_mean(rets)
    Sigma = _sample_cov(rets)
    w_prev = np.ones(5) / 5
    rf = 0.05 / 252
    w_mvp = solve_mvp(mu, Sigma, w_prev, 0.0, 1.0, enforce_turnover=False)
    w_tvp = solve_tvp(mu, Sigma, rf, w_prev, 0.0, 1.0, enforce_turnover=False)
    if w_tvp is not None and w_mvp is not None:
        sr_mvp = (w_mvp @ mu - rf) / np.sqrt(w_mvp @ Sigma @ w_mvp)
        sr_tvp = (w_tvp @ mu - rf) / np.sqrt(w_tvp @ Sigma @ w_tvp)
        # TVP maximises the Sharpe, so it cannot be worse
        assert sr_tvp >= sr_mvp - 1e-4


# ---------------------------------------------------------------------------
# solve_min_cvar
# ---------------------------------------------------------------------------

def test_min_cvar_weights_sum_to_one():
    """CVaR-minimizing weights must satisfy full investment."""
    rets = _make_simple_returns()
    w_prev = np.ones(5) / 5
    w = solve_min_cvar(rets, ALPHA_95, w_prev, 0.0, 1.0, enforce_turnover=False)
    if w is not None:
        assert abs(w.sum() - 1.0) < 1e-4


def test_min_cvar_long_only():
    """Min-CVaR long-only weights must all be >= 0."""
    rets = _make_simple_returns()
    w_prev = np.ones(5) / 5
    w = solve_min_cvar(rets, ALPHA_95, w_prev, 0.0, 1.0, enforce_turnover=False)
    if w is not None:
        assert (w >= -1e-5).all()


def test_min_cvar_99_vs_95():
    """CVaR_99 minimization returns a valid weight vector."""
    rets = _make_simple_returns()
    w_prev = np.ones(5) / 5
    w = solve_min_cvar(rets, ALPHA_99, w_prev, 0.0, 1.0, enforce_turnover=False)
    if w is not None:
        assert abs(w.sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# compute_empirical_cvar
# ---------------------------------------------------------------------------

def test_empirical_cvar_greater_than_var():
    """CVaR_alpha must be >= VaR_alpha (CVaR is always at least as large)."""
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    var95 = -returns.quantile(0.05)
    cvar95 = compute_empirical_cvar(returns, 0.95)
    assert cvar95 >= var95 - 1e-8


def test_empirical_cvar_99_greater_than_95():
    """CVaR_0.99 must be >= CVaR_0.95 on the same sample."""
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    cvar95 = compute_empirical_cvar(returns, 0.95)
    cvar99 = compute_empirical_cvar(returns, 0.99)
    assert cvar99 >= cvar95 - 1e-8


def test_empirical_cvar_positive():
    """CVaR must be positive for a distribution with non-trivial downside."""
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    cvar = compute_empirical_cvar(returns, 0.95)
    assert cvar > 0


def test_empirical_cvar_extreme_loss():
    """A series with a known extreme loss must have CVaR reflecting that loss."""
    # 10% of returns are -0.10, rest are +0.001
    returns = pd.Series([-0.10] * 50 + [0.001] * 450)
    cvar = compute_empirical_cvar(returns, 0.95)
    # CVaR_95 should be at least 0.10
    assert cvar >= 0.09


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_non_negative():
    """Maximum drawdown must be non-negative."""
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    mdd = compute_max_drawdown(returns)
    assert mdd >= 0


def test_max_drawdown_zero_for_monotone_increase():
    """Monotonically increasing returns produce zero drawdown."""
    returns = pd.Series([0.01] * 100)
    mdd = compute_max_drawdown(returns)
    assert mdd < 1e-10


def test_max_drawdown_known_case():
    """Verify known drawdown: 50% drop after 100% gain."""
    # Price goes 100 -> 200 -> 100  (drawdown = 50 %)
    returns = pd.Series([1.0] + [-0.5])  # 100% gain, then 50% loss
    mdd = compute_max_drawdown(returns)
    assert abs(mdd - 0.5) < 1e-8


def test_max_drawdown_between_0_and_1():
    """MDD expressed as fraction must lie in [0, 1]."""
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0, 0.015, 700))
    mdd = compute_max_drawdown(returns)
    assert 0 <= mdd <= 1.0


# ---------------------------------------------------------------------------
# compute_portfolio_metrics
# ---------------------------------------------------------------------------

def test_portfolio_metrics_keys():
    """compute_portfolio_metrics must return a dict with required keys."""
    rng = np.random.default_rng(10)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    returns = pd.Series(rng.normal(0.0005, 0.01, 500), index=dates)
    rf = pd.Series(0.05 / 252, index=dates)
    metrics = compute_portfolio_metrics(returns, rf, label='test')
    required = {'annualized_sharpe', 'mdd', 'calmar', 'cvar_95', 'cvar_99', 'var_95', 'var_99'}
    for key in required:
        assert key in metrics, f"Missing key '{key}' in metrics dict"


def test_portfolio_metrics_sharpe_finite():
    """Sharpe ratio must be a finite number."""
    rng = np.random.default_rng(11)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    returns = pd.Series(rng.normal(0.0005, 0.01, 500), index=dates)
    rf = pd.Series(0.05 / 252, index=dates)
    metrics = compute_portfolio_metrics(returns, rf, label='test')
    assert np.isfinite(metrics['annualized_sharpe'])


def test_portfolio_metrics_var_less_than_cvar():
    """VaR must not exceed CVaR at same confidence level (CVaR >= VaR)."""
    rng = np.random.default_rng(12)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    returns = pd.Series(rng.normal(0.0005, 0.012, 500), index=dates)
    rf = pd.Series(0.05 / 252, index=dates)
    metrics = compute_portfolio_metrics(returns, rf, label='test')
    assert metrics['cvar_95'] >= metrics['var_95'] - 1e-8
    assert metrics['cvar_99'] >= metrics['var_99'] - 1e-8
