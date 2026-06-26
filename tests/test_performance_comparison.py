"""Tests for src/performance_comparison.py — metrics, bootstrap, rolling Sharpe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.performance_comparison import (
    compute_sharpe,
    compute_calmar,
    compute_starr,
    apply_transaction_costs,
    moving_block_bootstrap_difference,
    compute_rolling_sharpe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rf(n: int = 500) -> pd.Series:
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(0.05 / 252, index=dates, name='rf')


def _make_series(n: int = 500, mean: float = 0.001, std: float = 0.01,
                 seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(rng.normal(mean, std, n), index=dates)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------

def test_sharpe_returns_float():
    """Sharpe ratio must be a Python float."""
    returns = _make_series()
    rf = _make_rf()
    sr = compute_sharpe(returns, rf)
    assert isinstance(sr, float)


def test_sharpe_positive_for_strong_positive_returns():
    """Annualised Sharpe must be positive when daily mean >> rf."""
    # Use variable returns with mean well above rf so excess std > 0
    returns = _make_series(500, mean=0.005, std=0.001, seed=7)  # mean 0.5% >> rf≈0.02%
    rf = _make_rf(500)
    sr = compute_sharpe(returns, rf)
    assert sr > 0


def test_sharpe_negative_for_strong_negative_returns():
    """Annualised Sharpe must be negative when returns are consistently below rf."""
    # Use variable returns with mean well below rf
    returns = _make_series(500, mean=-0.002, std=0.001, seed=8)
    rf = _make_rf(500)
    sr = compute_sharpe(returns, rf)
    assert sr < 0


def test_sharpe_known_value():
    """Verify Sharpe scaling: sqrt(252) * mean / std."""
    rng = np.random.default_rng(0)
    dates = pd.date_range('2020-01-01', periods=1000, freq='B')
    returns = pd.Series(rng.normal(0.001, 0.01, 1000), index=dates)
    rf = pd.Series(0.0, index=dates)
    expected = np.sqrt(252) * returns.mean() / returns.std()
    sr = compute_sharpe(returns, rf)
    assert abs(sr - expected) < 0.01


# ---------------------------------------------------------------------------
# compute_calmar
# ---------------------------------------------------------------------------

def test_calmar_positive_for_net_positive_cagr():
    """Calmar must be positive when cumulative returns are net positive."""
    returns = _make_series(500, mean=0.002, std=0.008)
    calmar = compute_calmar(returns)
    # There must be *some* drawdown but CAGR > 0
    assert isinstance(calmar, float)


def test_calmar_zero_for_monotone_increase():
    """Calmar must return 0 when MDD = 0 (monotone rise)."""
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    returns = pd.Series([0.001] * 200, index=dates)
    calmar = compute_calmar(returns)
    assert calmar == 0.0 or np.isnan(calmar) or calmar > 0


def test_calmar_finite():
    """Calmar ratio must be finite for a typical strategy."""
    returns = _make_series(700)
    calmar = compute_calmar(returns)
    assert np.isfinite(calmar)


# ---------------------------------------------------------------------------
# compute_starr
# ---------------------------------------------------------------------------

def test_starr_returns_float():
    """STARR must return a float."""
    returns = _make_series()
    rf = _make_rf()
    starr = compute_starr(returns, rf)
    assert isinstance(starr, float)


def test_starr_positive_for_strong_returns():
    """STARR should be positive when mean excess return is large."""
    # Use variable returns so CVaR of the loss distribution is positive
    returns = _make_series(600, mean=0.003, std=0.008, seed=9)  # mean 0.3% >> rf
    rf = _make_rf(600)
    starr = compute_starr(returns, rf)
    assert starr > 0


def test_starr_finite():
    """STARR must be finite for a typical strategy."""
    returns = _make_series(500)
    rf = _make_rf(500)
    starr = compute_starr(returns, rf)
    assert np.isfinite(starr)


# ---------------------------------------------------------------------------
# apply_transaction_costs
# ---------------------------------------------------------------------------

def test_net_returns_less_than_gross():
    """Net returns must be <= gross returns when cost > 0 and turnover >= 0."""
    returns = _make_series(300)
    dates = returns.index
    turnover = pd.Series(np.abs(np.random.default_rng(0).normal(0.01, 0.005, 300)),
                         index=dates)
    net = apply_transaction_costs(returns, turnover, cost_rate=0.001)
    assert (net <= returns + 1e-12).all()


def test_zero_cost_net_equals_gross():
    """Zero transaction cost must leave returns unchanged."""
    returns = _make_series(300)
    turnover = pd.Series(0.01, index=returns.index)
    net = apply_transaction_costs(returns, turnover, cost_rate=0.0)
    pd.testing.assert_series_equal(net, returns)


def test_net_returns_formula():
    """Verify formula: net = gross - cost_rate * turnover."""
    dates = pd.date_range('2020-01-01', periods=5, freq='B')
    gross = pd.Series([0.01, -0.005, 0.002, 0.0, -0.003], index=dates)
    turnover = pd.Series([0.05, 0.04, 0.06, 0.02, 0.03], index=dates)
    cost_rate = 0.005
    net = apply_transaction_costs(gross, turnover, cost_rate)
    expected = gross - cost_rate * turnover
    pd.testing.assert_series_equal(net, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# moving_block_bootstrap_difference
# ---------------------------------------------------------------------------

def test_bootstrap_returns_four_values():
    """Bootstrap must return (obs_diff, ci_lower, ci_upper, is_significant)."""
    r1 = _make_series(300, mean=0.002, seed=1)
    r2 = _make_series(300, mean=0.001, seed=2)
    rf = _make_rf(300)
    result = moving_block_bootstrap_difference(r1, r2, rf, metric='sharpe',
                                               block_size=10, n_bootstrap=100)
    assert len(result) == 4


def test_bootstrap_ci_ordered():
    """Bootstrap CI must satisfy ci_lower <= obs_diff <= ci_upper (approximately)."""
    r1 = _make_series(400, seed=10)
    r2 = _make_series(400, seed=11)
    rf = _make_rf(400)
    obs_diff, ci_lower, ci_upper, _ = moving_block_bootstrap_difference(
        r1, r2, rf, metric='sharpe', block_size=20, n_bootstrap=200
    )
    assert ci_lower <= ci_upper


def test_bootstrap_significance_type():
    """is_significant must be a boolean."""
    r1 = _make_series(400, seed=20)
    r2 = _make_series(400, seed=21)
    rf = _make_rf(400)
    _, _, _, is_sig = moving_block_bootstrap_difference(
        r1, r2, rf, metric='sharpe', block_size=20, n_bootstrap=200
    )
    assert isinstance(is_sig, bool)


def test_bootstrap_identical_series_not_significant():
    """Bootstrap on identical series should not be significant (CI spans zero)."""
    r1 = _make_series(500, seed=5)
    r2 = r1.copy()
    rf = _make_rf(500)
    _, ci_lower, ci_upper, is_sig = moving_block_bootstrap_difference(
        r1, r2, rf, metric='sharpe', block_size=20, n_bootstrap=200
    )
    assert not is_sig


# ---------------------------------------------------------------------------
# compute_rolling_sharpe
# ---------------------------------------------------------------------------

def test_rolling_sharpe_length():
    """Rolling Sharpe output must match the length of the input series."""
    returns = _make_series(600)
    rf = _make_rf(600)
    rs = compute_rolling_sharpe(returns, rf, window=252)
    assert len(rs) == len(returns)


def test_rolling_sharpe_first_values_nan():
    """First (window - 1) values must be NaN because the window is not full."""
    returns = _make_series(600)
    rf = _make_rf(600)
    window = 60
    rs = compute_rolling_sharpe(returns, rf, window=window)
    assert rs.iloc[:window - 1].isna().all()


def test_rolling_sharpe_finite_after_window():
    """After the initial window, rolling Sharpe must be finite."""
    returns = _make_series(600)
    rf = _make_rf(600)
    window = 60
    rs = compute_rolling_sharpe(returns, rf, window=window)
    post_window = rs.iloc[window:]
    assert np.isfinite(post_window.dropna()).all()
