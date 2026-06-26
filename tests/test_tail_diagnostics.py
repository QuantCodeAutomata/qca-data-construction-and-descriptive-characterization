"""Tests for src/tail_diagnostics.py — tail-risk metrics and Hill estimator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.tail_diagnostics import (
    compute_loss_series,
    compute_empirical_var,
    compute_empirical_cvar,
    compute_max_drawdown_from_returns,
    hill_estimator,
    compute_all_tail_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n: int = 1000, mean: float = 0.0005, std: float = 0.01,
                 seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(rng.normal(mean, std, n), index=dates, name='PORT')


# ---------------------------------------------------------------------------
# compute_loss_series
# ---------------------------------------------------------------------------

def test_loss_series_negation():
    """Loss must equal the negative of the return."""
    returns = _make_series(100)
    losses = compute_loss_series(returns)
    np.testing.assert_allclose(losses.values, -returns.values, atol=1e-12)


def test_loss_series_length():
    """Loss series must have same length as return series."""
    returns = _make_series(300)
    losses = compute_loss_series(returns)
    assert len(losses) == len(returns)


# ---------------------------------------------------------------------------
# compute_empirical_var
# ---------------------------------------------------------------------------

def test_var_is_positive_for_fat_tails():
    """VaR at 95% must be positive for a distribution centered near zero."""
    losses = compute_loss_series(_make_series(1000))
    var_95 = compute_empirical_var(losses, 0.95)
    assert var_95 > 0


def test_var_99_greater_than_95():
    """VaR at 99% must be greater than or equal to VaR at 95%."""
    losses = compute_loss_series(_make_series(2000))
    var_95 = compute_empirical_var(losses, 0.95)
    var_99 = compute_empirical_var(losses, 0.99)
    assert var_99 >= var_95 - 1e-8


def test_var_known_case():
    """For a uniform distribution losses in [0,1], VaR_95 = 0.95."""
    rng = np.random.default_rng(0)
    losses = pd.Series(rng.uniform(0, 1, 10000))
    var_95 = compute_empirical_var(losses, 0.95)
    assert abs(var_95 - 0.95) < 0.02


# ---------------------------------------------------------------------------
# compute_empirical_cvar (tail_diagnostics version)
# ---------------------------------------------------------------------------

def test_cvar_at_least_var():
    """CVaR must be >= VaR at the same confidence level."""
    losses = compute_loss_series(_make_series(1000))
    var_95 = compute_empirical_var(losses, 0.95)
    cvar_95 = compute_empirical_cvar(losses, 0.95)
    assert cvar_95 >= var_95 - 1e-8


def test_cvar_positive():
    """CVaR must be positive for a heavy-tailed loss distribution."""
    losses = compute_loss_series(_make_series(500))
    cvar = compute_empirical_cvar(losses, 0.95)
    assert cvar > 0


def test_cvar_99_vs_95():
    """CVaR_99 >= CVaR_95 for the same loss series."""
    losses = compute_loss_series(_make_series(2000))
    cvar_95 = compute_empirical_cvar(losses, 0.95)
    cvar_99 = compute_empirical_cvar(losses, 0.99)
    assert cvar_99 >= cvar_95 - 1e-8


# ---------------------------------------------------------------------------
# compute_max_drawdown_from_returns
# ---------------------------------------------------------------------------

def test_mdd_non_negative():
    """MDD must be >= 0."""
    returns = _make_series(500)
    mdd = compute_max_drawdown_from_returns(returns)
    assert mdd >= 0


def test_mdd_zero_for_always_positive():
    """If all returns are positive, MDD is 0."""
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    returns = pd.Series([0.005] * 100, index=dates)
    mdd = compute_max_drawdown_from_returns(returns)
    assert mdd < 1e-10


def test_mdd_bounded_by_one():
    """MDD (as fraction) cannot exceed 1.0."""
    returns = _make_series(800, std=0.03)
    mdd = compute_max_drawdown_from_returns(returns)
    assert mdd <= 1.0 + 1e-8


# ---------------------------------------------------------------------------
# hill_estimator
# ---------------------------------------------------------------------------

def test_hill_estimator_output_type():
    """hill_estimator must return two numpy arrays: (k_array, alpha_array)."""
    losses = compute_loss_series(_make_series(2000))
    ks, alphas = hill_estimator(losses)
    assert isinstance(alphas, np.ndarray)
    assert isinstance(ks, np.ndarray)


def test_hill_estimator_positive_values():
    """All Hill alpha estimates must be strictly positive (alpha > 0)."""
    losses = compute_loss_series(_make_series(2000))
    ks, alphas = hill_estimator(losses)
    assert (alphas > 0).all(), "Hill alpha must be strictly positive"


def test_hill_estimator_length_consistency():
    """Lengths of alphas and ks arrays must match."""
    losses = compute_loss_series(_make_series(2000))
    ks, alphas = hill_estimator(losses)
    assert len(alphas) == len(ks)


def test_hill_estimator_finite():
    """All Hill alpha estimates must be finite."""
    rng = np.random.default_rng(99)
    losses = pd.Series(np.abs(rng.standard_t(3, 3000)) * 0.01)
    ks, alphas = hill_estimator(losses)
    assert np.isfinite(alphas).all()


def test_hill_estimator_thin_tail():
    """For a Gaussian-like distribution, Hill alpha should be large (thin tail)."""
    rng = np.random.default_rng(77)
    losses = pd.Series(np.abs(rng.normal(0, 0.01, 5000)))
    ks, alphas = hill_estimator(losses)
    # For Gaussian tails, alpha should be high (> 2)
    assert np.median(alphas) > 2.0


# ---------------------------------------------------------------------------
# compute_all_tail_metrics
# ---------------------------------------------------------------------------

def test_all_tail_metrics_returns_dataframe():
    """compute_all_tail_metrics must return a DataFrame."""
    rng = np.random.default_rng(88)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    strategy_returns = {
        'Strategy_A': pd.Series(rng.normal(0.0005, 0.01, 500), index=dates),
        'Strategy_B': pd.Series(rng.normal(0.0003, 0.008, 500), index=dates),
    }
    metrics = compute_all_tail_metrics(strategy_returns)
    assert isinstance(metrics, pd.DataFrame)


def test_all_tail_metrics_row_count():
    """One row per strategy must be present."""
    rng = np.random.default_rng(55)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    strategy_returns = {
        f'S{i}': pd.Series(rng.normal(0, 0.01, 500), index=dates)
        for i in range(4)
    }
    metrics = compute_all_tail_metrics(strategy_returns)
    assert len(metrics) == 4


def test_all_tail_metrics_cvar_columns():
    """Output must contain CVaR and VaR columns."""
    rng = np.random.default_rng(22)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    strategy_returns = {
        'S1': pd.Series(rng.normal(0, 0.01, 500), index=dates),
    }
    metrics = compute_all_tail_metrics(strategy_returns)
    col_lower = [c.lower() for c in metrics.columns]
    assert any('cvar' in c or 'var' in c for c in col_lower), (
        "CVaR/VaR columns not found in tail metrics output"
    )
