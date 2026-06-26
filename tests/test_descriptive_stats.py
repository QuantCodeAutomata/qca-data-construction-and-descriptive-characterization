"""Tests for src/descriptive_stats.py — summary statistics and normalizations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.descriptive_stats import (
    compute_summary_statistics,
    compute_correlation_matrix,
    normalize_prices,
    compute_cumulative_returns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_returns(n: int = 300, k: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (n, k)),
        index=dates,
        columns=[f'ETF{i}' for i in range(k)],
    )


def _make_prices(n: int = 300, k: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    raw = np.cumprod(1 + rng.normal(0.0005, 0.01, (n, k)), axis=0) * 100
    return pd.DataFrame(raw, index=dates, columns=[f'ETF{i}' for i in range(k)])


# ---------------------------------------------------------------------------
# compute_summary_statistics
# ---------------------------------------------------------------------------

def test_summary_statistics_row_count():
    """Must return one row per ETF."""
    rets = _make_returns(300, 6)
    stats = compute_summary_statistics(rets)
    assert len(stats) == 6


def test_summary_statistics_required_columns():
    """Must contain mean, std, skewness, and kurtosis columns."""
    rets = _make_returns()
    stats = compute_summary_statistics(rets)
    expected = {'mean_daily', 'std_daily', 'skewness', 'excess_kurtosis'}
    assert expected.issubset(set(stats.columns)), (
        f"Missing columns: {expected - set(stats.columns)}"
    )


def test_summary_statistics_volatility_positive():
    """Sample standard deviation must be strictly positive."""
    rets = _make_returns()
    stats = compute_summary_statistics(rets)
    assert (stats['std_daily'] > 0).all()


def test_summary_statistics_single_asset():
    """Should work with a single-asset DataFrame."""
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    rets = pd.DataFrame({'X': np.random.randn(100) * 0.01}, index=dates)
    stats = compute_summary_statistics(rets)
    assert len(stats) == 1


def test_summary_statistics_known_skew():
    """Perfectly symmetric returns should yield near-zero skewness."""
    dates = pd.date_range('2020-01-01', periods=1000, freq='B')
    r = np.concatenate([-np.ones(500) * 0.01, np.ones(500) * 0.01])
    rets = pd.DataFrame({'A': r}, index=dates)
    stats = compute_summary_statistics(rets)
    skew_col = [c for c in stats.columns if 'skew' in c.lower()][0]
    assert abs(stats[skew_col].iloc[0]) < 0.1


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------

def test_correlation_matrix_shape():
    """Correlation matrix must be square k×k."""
    rets = _make_returns(300, 7)
    corr = compute_correlation_matrix(rets)
    assert corr.shape == (7, 7)


def test_correlation_matrix_diagonal_ones():
    """Diagonal entries must equal 1.0."""
    rets = _make_returns()
    corr = compute_correlation_matrix(rets)
    for i in range(len(corr)):
        assert abs(corr.iloc[i, i] - 1.0) < 1e-10


def test_correlation_matrix_bounds():
    """All entries must lie in [-1, 1]."""
    rets = _make_returns()
    corr = compute_correlation_matrix(rets)
    assert (corr.values >= -1.0).all() and (corr.values <= 1.0).all()


def test_correlation_matrix_symmetric():
    """Correlation matrix must be symmetric."""
    rets = _make_returns()
    corr = compute_correlation_matrix(rets)
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-12)


def test_perfect_correlation():
    """Two identical series must yield correlation = 1.0."""
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    r = np.random.randn(200) * 0.01
    rets = pd.DataFrame({'A': r, 'B': r}, index=dates)
    corr = compute_correlation_matrix(rets)
    assert abs(corr.loc['A', 'B'] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# normalize_prices
# ---------------------------------------------------------------------------

def test_normalize_prices_base_value():
    """First row of normalized prices must equal base_value (default 100)."""
    prices = _make_prices(200)
    normed = normalize_prices(prices)
    assert (normed.iloc[0] == 100.0).all()


def test_normalize_prices_custom_base():
    """Custom base_value must be reflected on the first row."""
    prices = _make_prices(200)
    normed = normalize_prices(prices, base_value=1.0)
    np.testing.assert_allclose(normed.iloc[0].values, np.ones(5), atol=1e-10)


def test_normalize_prices_shape_preserved():
    """Shape must be unchanged after normalization."""
    prices = _make_prices(200, 8)
    normed = normalize_prices(prices)
    assert normed.shape == prices.shape


def test_normalize_prices_monotone_consistent():
    """If original prices increase, normalized series must also increase."""
    dates = pd.date_range('2020-01-01', periods=5, freq='B')
    prices = pd.DataFrame({'A': [100.0, 110.0, 120.0, 130.0, 140.0]}, index=dates)
    normed = normalize_prices(prices)
    assert normed['A'].is_monotonic_increasing


# ---------------------------------------------------------------------------
# compute_cumulative_returns
# ---------------------------------------------------------------------------

def test_cumulative_returns_first_row_equals_first_return():
    """First cumulative return value must equal the first daily return."""
    rets = _make_returns(200)
    cum = compute_cumulative_returns(rets)
    np.testing.assert_allclose(cum.iloc[0].values, rets.iloc[0].values, atol=1e-12)


def test_cumulative_returns_shape():
    """Shape must be identical to input returns."""
    rets = _make_returns(200, 6)
    cum = compute_cumulative_returns(rets)
    assert cum.shape == rets.shape


def test_cumulative_returns_zero_input():
    """Zero daily returns must produce cumulative return of 0 at every row."""
    dates = pd.date_range('2020-01-01', periods=10, freq='B')
    rets = pd.DataFrame({'A': np.zeros(10)}, index=dates)
    cum = compute_cumulative_returns(rets)
    np.testing.assert_allclose(cum['A'].values, np.zeros(10), atol=1e-12)


def test_cumulative_returns_direction():
    """Positive daily returns must yield increasing cumulative return series."""
    dates = pd.date_range('2020-01-01', periods=10, freq='B')
    rets = pd.DataFrame({'A': np.full(10, 0.01)}, index=dates)
    cum = compute_cumulative_returns(rets)
    assert cum['A'].is_monotonic_increasing
