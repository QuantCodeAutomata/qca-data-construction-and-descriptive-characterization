"""Tests for src/data_loader.py — data construction and panel building."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    ETF_UNIVERSE,
    ETF_CATEGORIES,
    ALL_TICKERS,
    compute_arithmetic_returns,
    convert_tbill_to_daily,
    build_balanced_panel,
    get_audit_table,
    STUDY_START,
    STUDY_END,
)


# ---------------------------------------------------------------------------
# Universe constants
# ---------------------------------------------------------------------------

def test_etf_universe_total_count():
    """The universe must contain exactly 30 ETFs as specified in the paper."""
    assert len(ALL_TICKERS) == 30, f"Expected 30 tickers, got {len(ALL_TICKERS)}"


def test_etf_universe_category_counts():
    """Agriculture=3, Energy=8 (was listed as 8 in code: OILK USL UGA BNO DBE DBO USO UNG),
    Metals=10, Broad=9."""
    expected = {'Agriculture': 3, 'Energy': 8, 'Metals': 10, 'Broad': 9}
    for cat, count in expected.items():
        assert len(ETF_UNIVERSE[cat]) == count, (
            f"Category {cat}: expected {count}, got {len(ETF_UNIVERSE[cat])}"
        )


def test_all_tickers_in_categories():
    """Every ticker in ALL_TICKERS must have a category assignment."""
    for t in ALL_TICKERS:
        assert t in ETF_CATEGORIES, f"Ticker {t} not found in ETF_CATEGORIES"


def test_no_duplicate_tickers():
    """No ticker should appear in more than one category."""
    assert len(ALL_TICKERS) == len(set(ALL_TICKERS)), "Duplicate tickers detected"


def test_study_dates_format():
    """Study dates must be parseable as YYYY-MM-DD."""
    pd.Timestamp(STUDY_START)
    pd.Timestamp(STUDY_END)


# ---------------------------------------------------------------------------
# compute_arithmetic_returns
# ---------------------------------------------------------------------------

def test_arithmetic_returns_shape():
    """Returns should have one fewer row than prices."""
    dates = pd.date_range('2020-01-01', periods=10, freq='B')
    prices = pd.DataFrame(
        np.random.rand(10, 3) * 100 + 50, index=dates, columns=['A', 'B', 'C']
    )
    rets = compute_arithmetic_returns(prices)
    assert rets.shape == (9, 3)


def test_arithmetic_returns_formula():
    """r_t = P_t / P_{t-1} - 1 must hold exactly."""
    dates = pd.date_range('2020-01-01', periods=5, freq='B')
    prices = pd.DataFrame({'A': [100.0, 110.0, 99.0, 105.0, 100.0]}, index=dates)
    rets = compute_arithmetic_returns(prices)
    expected = [0.10, -0.10, 6/99, -5/105]
    for i, exp_val in enumerate(expected):
        assert abs(rets['A'].iloc[i] - exp_val) < 1e-10, (
            f"Row {i}: expected {exp_val:.6f}, got {rets['A'].iloc[i]:.6f}"
        )


def test_arithmetic_returns_no_missing():
    """If input prices have no NaN, output returns must have no NaN."""
    dates = pd.date_range('2020-01-01', periods=20, freq='B')
    prices = pd.DataFrame(
        np.random.rand(20, 5) * 50 + 100, index=dates, columns=list('ABCDE')
    )
    rets = compute_arithmetic_returns(prices)
    assert not rets.isnull().any().any(), "Unexpected NaN in returns"


def test_arithmetic_returns_constant_price():
    """Constant price produces zero return."""
    dates = pd.date_range('2020-01-01', periods=5, freq='B')
    prices = pd.DataFrame({'X': [50.0] * 5}, index=dates)
    rets = compute_arithmetic_returns(prices)
    assert (rets['X'] == 0.0).all()


# ---------------------------------------------------------------------------
# convert_tbill_to_daily
# ---------------------------------------------------------------------------

def test_tbill_daily_convention():
    """Daily rate = annualised yield / 252 (simple convention)."""
    annual_rate = 0.05
    dates = pd.date_range('2020-01-01', periods=5, freq='B')
    yield_series = pd.Series([annual_rate] * 5, index=dates)
    daily = convert_tbill_to_daily(yield_series)
    expected = annual_rate / 252
    assert abs(daily.iloc[0] - expected) < 1e-12


def test_tbill_daily_nonnegative():
    """Daily risk-free rate must be non-negative for non-negative annual yields."""
    dates = pd.date_range('2020-01-01', periods=10, freq='B')
    yields = pd.Series(np.abs(np.random.rand(10)) * 0.1, index=dates)
    daily = convert_tbill_to_daily(yields)
    assert (daily >= 0).all()


# ---------------------------------------------------------------------------
# build_balanced_panel
# ---------------------------------------------------------------------------

def _make_returns(n_assets: int = 5, n_dates: int = 600) -> tuple:
    """Return (returns_df, rf_series) with a common business-day index."""
    rng = np.random.default_rng(0)
    dates = pd.date_range('2018-12-12', periods=n_dates, freq='B')
    rets = pd.DataFrame(
        rng.normal(0, 0.01, (n_dates, n_assets)),
        index=dates,
        columns=[f'ETF{i}' for i in range(n_assets)],
    )
    rf = pd.Series(0.05 / 252, index=dates, name='rf')
    return rets, rf


def test_build_balanced_panel_shape():
    """Balanced panel must have the same number of columns as the input returns."""
    rets, rf = _make_returns(5, 600)
    balanced, _ = build_balanced_panel(rets, rf)
    assert balanced.shape[1] == rets.shape[1]


def test_build_balanced_panel_no_nans():
    """After building the balanced panel, no NaN values should remain in returns."""
    rets, rf = _make_returns(5, 600)
    balanced, _ = build_balanced_panel(rets, rf)
    assert not balanced.isnull().any().any()


def test_build_balanced_panel_rf_aligned():
    """The aligned risk-free series must share exactly the same index as returns."""
    rets, rf = _make_returns(5, 600)
    balanced, aligned_rf = build_balanced_panel(rets, rf)
    assert balanced.index.equals(aligned_rf.index), "Return and rf indices differ after alignment"


# ---------------------------------------------------------------------------
# get_audit_table
# ---------------------------------------------------------------------------

def _make_prices(n_assets: int = 5, n_dates: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='B')
    prices = pd.DataFrame(
        np.cumsum(rng.normal(0, 0.5, (n_dates, n_assets)), axis=0) + 100,
        index=dates,
        columns=[f'ETF{i}' for i in range(n_assets)],
    )
    return prices.clip(lower=1.0)


def test_audit_table_columns():
    """Audit table must contain ticker, date range, and count info per ticker."""
    prices = _make_prices(3, 200)
    audit = get_audit_table(prices)
    required_cols = {'ticker', 'first_date', 'last_date', 'n_obs'}
    assert required_cols.issubset(set(audit.columns)), (
        f"Missing columns: {required_cols - set(audit.columns)}"
    )


def test_audit_table_row_count():
    """Audit table must have one row per ticker."""
    prices = _make_prices(4, 200)
    audit = get_audit_table(prices)
    assert len(audit) == 4


def test_audit_table_n_obs_correct():
    """n_obs column must equal the number of non-NaN values per ticker."""
    prices = _make_prices(3, 50)
    # Introduce one NaN manually
    prices.iloc[5, 0] = np.nan
    audit = get_audit_table(prices)
    row = audit[audit['ticker'] == 'ETF0'].iloc[0]
    assert row['n_obs'] == 49
    assert row['missing_count'] == 1
