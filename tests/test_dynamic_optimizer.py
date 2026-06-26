"""Tests for src/dynamic_optimizer.py — ARMA-GARCH + copula dynamic optimization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.dynamic_optimizer import (
    fit_arma_garch_single,
    compute_pseudo_uniforms,
    fit_student_t_copula,
    simulate_student_t_copula_scenarios,
    generate_predictive_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns_array(n: int = 600, k: int = 4, seed: int = 7) -> np.ndarray:
    """Generate simple i.i.d. returns (n x k)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, (n, k))


def _make_df(n: int = 600, k: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2018-01-01', periods=n, freq='B')
    cols = [f'ETF{i}' for i in range(k)]
    data = rng.normal(0.0005, 0.01, (n, k))
    return pd.DataFrame(data, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# fit_arma_garch_single
# Returns (std_resids, cond_vol, one_step_mean, one_step_vol)
# ---------------------------------------------------------------------------

def test_arma_garch_returns_tuple():
    """fit_arma_garch_single must return a 4-element tuple."""
    rng = np.random.default_rng(1)
    series = rng.normal(0, 0.01, 600)
    result = fit_arma_garch_single(series)
    assert isinstance(result, tuple)
    assert len(result) == 4, "Expected (std_resids, cond_vol, one_step_mean, one_step_vol)"


def test_arma_garch_residuals_shape():
    """Standardized residuals (index 0) must have similar length as input."""
    rng = np.random.default_rng(2)
    series = rng.normal(0, 0.01, 550)
    std_resids = fit_arma_garch_single(series)[0]
    # GARCH may drop a small number of rows due to lag initialisation
    assert abs(len(std_resids) - len(series)) <= 5


def test_arma_garch_vol_positive():
    """Conditional volatilities (index 1) must all be positive."""
    rng = np.random.default_rng(3)
    series = rng.normal(0, 0.01, 600)
    cond_vols = fit_arma_garch_single(series)[1]
    assert (cond_vols > 0).all(), "Conditional volatilities must be strictly positive"


def test_arma_garch_std_resids_unit_scale():
    """Standardized residuals (index 0) should have sample std close to 1."""
    rng = np.random.default_rng(4)
    series = rng.normal(0, 0.01, 600)
    std_resids = fit_arma_garch_single(series)[0]
    std = np.std(std_resids)
    assert 0.5 < std < 2.5, f"Standardized residuals std = {std:.3f}, expected ~1"


def test_arma_garch_one_step_vol_positive():
    """One-step-ahead volatility forecast (index 3) must be positive."""
    rng = np.random.default_rng(99)
    series = rng.normal(0, 0.01, 600)
    one_step_vol = fit_arma_garch_single(series)[3]
    assert one_step_vol > 0, f"one_step_vol = {one_step_vol} must be > 0"


# ---------------------------------------------------------------------------
# compute_pseudo_uniforms
# ---------------------------------------------------------------------------

def test_pseudo_uniforms_in_unit_interval():
    """All pseudo-uniform values must be in (0, 1)."""
    rng = np.random.default_rng(5)
    resids = rng.standard_normal(500)
    u = compute_pseudo_uniforms(resids)
    assert (u > 0).all() and (u < 1).all(), "Pseudo-uniforms must be in (0, 1)"


def test_pseudo_uniforms_length():
    """Length of pseudo-uniforms must match input."""
    rng = np.random.default_rng(6)
    resids = rng.standard_normal(400)
    u = compute_pseudo_uniforms(resids)
    assert len(u) == 400


def test_pseudo_uniforms_uniform_distribution():
    """Pseudo-uniforms from Normal(0,1) should be roughly uniform."""
    rng = np.random.default_rng(7)
    resids = rng.standard_normal(1000)
    u = compute_pseudo_uniforms(resids)
    # Mean of a Uniform(0,1) is 0.5 — sample mean should be close
    assert abs(u.mean() - 0.5) < 0.05


# ---------------------------------------------------------------------------
# fit_student_t_copula
# ---------------------------------------------------------------------------

def test_student_t_copula_returns_corr_and_nu():
    """fit_student_t_copula must return (corr_matrix, nu) tuple."""
    rng = np.random.default_rng(8)
    U = rng.uniform(0.01, 0.99, (300, 4))
    result = fit_student_t_copula(U)
    assert isinstance(result, tuple)
    corr, nu = result
    assert isinstance(nu, float | np.floating)


def test_student_t_copula_corr_matrix_shape():
    """Correlation matrix must be k x k."""
    rng = np.random.default_rng(9)
    k = 5
    U = rng.uniform(0.01, 0.99, (300, k))
    corr, nu = fit_student_t_copula(U)
    assert corr.shape == (k, k)


def test_student_t_copula_corr_diagonal_ones():
    """Diagonal of correlation matrix must be 1.0."""
    rng = np.random.default_rng(10)
    U = rng.uniform(0.01, 0.99, (300, 4))
    corr, nu = fit_student_t_copula(U)
    np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-6)


def test_student_t_copula_nu_positive():
    """Degrees of freedom nu must be strictly positive."""
    rng = np.random.default_rng(11)
    U = rng.uniform(0.01, 0.99, (300, 4))
    _, nu = fit_student_t_copula(U)
    assert nu > 0, f"nu = {nu} must be > 0"


# ---------------------------------------------------------------------------
# simulate_student_t_copula_scenarios
# ---------------------------------------------------------------------------

def test_simulate_copula_shape():
    """Simulated matrix must have shape (n_scenarios, k)."""
    k = 4
    corr = np.eye(k)
    nu = 5.0
    U_sim = simulate_student_t_copula_scenarios(corr, nu, n_scenarios=200)
    assert U_sim.shape == (200, k)


def test_simulate_copula_in_unit_interval():
    """All simulated pseudo-uniform values must be in (0, 1)."""
    k = 3
    corr = np.eye(k)
    nu = 4.0
    U_sim = simulate_student_t_copula_scenarios(corr, nu, n_scenarios=500)
    assert np.all((U_sim > 0) & (U_sim < 1)), "Simulated copula values must be in (0, 1)"


# ---------------------------------------------------------------------------
# generate_predictive_scenarios
# ---------------------------------------------------------------------------

def test_predictive_scenarios_shape():
    """generate_predictive_scenarios must return (scenarios, mu, sigma) tuple."""
    window_data = _make_returns_array(n=520, k=4)
    result = generate_predictive_scenarios(window_data, n_scenarios=500)
    assert isinstance(result, tuple), "Expected a tuple return"
    scenarios = result[0]
    assert scenarios.shape[1] == 4
    assert scenarios.shape[0] == 500


def test_predictive_scenarios_finite():
    """All scenario returns must be finite (no NaN or Inf)."""
    window_data = _make_returns_array(n=520, k=4)
    scenarios = generate_predictive_scenarios(window_data, n_scenarios=500)[0]
    assert np.isfinite(scenarios).all(), "Predictive scenarios contain NaN or Inf"


def test_predictive_scenarios_small_magnitudes():
    """Daily return scenarios should have realistic magnitudes (|ret| < 50%)."""
    window_data = _make_returns_array(n=520, k=4)
    scenarios = generate_predictive_scenarios(window_data, n_scenarios=500)[0]
    assert np.abs(scenarios).max() < 0.5, "Some scenario returns are implausibly large"


def test_predictive_scenarios_forecast_shapes():
    """One-step mean and vol forecasts must match number of assets."""
    window_data = _make_returns_array(n=520, k=4)
    _, mu_f, sig_f = generate_predictive_scenarios(window_data, n_scenarios=300)
    assert len(mu_f) == 4
    assert len(sig_f) == 4
    assert np.all(sig_f > 0), "All volatility forecasts must be positive"
