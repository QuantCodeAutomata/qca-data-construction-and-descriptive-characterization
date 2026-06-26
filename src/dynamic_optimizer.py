import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
from scipy.optimize import minimize_scalar
from arch import arch_model

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

WINDOW = 504
N_SCENARIOS = 5000
ARMA_ORDER = (1, 1)
GARCH_ORDER = (1, 1)
C_TO = 1.0 / (30 * 252)
ALPHA_95 = 0.95
ALPHA_99 = 0.99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_psd(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Nudge a symmetric matrix to be positive semi-definite."""
    ev = np.linalg.eigvalsh(M)
    if ev.min() < eps:
        M = M + (eps - ev.min()) * np.eye(M.shape[0])
    return M


# ---------------------------------------------------------------------------
# ARMA-GARCH marginal fitting
# ---------------------------------------------------------------------------

def fit_arma_garch_single(returns: np.ndarray) -> tuple:
    """Fit an ARMA(1,1)-GARCH(1,1) model to a single return series.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of daily arithmetic returns for one asset.

    Returns
    -------
    tuple
        ``(std_resids, cond_vol, one_step_mean, one_step_vol)``

        * **std_resids** – standardised residuals (length T).
        * **cond_vol** – conditional volatilities on the *original* scale.
        * **one_step_mean** – one-step-ahead mean forecast (original scale).
        * **one_step_vol** – one-step-ahead volatility forecast (original scale).
    """
    try:
        am = arch_model(returns * 100, mean='ARX', lags=1, vol='GARCH', p=1, q=1)
        result = am.fit(disp='off')
        # arch may return numpy arrays or pandas Series depending on input type;
        # use numpy operations to handle both uniformly
        resid_arr = np.asarray(result.resid, dtype=float)
        condvol_arr = np.asarray(result.conditional_volatility, dtype=float)
        # arch produces NaN at the start due to lag initialisation; keep only finite rows
        valid = np.isfinite(resid_arr) & np.isfinite(condvol_arr) & (condvol_arr > 0)
        std_resids = resid_arr[valid] / condvol_arr[valid]
        cond_vol = condvol_arr[valid]
        forecast = result.forecast(horizon=1)
        one_step_mean = float(forecast.mean.iloc[-1, 0]) / 100.0
        one_step_vol = float(np.sqrt(forecast.variance.iloc[-1, 0])) / 100.0
        return std_resids, cond_vol, one_step_mean, one_step_vol
    except Exception as exc:
        logger.debug("ARMA-GARCH fit failed (%s); using mean/std fallback.", exc)
        mu = float(returns.mean())
        sig = float(returns.std())
        return (returns - mu), np.ones(len(returns)) * sig, mu, sig


# ---------------------------------------------------------------------------
# Empirical PIT
# ---------------------------------------------------------------------------

def compute_pseudo_uniforms(std_residuals: np.ndarray) -> np.ndarray:
    """Apply empirical probability-integral transform (Hazen plotting positions).

    Parameters
    ----------
    std_residuals : np.ndarray
        1-D array of standardised residuals.

    Returns
    -------
    np.ndarray
        Pseudo-uniform values in (0, 1).
    """
    ranks = stats.rankdata(std_residuals)
    return ranks / (len(std_residuals) + 1)


# ---------------------------------------------------------------------------
# Student-t copula estimation
# ---------------------------------------------------------------------------

def fit_student_t_copula(pseudo_uniforms: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the linear correlation matrix and degrees-of-freedom for a
    Student-t copula via a Gaussian copula / univariate t-fit approach.

    Parameters
    ----------
    pseudo_uniforms : np.ndarray
        Shape ``(T, n_assets)`` matrix of empirical PIT values.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(R, dof)`` – correlation matrix and degrees-of-freedom.
        ``dof`` is clipped to ``[2.5, 30]``.
    """
    Z = stats.norm.ppf(np.clip(pseudo_uniforms, 1e-6, 1.0 - 1e-6))
    # Drop any rows that still contain non-finite values
    finite_rows = np.isfinite(Z).all(axis=1)
    Z = Z[finite_rows]
    R = np.corrcoef(Z.T)
    dfs = []
    for i in range(Z.shape[1]):
        col = Z[:, i]
        col = col[np.isfinite(col)]
        if len(col) > 10:
            try:
                fitted = stats.t.fit(col)
                dfs.append(fitted[0])
            except Exception:
                pass
    dof = float(np.clip(np.median(dfs) if dfs else 5.0, 2.5, 30.0))
    return R, dof


# ---------------------------------------------------------------------------
# Scenario simulation
# ---------------------------------------------------------------------------

def simulate_student_t_copula_scenarios(
    R: np.ndarray,
    dof: float,
    n_scenarios: int,
) -> np.ndarray:
    """Draw Monte Carlo scenarios from a Student-t copula.

    Parameters
    ----------
    R : np.ndarray
        Correlation matrix, shape ``(n, n)``.
    dof : float
        Degrees of freedom for the Student-t distribution.
    n_scenarios : int
        Number of scenarios to simulate.

    Returns
    -------
    np.ndarray
        Pseudo-uniform copula samples, shape ``(n_scenarios, n_assets)``.
    """
    n = R.shape[0]
    R_pd = _ensure_psd(R)
    Z = np.random.multivariate_normal(np.zeros(n), R_pd, size=n_scenarios)
    chi2 = np.random.chisquare(dof, size=n_scenarios)
    t_samples = Z * np.sqrt(dof / chi2[:, None])
    u_samples = stats.t.cdf(t_samples, df=dof)
    return u_samples  # (n_scenarios, n_assets)


def generate_predictive_scenarios(
    returns_window: np.ndarray,
    n_scenarios: int = N_SCENARIOS,
) -> tuple:
    """Generate one-step-ahead return scenarios via ARMA-GARCH marginals and
    a Student-t copula.

    Parameters
    ----------
    returns_window : np.ndarray
        Shape ``(T, n_assets)`` array of in-sample returns.
    n_scenarios : int
        Number of Monte Carlo scenarios.

    Returns
    -------
    tuple
        ``(scenario_returns, mu_forecasts, sigma_forecasts)``

        * **scenario_returns** – shape ``(n_scenarios, n_assets)``.
        * **mu_forecasts** – one-step mean forecasts, shape ``(n_assets,)``.
        * **sigma_forecasts** – one-step vol forecasts, shape ``(n_assets,)``.
    """
    n_assets = returns_window.shape[1]
    std_resids_list: list[np.ndarray] = []
    mu_1step: list[float] = []
    sig_1step: list[float] = []

    for i in range(n_assets):
        sr, _cv, mu, sig = fit_arma_garch_single(returns_window[:, i])
        std_resids_list.append(sr)
        mu_1step.append(mu)
        sig_1step.append(sig)

    # Build PIT matrix: align to the shortest residual series
    min_len = min(len(sr) for sr in std_resids_list)
    pit_matrix = np.column_stack([
        compute_pseudo_uniforms(sr[-min_len:]) for sr in std_resids_list
    ])  # (min_len, n_assets)

    R, dof = fit_student_t_copula(pit_matrix)
    u_sim = simulate_student_t_copula_scenarios(R, dof, n_scenarios)
    # (n_scenarios, n_assets)

    scenario_returns = np.zeros((n_scenarios, n_assets))
    for i in range(n_assets):
        # Empirical quantile inverse: map uniform copula samples back through
        # the empirical CDF of standardised residuals
        z_sim = np.quantile(std_resids_list[i], u_sim[:, i])
        scenario_returns[:, i] = mu_1step[i] + sig_1step[i] * z_sim

    return (
        scenario_returns,
        np.array(mu_1step),
        np.array(sig_1step),
    )


# ---------------------------------------------------------------------------
# Portfolio optimisation solvers
# ---------------------------------------------------------------------------

def _solve_mvp_scenarios(
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    lb: float,
    ub: float,
    enforce_to: bool = False,
) -> Optional[np.ndarray]:
    """Minimum Variance Portfolio from a precomputed covariance matrix.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix, shape ``(n, n)``.
    w_prev : np.ndarray
        Previous portfolio weights (used only when *enforce_to* is True).
    lb, ub : float
        Per-asset weight bounds.
    enforce_to : bool
        If True, add a one-way turnover cap of ``2 * C_TO``.

    Returns
    -------
    Optional[np.ndarray]
        Optimal weights, or *None* if the solver fails.
    """
    n = Sigma.shape[0]
    Sigma_reg = _ensure_psd(Sigma)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub]
    if enforce_to:
        d = cp.Variable(n)
        constraints += [d >= w - w_prev, d >= -(w - w_prev), cp.sum(d) <= 2.0 * C_TO]
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma_reg)))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ('optimal', 'optimal_inaccurate') and w.value is not None:
            return np.array(w.value)
    except Exception as exc:
        logger.debug("MVP solve failed: %s", exc)
    return None


def _solve_min_cvar_scenarios(
    scenario_returns: np.ndarray,
    alpha: float,
    w_prev: np.ndarray,
    lb: float,
    ub: float,
    enforce_to: bool = False,
) -> Optional[np.ndarray]:
    """Minimum CVaR portfolio via Rockafellar-Uryasev (2000) LP formulation.

    Parameters
    ----------
    scenario_returns : np.ndarray
        Shape ``(n_scen, n_assets)`` scenario return matrix.
    alpha : float
        CVaR confidence level (e.g. 0.95).
    w_prev, lb, ub, enforce_to
        See :func:`_solve_mvp_scenarios`.

    Returns
    -------
    Optional[np.ndarray]
        Optimal weights, or *None* if the solver fails.
    """
    n_scen, n = scenario_returns.shape
    w = cp.Variable(n)
    zeta = cp.Variable()          # VaR threshold
    xi = cp.Variable(n_scen, nonneg=True)  # excess losses above VaR

    portfolio_returns = scenario_returns @ w
    constraints = [
        cp.sum(w) == 1,
        w >= lb,
        w <= ub,
        xi >= -portfolio_returns - zeta,
    ]
    if enforce_to:
        d = cp.Variable(n)
        constraints += [d >= w - w_prev, d >= -(w - w_prev), cp.sum(d) <= 2.0 * C_TO]

    cvar = zeta + cp.sum(xi) / (n_scen * (1.0 - alpha))
    objective = cp.Minimize(cvar)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ('optimal', 'optimal_inaccurate') and w.value is not None:
            return np.array(w.value)
    except Exception as exc:
        logger.debug("Min-CVaR (alpha=%.2f) solve failed: %s", alpha, exc)
    return None


def _solve_tvp_scenarios(
    mu_pred: np.ndarray,
    Sigma_pred: np.ndarray,
    rf: float,
    w_prev: np.ndarray,
    lb: float,
    ub: float,
    enforce_to: bool = False,
) -> Optional[np.ndarray]:
    """Tangency (maximum Sharpe) portfolio via efficient-frontier grid search.

    Iterates over a grid of target expected returns, solves the minimum-variance
    portfolio at each target, and returns the weights with the highest Sharpe
    ratio.

    Parameters
    ----------
    mu_pred : np.ndarray
        One-step-ahead mean return forecasts, shape ``(n,)``.
    Sigma_pred : np.ndarray
        One-step-ahead covariance forecast, shape ``(n, n)``.
    rf : float
        Daily risk-free rate.
    w_prev, lb, ub, enforce_to
        See :func:`_solve_mvp_scenarios`.

    Returns
    -------
    Optional[np.ndarray]
        Optimal weights, or *None* if no feasible point is found.
    """
    n = len(mu_pred)
    Sigma_reg = _ensure_psd((Sigma_pred + Sigma_pred.T) / 2.0)

    mu_lo = float(np.min(mu_pred))
    mu_hi = float(np.max(mu_pred))
    if mu_hi <= mu_lo:
        mu_hi = mu_lo + 1e-7

    best_w: Optional[np.ndarray] = None
    best_sharpe = -np.inf

    for mu_target in np.linspace(mu_lo, mu_hi, 12):
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= lb,
            w <= ub,
            w @ mu_pred >= mu_target,
        ]
        if enforce_to:
            d = cp.Variable(n)
            constraints += [d >= w - w_prev, d >= -(w - w_prev), cp.sum(d) <= 2.0 * C_TO]
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma_reg)))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status not in ('optimal', 'optimal_inaccurate') or w.value is None:
                continue
            pret = float(w.value @ mu_pred)
            pvar = float(w.value @ Sigma_pred @ w.value)
            pstd = np.sqrt(max(pvar, 0.0))
            if pstd > 1e-10:
                sharpe = (pret - rf) / pstd
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = np.array(w.value)
        except Exception:
            continue

    return best_w


def _solve_max_cvar_ratio_scenarios(
    scenario_returns: np.ndarray,
    mu_pred: np.ndarray,
    rf: float,
    alpha: float,
    w_prev: np.ndarray,
    lb: float,
    ub: float,
    enforce_to: bool = False,
) -> Optional[np.ndarray]:
    """Maximum return-to-CVaR ratio via CVaR-frontier grid search.

    For each target expected return on a grid, the minimum-CVaR portfolio is
    found; the portfolio with the largest ``(mean - rf) / CVaR`` is returned.

    Parameters
    ----------
    scenario_returns : np.ndarray
        Shape ``(n_scen, n_assets)`` scenario return matrix.
    mu_pred : np.ndarray
        One-step-ahead mean return forecasts, shape ``(n,)``.
    rf : float
        Daily risk-free rate.
    alpha : float
        CVaR confidence level.
    w_prev, lb, ub, enforce_to
        See :func:`_solve_mvp_scenarios`.

    Returns
    -------
    Optional[np.ndarray]
        Optimal weights, or *None* if no feasible point is found.
    """
    n_scen, n = scenario_returns.shape

    mu_lo = float(np.min(mu_pred))
    mu_hi = float(np.max(mu_pred))
    if mu_hi <= mu_lo:
        mu_hi = mu_lo + 1e-7

    best_w: Optional[np.ndarray] = None
    best_ratio = -np.inf

    for mu_target in np.linspace(mu_lo, mu_hi, 12):
        w = cp.Variable(n)
        zeta = cp.Variable()
        xi = cp.Variable(n_scen, nonneg=True)
        portfolio_returns = scenario_returns @ w
        constraints = [
            cp.sum(w) == 1,
            w >= lb,
            w <= ub,
            xi >= -portfolio_returns - zeta,
            w @ mu_pred >= mu_target,
        ]
        if enforce_to:
            d = cp.Variable(n)
            constraints += [d >= w - w_prev, d >= -(w - w_prev), cp.sum(d) <= 2.0 * C_TO]
        cvar_obj = zeta + cp.sum(xi) / (n_scen * (1.0 - alpha))
        objective = cp.Minimize(cvar_obj)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status not in ('optimal', 'optimal_inaccurate') or w.value is None:
                continue
            pret = float(w.value @ mu_pred)
            losses = -(scenario_returns @ w.value)
            var_val = np.quantile(losses, alpha)
            tail = losses[losses >= var_val]
            cvar_val = float(tail.mean()) if len(tail) > 0 else float(var_val)
            if cvar_val > 1e-10:
                ratio = (pret - rf) / cvar_val
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_w = np.array(w.value)
        except Exception:
            continue

    return best_w


# ---------------------------------------------------------------------------
# Rolling dynamic backtest
# ---------------------------------------------------------------------------

def run_dynamic_backtest(
    returns_df: pd.DataFrame,
    rf_series: pd.Series,
    variant: str = 'A',
    n_scenarios: int = N_SCENARIOS,
    rebalance_every: int = 21,
) -> tuple:
    """Execute a rolling dynamic portfolio backtest.

    At each step the ARMA-GARCH / Student-t copula pipeline generates
    ``n_scenarios`` one-step-ahead return scenarios that feed four optimisation
    objectives under two weight-bound regimes, yielding 12 strategies plus a
    buy-and-hold equal-weight benchmark (BHP).

    Assumptions logged at start
    ---------------------------
    * ARMA(1,1)-GARCH(1,1) marginal model order (paper unspecified).
    * Empirical PIT for marginal-to-uniform transformation.
    * Student-t copula; degrees of freedom = median of univariate t-fits.
    * 5 000 Monte Carlo scenarios per period (configurable).
    * Variant A = no dynamic turnover cap (primary); Variant B = cap applied.
    * Short-selling bounds: [-0.30, 1.30] per asset.
    * BHP = equally-weighted buy-and-hold (weights drift naturally).
    * Realised return for period t is returns_df.iloc[t] (same-day).
    * Window length = 504 trading days.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Balanced panel of daily arithmetic returns, shape ``(T, n_assets)``.
    rf_series : pd.Series
        Daily risk-free rate, indexed by date.
    variant : {'A', 'B'}
        'A' – no turnover cap; 'B' – enforce turnover cap ``C_TO``.
    n_scenarios : int
        Number of Monte Carlo scenarios per rebalancing date.
    rebalance_every : int
        Number of trading days between active rebalances for optimised
        strategies (default 21, i.e. approximately monthly).  BHP always
        drifts daily regardless of this setting.

    Returns
    -------
    tuple
        ``(strategy_returns_dict, strategy_weights_dict, strategy_turnover_dict)``
        Each value is a :class:`pd.Series` / :class:`pd.DataFrame` keyed by
        strategy name.
    """
    logger.info("ASSUMPTION: ARMA(1,1)-GARCH(1,1) marginal model order (paper unspecified)")
    logger.info("ASSUMPTION: Empirical PIT for marginal-to-uniform transformation")
    logger.info("ASSUMPTION: Student-t copula; dof = median of univariate t-fits, clipped [2.5, 30]")
    logger.info("ASSUMPTION: %d Monte Carlo scenarios per period", n_scenarios)
    logger.info("ASSUMPTION: Variant %s => enforce_to=%s", variant, variant == 'B')
    logger.info("ASSUMPTION: Short-selling bounds [-0.30, 1.30] per asset")
    logger.info("ASSUMPTION: BHP = equally-weighted buy-and-hold (weights drift naturally)")
    logger.info("ASSUMPTION: Rolling estimation window = %d trading days", WINDOW)
    logger.info("ASSUMPTION: Rebalance every %d trading days", rebalance_every)

    enforce_to = (variant == 'B')
    n_assets = returns_df.shape[1]
    dates = returns_df.index

    # Strategy specifications: (name, solver_type, lb, ub, alpha)
    strategy_configs: list[tuple] = [
        ('LO_MVP',   'mvp',  0.0,  1.0,  ALPHA_95),
        ('LO_C95',   'cvar', 0.0,  1.0,  ALPHA_95),
        ('LO_C99',   'cvar', 0.0,  1.0,  ALPHA_99),
        ('LO_TVP',   'tvp',  0.0,  1.0,  ALPHA_95),
        ('LO_MCR95', 'mcr',  0.0,  1.0,  ALPHA_95),
        ('LO_MCR99', 'mcr',  0.0,  1.0,  ALPHA_99),
        ('SS_MVP',   'mvp',  -0.3, 1.3,  ALPHA_95),
        ('SS_C95',   'cvar', -0.3, 1.3,  ALPHA_95),
        ('SS_C99',   'cvar', -0.3, 1.3,  ALPHA_99),
        ('SS_TVP',   'tvp',  -0.3, 1.3,  ALPHA_95),
        ('SS_MCR95', 'mcr',  -0.3, 1.3,  ALPHA_95),
        ('SS_MCR99', 'mcr',  -0.3, 1.3,  ALPHA_99),
    ]
    all_names = [cfg[0] for cfg in strategy_configs] + ['BHP']

    # Accumulate results as dicts of {date: value}
    ret_store: dict[str, dict] = {name: {} for name in all_names}
    wgt_store: dict[str, dict] = {name: {} for name in all_names}
    to_store:  dict[str, dict] = {name: {} for name in all_names}

    # Initialise current weights to equal-weight
    w_current: dict[str, np.ndarray] = {
        name: np.ones(n_assets) / n_assets for name in all_names
    }

    n_steps = len(returns_df) - WINDOW - 1

    # Cache scenarios/moments across non-rebalance days
    cached_scenario_returns: np.ndarray | None = None
    cached_mu_pred: np.ndarray | None = None
    cached_Sigma_pred: np.ndarray | None = None

    for step, t in enumerate(range(WINDOW, len(returns_df) - 1)):
        date = dates[t]
        realized = returns_df.iloc[t].values                # (n_assets,)
        rf_t = float(rf_series[date]) if date in rf_series.index else float(rf_series.mean())

        is_rebalance = (step % rebalance_every == 0)

        # ----------------------------------------------------------------- #
        # Generate predictive scenarios only on rebalance days               #
        # ----------------------------------------------------------------- #
        if is_rebalance:
            window_data = returns_df.iloc[t - WINDOW:t].values  # (WINDOW, n_assets)
            try:
                scenario_returns, mu_pred, _sig_pred = generate_predictive_scenarios(
                    window_data, n_scenarios
                )
            except Exception as exc:
                logger.warning(
                    "Scenario generation failed at %s (%s); falling back to historical window.",
                    date, exc,
                )
                scenario_returns = window_data.copy()
                mu_pred = window_data.mean(axis=0)

            cached_scenario_returns = scenario_returns
            cached_mu_pred = scenario_returns.mean(axis=0)
            cached_Sigma_pred = np.cov(scenario_returns.T)

        # Use cached values on non-rebalance days
        scenario_returns = cached_scenario_returns
        mu_pred_arr = cached_mu_pred
        Sigma_pred = cached_Sigma_pred

        # ----------------------------------------------------------------- #
        # BHP: equal-weight buy-and-hold, weights drift with market          #
        # ----------------------------------------------------------------- #
        bhp_w = w_current['BHP']
        ret_store['BHP'][date] = float(bhp_w @ realized)
        to_store['BHP'][date] = 0.0
        wgt_store['BHP'][date] = bhp_w.copy()
        # Drift weights forward
        new_bhp = bhp_w * (1.0 + realized)
        bhp_sum = new_bhp.sum()
        w_current['BHP'] = new_bhp / bhp_sum if bhp_sum > 1e-10 else bhp_w.copy()

        # ----------------------------------------------------------------- #
        # Optimised strategies: only rebalance on designated days            #
        # ----------------------------------------------------------------- #
        for name, solver_type, lb, ub, alpha in strategy_configs:
            w_prev = w_current[name]

            if is_rebalance:
                if solver_type == 'mvp':
                    w_new = _solve_mvp_scenarios(Sigma_pred, w_prev, lb, ub, enforce_to)
                elif solver_type == 'cvar':
                    w_new = _solve_min_cvar_scenarios(
                        scenario_returns, alpha, w_prev, lb, ub, enforce_to
                    )
                elif solver_type == 'tvp':
                    w_new = _solve_tvp_scenarios(
                        mu_pred_arr, Sigma_pred, rf_t, w_prev, lb, ub, enforce_to
                    )
                else:  # 'mcr'
                    w_new = _solve_max_cvar_ratio_scenarios(
                        scenario_returns, mu_pred_arr, rf_t, alpha, w_prev, lb, ub, enforce_to
                    )

                # Fall back to previous weights on solver failure
                if w_new is None:
                    w_new = w_prev.copy()
                else:
                    # Clip and renormalise
                    w_new = np.clip(w_new, lb, ub)
                    w_sum = w_new.sum()
                    if abs(w_sum) > 1e-10:
                        w_new /= w_sum
            else:
                w_new = w_prev.copy()

            turnover = float(np.sum(np.abs(w_new - w_prev)))
            ret_store[name][date] = float(w_new @ realized)
            to_store[name][date] = turnover
            wgt_store[name][date] = w_new.copy()

            # Drift weights forward
            drifted = w_new * (1.0 + realized)
            d_sum = drifted.sum()
            w_current[name] = drifted / d_sum if abs(d_sum) > 1e-10 else w_new.copy()

        if (step + 1) % 50 == 0:
            logger.info(
                "Dynamic backtest [variant=%s] step %d/%d  date=%s  rebalance=%s",
                variant, step + 1, n_steps, date.date(), is_rebalance,
            )

    # Convert to pd.Series / pd.DataFrame
    strategy_returns_dict: dict[str, pd.Series] = {
        name: pd.Series(ret_store[name], name=name) for name in all_names
    }
    strategy_weights_dict: dict[str, pd.DataFrame] = {
        name: pd.DataFrame(wgt_store[name], index=returns_df.columns).T
        for name in all_names
    }
    strategy_turnover_dict: dict[str, pd.Series] = {
        name: pd.Series(to_store[name], name=name) for name in all_names
    }

    return strategy_returns_dict, strategy_weights_dict, strategy_turnover_dict
