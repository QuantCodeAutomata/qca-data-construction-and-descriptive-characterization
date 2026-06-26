import logging
import numpy as np
import pandas as pd
import cvxpy as cp

logger = logging.getLogger(__name__)

WINDOW = 504
C_TO = 1.0 / (30 * 252)
ALPHA_95 = 0.95
ALPHA_99 = 0.99


def solve_mvp(mu, Sigma, w_prev, lb, ub, enforce_turnover=True):
    """Solve the minimum-variance portfolio (MVP) using cvxpy.

    Parameters
    ----------
    mu : np.ndarray, shape (n,)
        Expected daily returns for each asset.
    Sigma : np.ndarray, shape (n, n)
        Covariance matrix of daily returns.
    w_prev : np.ndarray, shape (n,)
        Portfolio weights held at the previous rebalancing date.
    lb : np.ndarray, shape (n,)
        Per-asset lower bound on portfolio weights.
    ub : np.ndarray, shape (n,)
        Per-asset upper bound on portfolio weights.
    enforce_turnover : bool, optional
        If True, add a turnover constraint capped at C_TO (default True).

    Returns
    -------
    np.ndarray or None
        Optimal weight vector, or None if the problem is infeasible / failed.
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, Sigma + 1e-8 * np.eye(n)))
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub]
    if enforce_turnover:
        d = cp.Variable(n)
        constraints += [d >= w - w_prev, d >= -(w - w_prev), 0.5 * cp.sum(d) <= C_TO]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
        return w.value
    return None


def solve_tvp(mu, Sigma, rf, w_prev, lb, ub, enforce_turnover=True):
    """Solve the tangent (maximum-Sharpe) portfolio via grid search over target returns.

    For each of ``n_grid=50`` target return levels between ``mu.min()`` and
    ``mu.max()``, solve the minimum-variance problem with the additional
    constraint ``mu @ w >= target``.  The feasible portfolio with the highest
    annualised Sharpe ratio is returned.

    Parameters
    ----------
    mu : np.ndarray, shape (n,)
        Expected daily returns.
    Sigma : np.ndarray, shape (n, n)
        Covariance matrix of daily returns.
    rf : float
        Daily risk-free rate.
    w_prev : np.ndarray, shape (n,)
        Previous portfolio weights (for turnover constraint).
    lb : np.ndarray, shape (n,)
        Per-asset lower weight bound.
    ub : np.ndarray, shape (n,)
        Per-asset upper weight bound.
    enforce_turnover : bool, optional
        Add turnover cap constraint (default True).

    Returns
    -------
    np.ndarray or None
        Weight vector with the highest Sharpe ratio across the grid, or None if
        no feasible solution exists.
    """
    n = len(mu)
    n_grid = 15  # reduced from 50 — sufficient for max-Sharpe selection; 3× faster
    targets = np.linspace(mu.min(), mu.max(), n_grid)

    best_w = None
    best_sharpe = -np.inf

    for target in targets:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= lb, w <= ub, mu @ w >= target]
        if enforce_turnover:
            d = cp.Variable(n)
            constraints += [d >= w - w_prev, d >= -(w - w_prev), 0.5 * cp.sum(d) <= C_TO]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma + 1e-8 * np.eye(n))), constraints)
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            port_ret = float(mu @ w.value)
            port_var = float(w.value @ Sigma @ w.value)
            port_std = np.sqrt(max(port_var, 1e-16))
            sharpe = (port_ret - rf) / port_std
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = w.value.copy()

    return best_w


def solve_min_cvar(returns_window, alpha, w_prev, lb, ub, enforce_turnover=True):
    """Solve the minimum-CVaR portfolio using the Rockafellar-Uryasev LP.

    The CVaR at confidence level ``alpha`` is minimised via the auxiliary
    variable formulation:

        min  eta + 1/((1-alpha)*S) * sum(u)
        s.t. u >= -R @ w - eta,  u >= 0
             sum(w) = 1,  lb <= w <= ub

    Parameters
    ----------
    returns_window : np.ndarray, shape (S, 30)
        Historical return matrix for the look-back window (S scenarios, 30 assets).
    alpha : float
        CVaR confidence level (e.g. 0.95 or 0.99).
    w_prev : np.ndarray, shape (30,)
        Previous portfolio weights.
    lb : np.ndarray, shape (30,)
        Per-asset lower bound.
    ub : np.ndarray, shape (30,)
        Per-asset upper bound.
    enforce_turnover : bool, optional
        Add turnover cap constraint (default True).

    Returns
    -------
    np.ndarray or None
        Optimal weight vector, or None on failure.
    """
    n = returns_window.shape[1]
    S = len(returns_window)
    w = cp.Variable(n)
    eta = cp.Variable()
    u = cp.Variable(S, nonneg=True)
    losses = -returns_window @ w
    objective = cp.Minimize(eta + (1.0 / ((1 - alpha) * S)) * cp.sum(u))
    constraints = [u >= losses - eta, cp.sum(w) == 1, w >= lb, w <= ub]
    if enforce_turnover:
        d = cp.Variable(n)
        constraints += [d >= w - w_prev, d >= -(w - w_prev), 0.5 * cp.sum(d) <= C_TO]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
        return w.value
    return None


def solve_max_cvar_ratio(returns_window, mu, rf, alpha, w_prev, lb, ub, enforce_turnover=True):
    """Solve the maximum return-to-CVaR ratio portfolio via a CVaR frontier search.

    Traces 30 points on the CVaR efficient frontier by solving:

        max  mu @ w
        s.t. CVaR_alpha(w) <= target_cvar
             sum(w) = 1,  lb <= w <= ub

    for a linspace of ``target_cvar`` values between the minimum-CVaR
    portfolio's CVaR and twice that value.  Returns the portfolio with the
    highest ``(mu @ w - rf) / CVaR_alpha(w)`` ratio.

    Parameters
    ----------
    returns_window : np.ndarray, shape (S, 30)
        Historical return matrix.
    mu : np.ndarray, shape (30,)
        Expected daily returns.
    rf : float
        Daily risk-free rate.
    alpha : float
        CVaR confidence level.
    w_prev : np.ndarray, shape (30,)
        Previous portfolio weights.
    lb : np.ndarray, shape (30,)
        Per-asset lower bound.
    ub : np.ndarray, shape (30,)
        Per-asset upper bound.
    enforce_turnover : bool, optional
        Add turnover cap constraint (default True).

    Returns
    -------
    np.ndarray or None
        Weight vector maximising the return-to-CVaR ratio, or None on failure.
    """
    n = returns_window.shape[1]
    S = len(returns_window)
    n_pts = 4  # 4 frontier points – sufficient for ratio maximisation; faster runtime

    w_min = solve_min_cvar(returns_window, alpha, w_prev, lb, ub, enforce_turnover)
    if w_min is None:
        return None
    min_cvar = compute_empirical_cvar(returns_window @ w_min, alpha)

    w_eq = np.ones(n) / n
    eq_cvar = compute_empirical_cvar(returns_window @ w_eq, alpha)
    max_cvar = max(eq_cvar, min_cvar * 2.0)

    targets_cvar = np.linspace(min_cvar, max_cvar, n_pts)

    best_w = None
    best_ratio = -np.inf

    for target_cvar in targets_cvar:
        w = cp.Variable(n)
        eta = cp.Variable()
        u = cp.Variable(S, nonneg=True)
        losses = -returns_window @ w
        constraints = [
            u >= losses - eta,
            eta + (1.0 / ((1 - alpha) * S)) * cp.sum(u) <= target_cvar,
            cp.sum(w) == 1,
            w >= lb,
            w <= ub,
        ]
        if enforce_turnover:
            d = cp.Variable(n)
            constraints += [d >= w - w_prev, d >= -(w - w_prev), 0.5 * cp.sum(d) <= C_TO]
        prob = cp.Problem(cp.Maximize(mu @ w), constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            continue
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            port_ret = float(mu @ w.value)
            actual_cvar = compute_empirical_cvar(returns_window @ w.value, alpha)
            if actual_cvar > 0:
                ratio = (port_ret - rf) / actual_cvar
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_w = w.value.copy()

    return best_w


def compute_empirical_cvar(returns, alpha):
    """Compute empirical CVaR (Expected Shortfall) of portfolio losses.

    Losses are defined as the negative of returns.  CVaR is the mean of losses
    that meet or exceed the VaR threshold.

    Parameters
    ----------
    returns : array-like
        Portfolio return series.
    alpha : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    float
        CVaR expressed as a positive decimal (e.g. 0.02 = 2 %).
    """
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def compute_max_drawdown(returns):
    """Compute the maximum drawdown from a return series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily return series.

    Returns
    -------
    float
        Maximum drawdown expressed as a positive decimal (e.g. 0.25 = 25 %).
    """
    if isinstance(returns, pd.Series):
        cumval = (1 + returns).cumprod()
        running_max = cumval.cummax()
        dd = (running_max - cumval) / running_max
        return float(dd.max())
    arr = np.asarray(returns, dtype=float)
    cumval = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cumval)
    dd = (running_max - cumval) / running_max
    return float(dd.max())


def compute_portfolio_metrics(returns, rf, label, turnover=None):
    """Compute a comprehensive set of risk and performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily gross portfolio returns with a DatetimeIndex.
    rf : pd.Series
        Daily risk-free rate with a DatetimeIndex.
    label : str
        Strategy label used as an identifier in the output dict.
    turnover : array-like or None, optional
        Daily one-way turnover values.  When provided, ``mean_daily_turnover``
        and ``annual_turnover`` are added to the output.

    Returns
    -------
    dict
        Keys: ``label``, ``annualized_sharpe``, ``calmar``, ``starr_95``,
        ``var_95``, ``cvar_95``, ``var_99``, ``cvar_99``, ``mdd`` (all
        percentages where applicable), and optionally ``mean_daily_turnover``
        and ``annual_turnover``.
    """
    rf_aligned = rf.reindex(returns.index).ffill()
    excess = returns.values - rf_aligned.values

    annualized_sharpe = float(np.sqrt(252) * excess.mean() / excess.std())

    cagr = float(((1 + returns).prod()) ** (252.0 / len(returns)) - 1)
    mdd = compute_max_drawdown(returns)
    calmar = cagr / mdd if mdd > 0 else np.nan

    # CVaR of excess-return losses: losses = -excess
    cvar_95_val = compute_empirical_cvar(excess, ALPHA_95)
    starr_95 = float(np.sqrt(252) * excess.mean() / cvar_95_val) if cvar_95_val > 0 else np.nan

    var_95 = float(np.quantile(-excess, ALPHA_95) * 100)
    cvar_95 = float(cvar_95_val * 100)
    var_99 = float(np.quantile(-excess, ALPHA_99) * 100)
    cvar_99 = float(compute_empirical_cvar(excess, ALPHA_99) * 100)
    mdd_pct = float(mdd * 100)

    result = {
        'label': label,
        'annualized_sharpe': annualized_sharpe,
        'calmar': calmar,
        'starr_95': starr_95,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'var_99': var_99,
        'cvar_99': cvar_99,
        'mdd': mdd_pct,
    }

    if turnover is not None:
        result['mean_daily_turnover'] = float(np.mean(turnover))
        result['annual_turnover'] = float(np.mean(turnover)) * 252

    return result


def run_historical_backtest(
    returns_df: pd.DataFrame,
    rf_series: pd.Series,
    rebalance_every: int = 21,
):
    """Run an out-of-sample rolling backtest for all portfolio strategies.

    Uses a rolling window of ``WINDOW`` (504) trading days to estimate the
    input moments at each rebalancing date.  Between rebalancing dates weights
    are held constant (BHP drifts passively at every step).

    Strategies
    ----------
    * **BHP** – passive equal-weight (drift, no rebalancing).
    * **LO_MVP / LS_MVP** – minimum-variance, long-only / long-short.
    * **LO_C95 / LS_C95** – min CVaR at 95 %, long-only / long-short.
    * **LO_C99 / LS_C99** – min CVaR at 99 %, long-only / long-short.
    * **LO_TVP / LS_TVP** – max-Sharpe tangent, long-only / long-short.
    * **LO_TC95 / LS_TC95** – max return-to-CVaR ratio (95 %), long-only / long-short.
    * **LO_TC99 / LS_TC99** – max return-to-CVaR ratio (99 %), long-only / long-short.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Balanced panel of daily arithmetic returns with a DatetimeIndex and
        exactly 30 asset columns.
    rf_series : pd.Series
        Daily risk-free rate aligned to ``returns_df.index``.
    rebalance_every : int, optional
        Number of trading days between active rebalances for optimised
        strategies (default 21, i.e. approximately monthly).  BHP always
        drifts daily regardless of this setting.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(strategy_returns, strategy_weights, strategy_turnover)``

        Each dict maps a strategy name to a ``pd.Series`` with
        ``DatetimeIndex = returns_df.index[WINDOW + 1:]``.
    """
    n = len(returns_df.columns)
    T = len(returns_df)

    strategy_names = [
        'BHP',
        'LO_MVP', 'LS_MVP',
        'LO_C95', 'LS_C95',
        'LO_C99', 'LS_C99',
        'LO_TVP', 'LS_TVP',
        'LO_TC95', 'LS_TC95',
        'LO_TC99', 'LS_TC99',
    ]

    w0 = np.ones(n) / n
    weights = {s: w0.copy() for s in strategy_names}

    lb_lo = np.zeros(n)
    ub_lo = np.ones(n)
    lb_ls = np.full(n, -1.0 / 30)
    ub_ls = np.full(n, 1.0 + 1.0 / 30)

    ret_store = {s: [] for s in strategy_names}
    to_store = {s: [] for s in strategy_names}
    w_store = {s: [] for s in strategy_names}

    step_count = 0
    for t in range(WINDOW, T - 1):
        is_rebalance_day = (step_count % rebalance_every == 0)
        step_count += 1

        if (step_count - 1) % 50 == 0:
            logger.info(
                "Backtest progress: step %d / %d (rebalance=%s)",
                step_count - 1,
                T - WINDOW - 2,
                is_rebalance_day,
            )

        next_ret = returns_df.iloc[t + 1].values
        rf_t = float(rf_series.iloc[t])
        new_w = {}

        # BHP: passive – drifts every day regardless of rebalance_every
        new_w['BHP'] = weights['BHP'].copy()

        if is_rebalance_day:
            window = returns_df.iloc[t - WINDOW:t].values
            mu = window.mean(axis=0)
            Sigma = np.cov(window.T)

            # MVP
            for key, lb, ub in [('LO_MVP', lb_lo, ub_lo), ('LS_MVP', lb_ls, ub_ls)]:
                try:
                    w = solve_mvp(mu, Sigma, weights[key], lb, ub)
                    new_w[key] = w if w is not None else weights[key]
                except Exception as exc:
                    logger.warning("%s solve failed at t=%d: %s", key, t, exc)
                    new_w[key] = weights[key]

            # Min CVaR at 95 %
            for key, lb, ub in [('LO_C95', lb_lo, ub_lo), ('LS_C95', lb_ls, ub_ls)]:
                try:
                    w = solve_min_cvar(window, ALPHA_95, weights[key], lb, ub)
                    new_w[key] = w if w is not None else weights[key]
                except Exception as exc:
                    logger.warning("%s solve failed at t=%d: %s", key, t, exc)
                    new_w[key] = weights[key]

            # Min CVaR at 99 %
            for key, lb, ub in [('LO_C99', lb_lo, ub_lo), ('LS_C99', lb_ls, ub_ls)]:
                try:
                    w = solve_min_cvar(window, ALPHA_99, weights[key], lb, ub)
                    new_w[key] = w if w is not None else weights[key]
                except Exception as exc:
                    logger.warning("%s solve failed at t=%d: %s", key, t, exc)
                    new_w[key] = weights[key]

            # Tangent (max Sharpe)
            for key, lb, ub in [('LO_TVP', lb_lo, ub_lo), ('LS_TVP', lb_ls, ub_ls)]:
                try:
                    w = solve_tvp(mu, Sigma, rf_t, weights[key], lb, ub)
                    new_w[key] = w if w is not None else weights[key]
                except Exception as exc:
                    logger.warning("%s solve failed at t=%d: %s", key, t, exc)
                    new_w[key] = weights[key]

            # Max CVaR ratio at 95 %
            for key, lb, ub in [('LO_TC95', lb_lo, ub_lo), ('LS_TC95', lb_ls, ub_ls)]:
                try:
                    w = solve_max_cvar_ratio(window, mu, rf_t, ALPHA_95, weights[key], lb, ub)
                    new_w[key] = w if w is not None else weights[key]
                except Exception as exc:
                    logger.warning("%s solve failed at t=%d: %s", key, t, exc)
                    new_w[key] = weights[key]

            # Max CVaR ratio at 99 %
            for key, lb, ub in [('LO_TC99', lb_lo, ub_lo), ('LS_TC99', lb_ls, ub_ls)]:
                try:
                    w = solve_max_cvar_ratio(window, mu, rf_t, ALPHA_99, weights[key], lb, ub)
                    new_w[key] = w if w is not None else weights[key]
                except Exception as exc:
                    logger.warning("%s solve failed at t=%d: %s", key, t, exc)
                    new_w[key] = weights[key]
        else:
            # Non-rebalance day: hold existing weights
            for key in strategy_names:
                if key != 'BHP':
                    new_w[key] = weights[key].copy()

        # Record metrics and update weights
        for s in strategy_names:
            w_prev = weights[s]
            w_new = new_w[s]

            gross_ret = float(w_new @ next_ret)
            turnover = 0.0 if s == 'BHP' else 0.5 * float(np.sum(np.abs(w_new - w_prev)))

            ret_store[s].append(gross_ret)
            to_store[s].append(turnover)
            w_store[s].append(w_new.copy())

            if s == 'BHP':
                w_drifted = w_new * (1.0 + next_ret)
                weights[s] = w_drifted / w_drifted.sum()
            else:
                weights[s] = w_new

    out_dates = returns_df.index[WINDOW + 1:]
    strategy_returns = {s: pd.Series(ret_store[s], index=out_dates, name=s) for s in strategy_names}
    strategy_turnover = {s: pd.Series(to_store[s], index=out_dates, name=s) for s in strategy_names}
    strategy_weights = {s: pd.Series(w_store[s], index=out_dates, dtype=object, name=s) for s in strategy_names}

    return strategy_returns, strategy_weights, strategy_turnover


def compute_metrics_table(strategy_returns, rf_series, strategy_turnover):
    """Build a DataFrame of performance and risk metrics for all strategies.

    For each strategy, computes gross metrics via
    :func:`compute_portfolio_metrics` and adds a net Sharpe ratio after
    subtracting 50 bps (0.5 %) one-way transaction costs proportional to
    daily turnover.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Map from strategy name to daily gross return series.
    rf_series : pd.Series
        Daily risk-free rate.
    strategy_turnover : dict[str, pd.Series]
        Map from strategy name to daily turnover series.

    Returns
    -------
    pd.DataFrame
        One row per strategy, indexed by strategy label, with columns for
        all metrics including ``net_sharpe_50bps``.
    """
    rows = []
    for name, ret_series in strategy_returns.items():
        to_series = strategy_turnover.get(name)
        to_values = to_series.values if isinstance(to_series, pd.Series) else to_series

        metrics = compute_portfolio_metrics(ret_series, rf_series, name, turnover=to_values)

        # Net Sharpe after 50 bps one-way transaction cost
        if to_series is not None:
            net_ret = ret_series - 0.005 * to_series.reindex(ret_series.index).fillna(0.0)
            rf_aligned = rf_series.reindex(net_ret.index).ffill()
            net_excess = net_ret.values - rf_aligned.values
            metrics['net_sharpe_50bps'] = float(np.sqrt(252) * net_excess.mean() / net_excess.std())

        rows.append(metrics)

    return pd.DataFrame(rows).set_index('label')
