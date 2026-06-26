# AGENTS.md – Repository Knowledge Base

## Project Overview
Commodity ETF dynamic portfolio optimisation study.  
Universe: 30 ETFs across Agriculture, Energy, Metals, Broad categories (see `src/data_loader.py`).  
Study period: 2018-12-12 → 2024-12-16.  
Data source: Polygon.io REST API (`MASSIVE_TOKEN` env variable).

## Module Map
| File | Purpose |
|------|---------|
| `src/data_loader.py` | Download / cache prices + T-bill rate; `load_or_download_data()` is the main entry point |
| `src/descriptive_stats.py` | Summary statistics, correlation, plots |
| `src/dynamic_optimizer.py` | ARMA-GARCH marginals + Student-t copula + four CVXPY solvers + rolling backtest |
| `src/performance_comparison.py` | Sharpe / Calmar / STARR, net-of-cost returns, MBB bootstrap, plots |
| `src/historical_optimizer.py` | Rolling historical backtest: 13 strategies (BHP, LO/LS × MVP/CVaR95/CVaR99/TVP/TC95/TC99); `run_historical_backtest`, `compute_metrics_table` |
| `src/tail_diagnostics.py` | Hill tail-index estimator, empirical VaR/CVaR, MDD, bar/curve plots |
| `exp/exp1_descriptive.py` | Experiment 1: descriptive stats |
| `exp/exp2_historical.py` | Experiment 2: run historical backtest, save parquets + cumulative-value PNG |
| `exp/exp3_tail.py` | Experiment 3: Hill curves and tail-metric comparison from exp2 outputs |
| `exp/exp4_dynamic.py` | Experiment 4: run dynamic backtest, save parquet + PNG |
| `exp/exp5_comparison.py` | Experiment 5: compare historical vs dynamic, bootstrap tests |

## Key Design Decisions

### historical_optimizer.py
- Constants: `WINDOW=504`, `C_TO=1/(30*252)`, `ALPHA_95=0.95`, `ALPHA_99=0.99`.
- **CVaR convention**: `compute_empirical_cvar(returns, alpha)` takes *returns* and internally sets `losses = -returns`.  In `compute_portfolio_metrics`, pass `excess` (not `-excess`) to get CVaR of excess-return losses — this is consistent with `var_95 = np.quantile(-excess, 0.95)`.
- Bounds: LO → `lb=0, ub=1`; LS → `lb=-1/30, ub=1+1/30`.
- `solve_max_cvar_ratio`: determines CVaR frontier range by solving min-CVaR first; upper bound = max(eq-weight CVaR, 2× min-CVaR); 30-point linspace.
- BHP drifts: after each day, `w_bhp *= (1 + next_ret)` then renormalize; all other strategies hold solved weights until next rebalancing.
- Turnover stored as `0.5 * |w_new - w_prev|`; BHP turnover = 0.
- `compute_metrics_table` adds `net_sharpe_50bps` = Sharpe after subtracting `0.005 * turnover` from daily returns.

### tail_diagnostics.py
- `compute_empirical_cvar(losses, alpha)`: takes *losses* directly (not returns) — different from the `historical_optimizer` version.
- `hill_estimator`: filters to positive losses, sorts descending; `k_min=max(5, ceil(0.01*m))`, `k_max=floor(0.05*m)`; skips k if any `log(X[:k]/X[k]) <= 0`.

### dynamic_optimizer.py
- `arch_model(returns*100, mean='ARX', lags=1, vol='GARCH', p=1, q=1)` – model fitted on 100× returns for numerical stability; forecasts divided back by 100.
- `result.resid` / `result.conditional_volatility` are **numpy arrays** (not pandas Series) when the arch input is a numpy array; always use `np.asarray(...)` + `np.isfinite(...)` to filter NaN, never `.notna()`.
- GARCH initialisation produces exactly 1 NaN at position 0; after filtering, `std_resids.std ≈ 1.0` as expected.
- `fit_student_t_copula`: drops non-finite rows from Z before `corrcoef` and `t.fit`; dof clipped to [2.5, 30].
- 12 strategy slots: 6 objectives (MVP, CVaR-95, CVaR-99, TVP, MCR-95, MCR-99) × 2 bound sets (LO=[0,1], SS=[-0.3,1.3]), plus BHP (equal-weight drift).
- `_solve_tvp_scenarios` / `_solve_max_cvar_ratio_scenarios` use a 30-point / 20-point efficient-frontier grid search.
- Turnover cap `C_TO = 1/(30×252)` enforced via auxiliary variable `d` with `sum(d) ≤ 2×C_TO`.

### performance_comparison.py
- `matplotlib.use('Agg')` called at module level before any `import matplotlib.pyplot`.
- `moving_block_bootstrap_difference`: manually constructs overlapping blocks of length `BLOCK_SIZE=20`; resamples with `np.random.default_rng(42)` for reproducibility; CI from 2.5/97.5 percentiles.
- `HAS_ARCH_BOOTSTRAP` flag present but unused (manual MBB is always used).

## Common Pitfalls
- `arch` result attributes are **numpy arrays** when input is a plain numpy array. Calling `.notna()` raises `AttributeError`, which silently falls into the except-branch and returns the unscaled fallback (std_resids ≈ 0.01 instead of ≈ 1, causing scenario std ≈ 0.0001 instead of ≈ 0.01).
- Ensure `generate_predictive_scenarios` receives a raw numpy array `returns_window` (extracted via `.values` from a DataFrame).
- CVXPY default solver works reliably; no need to force `cp.CLARABEL` explicitly.

## Experiment Results Directory
`results/` – all CSVs, Parquets, and PNGs written here.  
Key artefacts:
- `exp4_dynamic_returns_varA.parquet` – 13-strategy daily returns from dynamic backtest
- `exp4_dynamic_turnover_varA.parquet` – daily turnover per strategy
- `exp5_gross_metrics.csv`, `exp5_net_metrics.csv`, `exp5_bootstrap_results.csv`

## Running Experiments
```bash
MASSIVE_TOKEN=<key> python exp/exp2_historical.py   # historical backtest → exp2_strategy_returns.parquet
python exp/exp3_tail.py                             # tail diagnostics → requires exp2 outputs
MASSIVE_TOKEN=<key> python exp/exp4_dynamic.py
MASSIVE_TOKEN=<key> python exp/exp5_comparison.py   # requires exp2 + exp4 outputs
```

## Key Artefacts in results/
- `exp2_strategy_returns.parquet` – 13-strategy daily gross returns from historical backtest
- `exp2_turnover.parquet` – daily turnover per strategy
- `exp2_metrics_gross.csv` – full performance table incl. `net_sharpe_50bps`
- `exp2_cumulative_value.png` – all-strategy cumulative value chart
- `exp3_tail_metrics.csv` – VaR/CVaR/MDD/Hill alpha per strategy
- `exp3_hill_curves.png`, `exp3_tail_comparison.png` – tail diagnostic plots
