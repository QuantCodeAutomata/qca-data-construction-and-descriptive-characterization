# Commodity ETF Portfolio Optimization

A reproducible research repository implementing five experiments on a universe of
30 U.S.-listed commodity ETFs over 12 Dec 2018 – 16 Dec 2024.

## Repository Structure

```
repo/
├── data/               # Cleaned daily price / return panel (parquet) + metadata
├── src/                # Core modules: data loader, descriptive stats, optimizers
│   ├── data_loader.py
│   ├── descriptive_stats.py
│   ├── historical_optimizer.py
│   ├── tail_diagnostics.py
│   ├── dynamic_optimizer.py
│   └── performance_comparison.py
├── exp/                # Experiment entry-points (exp1 – exp5)
├── tests/              # pytest unit + integration tests
├── results/            # Artefacts: tables, plots, RESULTS.md
├── run_all.py          # Sequential pipeline runner
└── requirements.txt
```

## Experiments

| # | Title | Key techniques |
|---|-------|---------------|
| 1 | Descriptive characterisation | Summary stats, correlation heatmap, normalised plots |
| 2 | Historical rolling-window optimisation | MV / CVaR / Tangent – long-only & restricted LS, turnover cap |
| 3 | Extreme-value tail diagnostics | Hill estimator, empirical VaR/CVaR/MDD |
| 4 | Dynamic ARMA-GARCH + Student-t copula | AR(1)-GARCH(1,1) marginals, t-copula scenarios, predictive optimisation |
| 5 | Historical vs dynamic comparison | Transaction-cost analysis, moving-block bootstrap CI |

## Data

ETF universe (30 assets across Agriculture, Energy, Metals, Broad Commodity Index):

**Agriculture:** WEAT, CORN, DBA  
**Energy:** OILK, USL, UGA, BNO, DBE, DBO, USO, UNG  
**Metals:** DBP, CPER, PALL, DBB, GLTR, PPLT, GLDM, SLV, IAU, GLD  
**Broad Commodity Index:** TAGS, USCI, FTGC, CMDY, BCI, COMT, GSG, DBC, PDBC

Data is loaded via the Massive (Polygon.io) REST API when `MASSIVE_TOKEN` is
set; otherwise a synthetic GBM panel with block correlation is used as fallback.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments sequentially
python run_all.py

# Run individual experiments
python -m exp.exp1_descriptive
python -m exp.exp2_historical
python -m exp.exp3_tail
python -m exp.exp4_dynamic
python -m exp.exp5_comparison

# Run tests
pytest tests/ -v
```

## Key Assumptions

- **Risk-free rate:** simple daily rate = annualised 3-month T-bill yield / 252.
- **Returns:** arithmetic daily returns from adjusted close prices.
- **Rolling window:** 504 trading days.
- **ARMA-GARCH order:** AR(1)-GARCH(1,1) (paper unspecified).
- **Copula:** Student-t; dof estimated as median of univariate t-fits.
- **Scenarios:** 2 000 Monte Carlo draws per rebalance period.
- **Turnover cap:** C_TO = 1 / (30 × 252) ≈ 1.32 × 10⁻⁴ per rebalance.
- **Transaction costs:** 0 bps and 50 bps per unit turnover.
- **Bootstrap:** moving-block, block length = 20, B = 2 000 replications.

See `results/RESULTS.md` for full results and artefacts.
