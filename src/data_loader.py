import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from massive import RESTClient

logger = logging.getLogger(__name__)

STUDY_START = '2018-12-12'
STUDY_END = '2024-12-16'
DATA_DIR = Path(__file__).parent.parent / 'data'

ETF_UNIVERSE = {
    'Agriculture': ['WEAT', 'CORN', 'DBA'],
    'Energy': ['OILK', 'USL', 'UGA', 'BNO', 'DBE', 'DBO', 'USO', 'UNG'],
    'Metals': ['DBP', 'CPER', 'PALL', 'DBB', 'GLTR', 'PPLT', 'GLDM', 'SLV', 'IAU', 'GLD'],
    'Broad': ['TAGS', 'USCI', 'FTGC', 'CMDY', 'BCI', 'COMT', 'GSG', 'DBC', 'PDBC'],
}
ALL_TICKERS = [t for tickers in ETF_UNIVERSE.values() for t in tickers]
ETF_CATEGORIES = {t: cat for cat, tickers in ETF_UNIVERSE.items() for t in tickers}


def download_etf_prices(
    tickers: list[str],
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame:
    """Download daily close prices for a list of ETF tickers via the Polygon REST API.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end : str
        End date in 'YYYY-MM-DD' format (inclusive).
    api_key : str
        Polygon.io API key passed to the RESTClient.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of daily close prices indexed by date (DatetimeIndex),
        one column per ticker.  Tickers that fail are silently omitted.
    """
    client = RESTClient(api_key)
    price_series: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            aggs = client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=start,
                to=end,
                limit=50000,
            )
            records = {
                pd.Timestamp(agg.timestamp, unit='ms').date(): agg.close
                for agg in aggs
            }
            if not records:
                logger.warning("No data returned for ticker %s; skipping.", ticker)
                continue
            price_series[ticker] = pd.Series(records, name=ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to download %s: %s; skipping.", ticker, exc)

    if not price_series:
        return pd.DataFrame()

    prices_df = pd.DataFrame(price_series)
    prices_df.index = pd.DatetimeIndex(prices_df.index)
    prices_df.index.name = 'date'
    prices_df.sort_index(inplace=True)
    return prices_df


def download_tbill_rate(start: str, end: str) -> pd.Series:
    """Download the 13-week T-bill annualised yield (^IRX) from Yahoo Finance.

    The returned series is expressed as a decimal (e.g. 5 % → 0.05).

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.Series
        Daily annualised T-bill yield as a decimal, indexed by date.
        Falls back to a flat 0.05 series on business dates if the download
        fails.
    """
    try:
        raw = web.DataReader('^IRX', 'yahoo', start, end)
        yield_series = raw['Close'] / 100.0
        yield_series.index = pd.DatetimeIndex(yield_series.index)
        yield_series.name = 'rf_annual'
        return yield_series
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not download T-bill rate (%s). "
            "ASSUMPTION: using constant annualised risk-free rate of 5 %% (0.05) "
            "for the entire study period.",
            exc,
        )
        biz_dates = pd.bdate_range(start=start, end=end)
        return pd.Series(0.05, index=biz_dates, name='rf_annual')


def compute_arithmetic_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) daily returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide DataFrame of prices with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Daily arithmetic returns, first row dropped (NaN after pct_change).
    """
    return prices.pct_change().dropna()


def convert_tbill_to_daily(yield_series: pd.Series) -> pd.Series:
    """Convert annualized yield (decimal) to daily rate using simple convention.

    ASSUMPTION: simple daily rate = annualized yield / 252.

    Parameters
    ----------
    yield_series : pd.Series
        Annualised yield expressed as a decimal (e.g. 0.05 for 5 %).

    Returns
    -------
    pd.Series
        Daily risk-free rate (decimal).
    """
    return yield_series / 252


def build_balanced_panel(
    returns: pd.DataFrame,
    rf_daily: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align returns and the daily risk-free rate on their common dates.

    Rows containing any NaN value in the returns DataFrame are dropped.
    The risk-free series is then restricted to the same dates.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns with a DatetimeIndex.
    rf_daily : pd.Series
        Daily risk-free rate with a DatetimeIndex.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(balanced_returns, aligned_rf)`` where both share an identical index.
    """
    common_dates = returns.index.intersection(rf_daily.index)
    aligned_returns = returns.loc[common_dates]
    aligned_rf = rf_daily.loc[common_dates]

    n_before = len(aligned_returns)
    aligned_returns = aligned_returns.dropna()
    n_dropped = n_before - len(aligned_returns)
    if n_dropped:
        logger.info("build_balanced_panel: dropped %d date(s) containing NaN.", n_dropped)

    aligned_rf = aligned_rf.loc[aligned_returns.index]
    return aligned_returns, aligned_rf


def _generate_synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate realistic synthetic daily prices and a T-bill yield series.

    Uses correlated GBM with category-specific drift and volatility parameters
    calibrated to broadly match commodity ETF behaviour.  This fallback is
    used only when no Polygon API key is available.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(prices_df, tbill_yield)``
        prices_df  – wide DataFrame of simulated daily close prices.
        tbill_yield – annualised 3-month T-bill yield as a decimal (daily index).
    """
    rng = np.random.default_rng(0)

    # Business-day date range
    dates = pd.bdate_range(start=STUDY_START, end=STUDY_END)
    T = len(dates)
    n = len(ALL_TICKERS)

    # Per-asset annual drift (mu) and volatility (sigma) – realistic ranges
    category_params: dict[str, tuple[float, float]] = {
        'Agriculture': (0.02, 0.25),
        'Energy':      (-0.02, 0.40),
        'Metals':      (0.06, 0.15),
        'Broad':       (0.01, 0.20),
    }

    mus = np.array([category_params[ETF_CATEGORIES[t]][0] for t in ALL_TICKERS]) / 252
    sigmas = np.array([category_params[ETF_CATEGORIES[t]][1] for t in ALL_TICKERS]) / np.sqrt(252)

    # Build a block-correlation matrix: 0.70 within category, 0.25 across
    C = np.full((n, n), 0.25)
    cat_list = [ETF_CATEGORIES[t] for t in ALL_TICKERS]
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i, j] = 1.0
            elif cat_list[i] == cat_list[j]:
                C[i, j] = 0.70

    # Cholesky decomposition for correlated shocks
    L = np.linalg.cholesky(C)

    # Simulate log-returns and compound to prices
    z = rng.standard_normal((T, n))
    z_corr = z @ L.T  # correlated standard normals
    log_rets = mus - 0.5 * sigmas ** 2 + sigmas * z_corr
    log_prices = np.cumsum(log_rets, axis=0)
    prices = 100.0 * np.exp(log_prices)

    prices_df = pd.DataFrame(prices, index=dates, columns=ALL_TICKERS)
    prices_df.index.name = 'date'

    # Synthetic T-bill: starts ~2 %, rises to ~5.3 %, ticks daily
    t_arr = np.arange(T)
    tbill_rate = 0.02 + 0.033 * (1 - np.exp(-t_arr / (T * 0.5)))
    tbill_rate += rng.normal(0, 0.0003, T)
    tbill_yield = pd.Series(tbill_rate, index=dates, name='rf_annual')

    logger.info(
        "Synthetic data generated: %d assets × %d trading days "
        "(GBM, block-correlation, no real market data).",
        n, T,
    )
    return prices_df, tbill_yield


def load_or_download_data(
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Load cached data from disk or download it if the cache is absent.

    Checks for ``DATA_DIR/returns.parquet``.  If found, loads all artefacts
    from disk and returns them.  Otherwise downloads prices and the T-bill
    rate, builds the balanced panel, persists artefacts, and returns them.

    Parameters
    ----------
    api_key : str | None
        Polygon.io API key.  Required when downloading; ignored when loading
        from cache.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]
        ``(prices_df, returns_df, rf_daily, metadata)``

        * **prices_df** – raw daily close prices.
        * **returns_df** – balanced panel of arithmetic daily returns.
        * **rf_daily** – aligned daily risk-free rate.
        * **metadata** – dict with keys ``categories``, ``start``, ``end``,
          ``rf_convention``, ``n_assets``, ``n_tickers_by_category``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = DATA_DIR / 'returns.parquet'
    prices_path = DATA_DIR / 'prices.parquet'
    rf_path = DATA_DIR / 'rf.parquet'
    meta_path = DATA_DIR / 'metadata.json'

    if returns_path.exists():
        logger.info("Loading cached data from %s.", DATA_DIR)
        returns_df = pd.read_parquet(returns_path)
        rf_daily = pd.read_parquet(rf_path)['rf']
        rf_daily.name = 'rf_annual'

        if prices_path.exists():
            prices_df = pd.read_parquet(prices_path)
        else:
            logger.warning(
                "prices.parquet not found; reconstructing approximate prices "
                "from returns via cumprod.  Absolute price levels will differ "
                "from original download.",
            )
            prices_df = (1 + returns_df).cumprod()

        with open(meta_path) as fh:
            metadata = json.load(fh)

        return prices_df, returns_df, rf_daily, metadata

    # ------------------------------------------------------------------ #
    # Fresh download (or synthetic fallback)                              #
    # ------------------------------------------------------------------ #
    if api_key:
        logger.info("Downloading ETF prices for %d tickers.", len(ALL_TICKERS))
        prices_df = download_etf_prices(ALL_TICKERS, STUDY_START, STUDY_END, api_key)
        logger.info("Downloading T-bill rate.")
        yield_series = download_tbill_rate(STUDY_START, STUDY_END)
        rf_daily = convert_tbill_to_daily(yield_series)
    else:
        logger.warning(
            "No MASSIVE_TOKEN found – using SYNTHETIC GBM data as a reproducible "
            "placeholder.  Results are for methodology validation only, NOT real "
            "market analysis."
        )
        prices_df, yield_series = _generate_synthetic_data()
        rf_daily = convert_tbill_to_daily(yield_series)

    raw_returns = compute_arithmetic_returns(prices_df)
    returns_df, rf_daily = build_balanced_panel(raw_returns, rf_daily)

    n_tickers_by_category = {
        cat: len(tickers) for cat, tickers in ETF_UNIVERSE.items()
    }
    metadata: dict = {
        'categories': ETF_CATEGORIES,
        'start': STUDY_START,
        'end': STUDY_END,
        'rf_convention': 'simple_daily_annualized_over_252',
        'n_assets': len(returns_df.columns),
        'n_tickers_by_category': n_tickers_by_category,
        'data_source': 'polygon_massive_api' if api_key else 'synthetic_gbm_fallback',
    }

    prices_df.to_parquet(prices_path)
    returns_df.to_parquet(returns_path)
    rf_daily.to_frame(name='rf').to_parquet(rf_path)
    with open(meta_path, 'w') as fh:
        json.dump(metadata, fh, indent=2, default=str)

    logger.info(
        "Saved data artefacts to %s (%d assets, %d dates).",
        DATA_DIR,
        len(returns_df.columns),
        len(returns_df),
    )
    return prices_df, returns_df, rf_daily, metadata


def get_audit_table(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-ticker data-quality audit table.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide DataFrame of daily close prices with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        One row per ticker with columns:
        ``ticker``, ``first_date``, ``last_date``, ``n_obs``,
        ``missing_count``, ``category``.
    """
    rows = []
    for ticker in prices_df.columns:
        col = prices_df[ticker]
        rows.append({
            'ticker': ticker,
            'first_date': col.first_valid_index(),
            'last_date': col.last_valid_index(),
            'n_obs': col.notna().sum(),
            'missing_count': col.isna().sum(),
            'category': ETF_CATEGORIES.get(ticker, 'Unknown'),
        })
    return pd.DataFrame(rows)
