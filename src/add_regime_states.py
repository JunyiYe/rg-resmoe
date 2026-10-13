"""Add market (systemic) and idiosyncratic volatility features to the panel.

market_ret(t): cross-sectional (equal-weighted) average log return across all
tickers on day t -- a synthetic market-index return series, used in place of
SPY since the panel doesn't otherwise carry an index series.

For each ticker: a rolling beta (120-day window) of its own return against
market_ret, then idiosyncratic return = stock_ret - beta * market_ret (a
CAPM-residual proxy, not a level subtraction of vol). market_vol_20 and
idio_vol_20 are then the 20-day annualized rolling std of market_ret and the
idiosyncratic return series respectively -- market_vol_20 is identical across
tickers on a given day (it's a property of the market only); idio_vol_20 is
ticker-specific.

Saved as a separate panel_systemic.parquet (not overwriting panel.parquet)
so existing scripts/results that read panel.parquet are unaffected.
"""
import os

import numpy as np
import pandas as pd

from walk_forward import PROCESSED_DIR, add_lag_features

BETA_WINDOW = 120


def main():
    # PANEL_IN/PANEL_OUT env overrides support secondary panels (e.g. Japan:
    # PANEL_IN=panel_jp.parquet PANEL_OUT=panel_systemic_jp.parquet).
    panel = pd.read_parquet(PROCESSED_DIR / os.environ.get("PANEL_IN", "panel.parquet"))
    panel = add_lag_features(panel)
    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)

    market_ret = panel.groupby("Date")["log_ret"].mean().rename("market_ret")
    panel = panel.merge(market_ret, on="Date", how="left")

    market_vol_series = market_ret.sort_index().rolling(20, min_periods=5).std() * np.sqrt(252)
    market_vol_series = market_vol_series.rename("market_vol_20")

    print("computing rolling beta + idiosyncratic returns per ticker...")
    groups = []
    for _, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("Date").copy()
        cov = g["log_ret"].rolling(BETA_WINDOW, min_periods=21).cov(g["market_ret"])
        var = g["market_ret"].rolling(BETA_WINDOW, min_periods=21).var()
        beta = cov / var.replace(0, np.nan)
        beta = beta.clip(-5, 5)
        g["beta"] = beta
        g["idio_ret"] = g["log_ret"] - beta * g["market_ret"]
        groups.append(g)
    panel = pd.concat(groups, ignore_index=True)

    panel["idio_vol_20"] = panel.groupby("ticker")["idio_ret"].transform(
        lambda s: s.rolling(20, min_periods=5).std() * np.sqrt(252)
    )
    panel = panel.merge(market_vol_series, left_on="Date", right_index=True, how="left")

    before = len(panel)
    panel = panel.dropna(subset=["beta", "idio_vol_20", "market_vol_20"]).reset_index(drop=True)
    print(f"rows: {before} -> {len(panel)} after dropping beta/idio warmup NaNs")

    out_path = PROCESSED_DIR / os.environ.get("PANEL_OUT", "panel_systemic.parquet")
    panel.to_parquet(out_path, index=False)
    print(f"saved {out_path}")
    print(f"tickers={panel['ticker'].nunique()}  rows={len(panel)}")
    print("\nmarket_vol_20 describe:\n", panel["market_vol_20"].describe())
    print("\nidio_vol_20 describe:\n", panel["idio_vol_20"].describe())
    print("\nbeta describe:\n", panel["beta"].describe())
    print("\ncorrelation(market_vol_20, idio_vol_20):", panel["market_vol_20"].corr(panel["idio_vol_20"]))
    print("correlation(vol_20, idio_vol_20):", panel["vol_20"].corr(panel["idio_vol_20"]))
    print("correlation(vol_20, market_vol_20):", panel["vol_20"].corr(panel["market_vol_20"]))


if __name__ == "__main__":
    main()
