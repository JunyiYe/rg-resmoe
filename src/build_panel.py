"""Build a feature panel (returns, realized vol, momentum) from the raw OHLCV CSVs.

Output: data/processed/panel.parquet with one row per (ticker, date).
"""
import os
import pathlib

import numpy as np
import pandas as pd

# RAW_DIR_NAME/PANEL_OUT env overrides support secondary panels (e.g. the
# Japan appendix panel: RAW_DIR_NAME=raw_jp PANEL_OUT=panel_jp.parquet);
# defaults reproduce the original U.S. build exactly.
RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / os.environ.get("RAW_DIR_NAME", "raw")
PANEL_OUT = os.environ.get("PANEL_OUT", "panel.parquet")
TARGET_HORIZON = int(os.environ.get("TARGET_HORIZON", "5"))
TARGET_COL = f"target_fwd_vol_{TARGET_HORIZON}"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# SW (Smurfit WestRock) has zero-volume/frozen-price days scattered across
# ~87% of its nominal 2015-2025 history (not just a pre-listing prefix), so
# after masking those out almost nothing usable remains. RIBT spends much of
# 2024-2025 trading in fractions of a cent (real trades, real volume, but
# tick-size-driven "returns" of thousands of percent that swamp any vol
# feature); it's the only other ticker in the universe with sub-$0.10 days.
# Both are excluded entirely rather than kept as degenerate stubs.
EXCLUDED_TICKERS = {"SW", "RIBT"}

# CHRD (Chord Energy): Whiting Petroleum's Chapter 11 legally emerged
# 2020-09-01 (75:1 share exchange), but the vendor feed under this ticker
# keeps pricing the old, soon-to-be-cancelled equity smoothly down to
# ~$0.07 through 2020-11-19, then jumps to ~$19-31 on 2020-11-20 -- too
# large to be the 75:1 exchange ratio itself, so this is the point the feed
# actually switches to the reorganized entity. Truncate to that series,
# same as a fresh listing.
TRUNCATE_START = {"CHRD": "2020-11-20"}


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_one(path: pathlib.Path, ticker: str, sector: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    if ticker in TRUNCATE_START:
        df = df[df["Date"] >= pd.Timestamp(TRUNCATE_START[ticker])].reset_index(drop=True)

    df["ticker"] = ticker
    df["sector"] = sector

    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]

    # Volume == 0 marks days the vendor backfilled with a stale/frozen price
    # (pre-listing history, trading halts) rather than a real trading day.
    # Treat the price as missing so log_ret/rsi/rolling windows over these
    # days come out NaN instead of a fabricated zero return, and get dropped
    # below rather than silently biasing vol/momentum features and the target.
    df.loc[df["Volume"] == 0, price_cols] = np.nan

    df["log_ret"] = np.log(df["Adj Close"]).diff()
    # realized vol = annualized rolling std of daily log returns
    df["vol_5"] = df["log_ret"].rolling(5).std() * np.sqrt(252)
    df["vol_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
    df["vol_60"] = df["log_ret"].rolling(60).std() * np.sqrt(252)
    df["ret_5"] = df["log_ret"].rolling(5).sum()
    df["ret_20"] = df["log_ret"].rolling(20).sum()
    df["rsi_14"] = rsi(df["Close"], 14)

    # forecasting target: forward realized vol over days t+1..t+H (H=5 by
    # default; TARGET_HORIZON env builds the horizon-sweep panels), shifted
    # so it is strictly future information relative to row date.
    df[TARGET_COL] = df["log_ret"].shift(-1).rolling(TARGET_HORIZON).std().shift(-(TARGET_HORIZON - 1)) * np.sqrt(252)
    # next-day return, used later for the inverse-vol-targeting backtest
    df["fwd_ret_1"] = df["log_ret"].shift(-1)

    return df


def main():
    manifest = pd.read_csv(RAW_DIR / "manifest.csv")
    manifest = manifest[~manifest["ticker"].isin(EXCLUDED_TICKERS)]
    frames = []
    for _, row in manifest.iterrows():
        path = RAW_DIR / f"{row['ticker']}.csv"
        frames.append(build_one(path, row["ticker"], row["sector"]))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(
        subset=["vol_5", "vol_20", "vol_60", "ret_5", "ret_20", "rsi_14", TARGET_COL, "fwd_ret_1"]
    ).reset_index(drop=True)
    out_path = PROCESSED_DIR / PANEL_OUT
    panel.to_parquet(out_path, index=False)
    print(f"panel shape: {panel.shape}")
    print(f"date range: {panel['Date'].min()} -> {panel['Date'].max()}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
