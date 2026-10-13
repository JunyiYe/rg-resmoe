"""Download daily OHLCV for the Japan TSE Prime panel via yfinance.

Same source and CSV schema as the U.S. panel (data/raw/), so build_panel.py
can be pointed at data/raw_jp/ unchanged. Universe: data/raw_jp/
tse_prime_constituents.csv (1,559 TSE Prime domestic common stocks, from
JPX's official listed-issues file). The shipped data/raw_jp/ already
contains this run's output; re-running only re-fetches missing tickers.

Usage: python src/download_data_jp.py
Output: data/raw_jp/<ticker>.csv + data/raw_jp/manifest.csv
"""
import pathlib
import time

import pandas as pd
import yfinance as yf

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw_jp"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Same window as the U.S. panel (data/raw/: 2015-12-01 .. 2025-11-28).
START = "2015-12-01"
END = "2025-12-01"

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def fetch_one(ticker: str) -> pd.DataFrame | None:
    df = yf.download(
        ticker, start=START, end=END, auto_adjust=False,
        threads=False, progress=False,
    )
    if df is None or not len(df):
        return None
    df = df.dropna(how="all").reset_index()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    if not set(PRICE_COLS).issubset(df.columns):
        return None
    return df[["Date"] + PRICE_COLS]


def main():
    # One ticker at a time, no thread fan-out, a short sleep between
    # requests, resumable (skips tickers whose CSV already exists).
    cons = pd.read_csv(RAW_DIR / "tse_prime_constituents.csv", dtype={"code": str})
    tickers = [f"{c}.T" for c in cons["code"]]
    sector = {f"{c}.T": s for c, s in zip(cons["code"], cons["sector"])}

    manifest = []
    n_new = 0
    for i, t in enumerate(tickers):
        out_path = RAW_DIR / f"{t}.csv"
        if out_path.exists():
            df = pd.read_csv(out_path)
        else:
            try:
                df = fetch_one(t)
            except Exception as e:
                print(f"FAILED {t}: {e}", flush=True)
                df = None
            if df is None:
                print(f"FAILED: {t}", flush=True)
                continue
            df.to_csv(out_path, index=False)
            n_new += 1
            if i % 20 == 0:
                print(f"{i+1}/{len(tickers)} done", flush=True)
            time.sleep(1.5)
        manifest.append({
            "ticker": t,
            "sector": sector[t],
            "n_rows": len(df),
            "start": str(df["Date"].min())[:10],
            "end": str(df["Date"].max())[:10],
        })
    pd.DataFrame(manifest).to_csv(RAW_DIR / "manifest.csv", index=False)
    print(f"done: {len(manifest)}/{len(tickers)} tickers ({n_new} newly fetched) -> {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
