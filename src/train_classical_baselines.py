"""
Walk-forward GARCH(1,1) and HAR-RV baselines.
Same protocol as neural models: TRAIN_WINDOW=504, STEP=63, VAL_FRAC=0.15.
Also computes historical-vol (vol_20) as a trivial floor baseline.

Outputs:
  results/predictions/classical/garch.parquet       — GARCH(1,1) predictions
  results/predictions/classical/har.parquet      — HAR-RV predictions
  results/predictions/classical/persistence.parquet    — Historical vol (vol_20) baseline
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# ── config (matches walk_forward.py) ───────────────────────────────────────
DATA_PATH         = "data/processed/panel_systemic.parquet"
RESULTS_DIR       = "results/predictions/classical"  # classical baselines only ever ran for the US main experiment
TARGET            = "target_fwd_vol_5"
TRAIN_WINDOW_DAYS = 504
STEP_DAYS         = 63
VAL_FRAC          = 0.15   # last 15% of train window = val (unused here, kept for parity)
MIN_TRAIN_OBS     = 100    # min returns to fit GARCH per stock
GARCH_MAX_ITER    = 200


def walk_forward_windows(dates):
    """Yield (window_start, cutoff, test_start, test_end) for each rebalance date."""
    rebalance = dates[TRAIN_WINDOW_DAYS::STEP_DAYS]
    for cutoff in rebalance:
        idx = np.searchsorted(dates, cutoff)
        window_start = dates[max(0, idx - TRAIN_WINDOW_DAYS)]
        test_dates   = dates[idx: idx + STEP_DAYS]
        if len(test_dates) == 0:
            continue
        yield window_start, cutoff, test_dates[0], test_dates[-1]


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────
def garch_forecast_5d(returns, horizon=5):
    """
    Fit GARCH(1,1) on `returns`, return annualized 5-day ahead vol forecast.
    Returns scalar (annualized realized vol ≈ sqrt(sum_h σ²_h × 252)).
    """
    from arch import arch_model
    if len(returns) < MIN_TRAIN_OBS or returns.std() < 1e-8:
        return returns.std() * np.sqrt(252)
    try:
        am  = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": GARCH_MAX_ITER})
        fc  = res.forecast(horizon=horizon, reindex=False)
        # variance in (ret*100)² units; convert back and annualize
        var_h = fc.variance.values[-1]          # shape (horizon,)
        daily_var = var_h / (100**2)             # back to decimal returns
        # Match the target's unit convention: target_fwd_vol_5 is the
        # STD OF DAILY RETURNS over t+1..t+5, annualized by sqrt(252)
        # (build_panel.py). That is sqrt(mean daily variance * 252),
        # not sqrt(sum * 252) -- summing inflates the forecast by sqrt(5).
        rv5 = np.sqrt(daily_var.mean() * 252)   # annualized daily vol over 5d
        if not np.isfinite(rv5) or rv5 <= 0:
            return returns.std() * np.sqrt(252)
        return rv5
    except Exception:
        return returns.std() * np.sqrt(252)


def run_garch(panel):
    """Walk-forward GARCH(1,1): one forecast per (Date, ticker) in test windows."""
    dates = np.sort(panel["Date"].unique())
    records = []

    for w_idx, (wstart, cutoff, tstart, tend) in enumerate(walk_forward_windows(dates)):
        wstart_d = pd.Timestamp(wstart).date(); cutoff_d = pd.Timestamp(cutoff).date()
        tstart_d = pd.Timestamp(tstart).date(); tend_d   = pd.Timestamp(tend).date()
        print(f"  window {w_idx+1}: {wstart_d} → {cutoff_d} | test {tstart_d}–{tend_d}", flush=True)
        train_mask = (panel["Date"] >= wstart) & (panel["Date"] < cutoff)
        test_mask  = (panel["Date"] >= tstart) & (panel["Date"] <= tend)
        train_df   = panel[train_mask]
        test_df    = panel[test_mask][["Date", "ticker", TARGET, "fwd_ret_1"]].copy()
        if test_df.empty:
            continue

        # fit one GARCH per stock using all training returns
        garch_fc = {}
        for ticker, grp in train_df.groupby("ticker"):
            ret = grp.sort_values("Date")["log_ret"].dropna().values
            garch_fc[ticker] = garch_forecast_5d(ret)

        test_df["garch_pred"] = test_df["ticker"].map(garch_fc)
        records.append(test_df.dropna(subset=["garch_pred"]))

    return pd.concat(records, ignore_index=True)


# ── HAR-RV ───────────────────────────────────────────────────────────────────
def run_har_rv(panel):
    """
    Walk-forward HAR-RV (Corsi 2009) using vol_5, vol_20, vol_60 as
    daily/weekly/monthly RV proxies. Fit OLS per window (pooled across stocks).
    """
    dates  = np.sort(panel["Date"].unique())
    FEAT   = ["vol_5", "vol_20", "vol_60"]
    records = []

    for w_idx, (wstart, cutoff, tstart, tend) in enumerate(walk_forward_windows(dates)):
        wstart_d = pd.Timestamp(wstart).date(); cutoff_d = pd.Timestamp(cutoff).date()
        tstart_d = pd.Timestamp(tstart).date(); tend_d   = pd.Timestamp(tend).date()
        print(f"  window {w_idx+1}: {wstart_d} → {cutoff_d} | test {tstart_d}–{tend_d}", flush=True)
        train_mask = (panel["Date"] >= wstart) & (panel["Date"] < cutoff)
        test_mask  = (panel["Date"] >= tstart) & (panel["Date"] <= tend)
        train_df   = panel[train_mask].dropna(subset=FEAT + [TARGET])
        test_df    = panel[test_mask].copy()

        X_tr = train_df[FEAT].values
        y_tr = train_df[TARGET].values
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(X_tr, y_tr)

        X_te = test_df[FEAT].values
        test_df = test_df[["Date", "ticker", TARGET, "fwd_ret_1"]].copy()
        test_df["har_pred"] = model.predict(X_te).clip(min=0)
        records.append(test_df)

    return pd.concat(records, ignore_index=True)


# ── Historical vol (vol_20) ──────────────────────────────────────────────────
def run_hist_vol(panel):
    """Trivial baseline: predict target_fwd_vol_5 = vol_20 (20-day trailing vol)."""
    dates = np.sort(panel["Date"].unique())
    records = []
    for wstart, cutoff, tstart, tend in walk_forward_windows(dates):
        test_mask = (panel["Date"] >= tstart) & (panel["Date"] <= tend)
        test_df = panel[test_mask][["Date", "ticker", TARGET, "fwd_ret_1", "vol_20"]].copy()
        test_df["hist_vol_pred"] = test_df["vol_20"]
        records.append(test_df.drop(columns=["vol_20"]))
    return pd.concat(records, ignore_index=True)


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading panel data...", flush=True)
    panel = pd.read_parquet(DATA_PATH)
    panel = panel.dropna(subset=["log_ret", "vol_5", "vol_20", "vol_60", TARGET])
    panel["Date"] = pd.to_datetime(panel["Date"])
    print(f"  {panel.shape[0]:,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['Date'].min().date()} – {panel['Date'].max().date()}", flush=True)

    # ── GARCH ──
    print("\n[1/3] Running GARCH(1,1)...", flush=True)
    garch_df = run_garch(panel)
    out = f"{RESULTS_DIR}/garch.parquet"
    garch_df.to_parquet(out, index=False)
    print(f"  saved → {out}  ({len(garch_df):,} rows)", flush=True)

    # ── HAR-RV ──
    print("\n[2/3] Running HAR-RV...", flush=True)
    har_df = run_har_rv(panel)
    out = f"{RESULTS_DIR}/har.parquet"
    har_df.to_parquet(out, index=False)
    print(f"  saved → {out}  ({len(har_df):,} rows)", flush=True)

    # ── Historical vol ──
    print("\n[3/3] Running Historical Vol (vol_20)...", flush=True)
    hv_df = run_hist_vol(panel)
    out = f"{RESULTS_DIR}/persistence.parquet"
    hv_df.to_parquet(out, index=False)
    print(f"  saved → {out}  ({len(hv_df):,} rows)", flush=True)

    print("\nDone.", flush=True)
