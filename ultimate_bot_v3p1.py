#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ultimate_bot_v4_4.py

Selective FX/Gold bot with strict confluence, session filters, and portfolio risk caps.
Fixes schema pattern error by using fnmatch only (no regex on user patterns).
Targets lower trade frequency, tighter risk, and cleaner position management.

CLI examples:
  py ultimate_bot_v4_4.py --backtest --symbols "EURUSD,GBPUSD,XAUUSD" --precision
  py ultimate_bot_v4_4.py --live-csv --symbols "EURUSD,GBPUSD" --precision
  py ultimate_bot_v4_4.py --live --symbols "EURUSD,GBPUSD" --gran M5 --dry-run True --precision
"""

import requests
import matplotlib.pyplot as plt
import os
import re
import glob
import json
import time
import argparse
import logging
import fnmatch
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ---- optional keys module for OANDA ----
try:
    import keys  # type: ignore
    _HAS_KEYS = True
except Exception:
    _HAS_KEYS = False

logger = logging.getLogger("Bot")


# ========================= MONTE CARLO UTILS =========================
def run_monte_carlo_from_trades(
    trades_df: pd.DataFrame,
    initial_capital: float,
    n_paths: int = 2000,
    path_len: Optional[int] = None,
    seed: int = 42,
    save_plot_path: Optional[str] = "equity_mc_quantiles.png",
) -> Optional[Dict]:
    """
    Monte Carlo on per-trade returns bootstrapped from backtest trades.

    - trades_df: DataFrame with at least a 'pnl' column and entry/exit times.
    - initial_capital: starting equity.
    - n_paths: number of synthetic equity paths.
    - path_len: number of trades per path (default = len(observed trade returns)).
    - seed: RNG seed.
    - save_plot_path: if provided, saves a quantile fan chart.

    Returns summary stats dict or None if MC cannot be run.
    """
    if trades_df is None or trades_df.empty:
        logger.warning("[MC] No trades for Monte Carlo.")
        return None

    df = trades_df.copy()

    # Try to sort chronologically by exit_time (fallback to entry_time if needed)
    for col in ("exit_time", "entry_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    sort_col = "exit_time" if "exit_time" in df.columns else (
        "entry_time" if "entry_time" in df.columns else None
    )
    if sort_col:
        df = df.sort_values(sort_col)

    if "pnl" not in df.columns:
        logger.warning("[MC] No 'pnl' column for Monte Carlo.")
        return None

    pnl = df["pnl"].astype(float).values
    if pnl.size == 0:
        logger.warning("[MC] No PnL values for Monte Carlo.")
        return None

    # Derive per-trade returns as pnl / capital_before
    capital = float(initial_capital)
    rets: List[float] = []
    for p in pnl:
        if capital <= 0:
            break
        r = p / capital
        rets.append(r)
        capital += p

    if not rets:
        logger.warning("[MC] Could not derive returns for Monte Carlo.")
        return None

    rets = np.array(rets, dtype=float)

    if path_len is None:
        path_len = len(rets)
    path_len = max(1, int(path_len))
    n_paths = max(1, int(n_paths))

    rng = np.random.default_rng(seed)
    eq_paths = np.empty((n_paths, path_len), dtype=float)
    final_eq = np.empty(n_paths, dtype=float)
    max_dd = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        cap = float(initial_capital)
        peak = cap
        local_max_dd = 0.0
        for t in range(path_len):
            r = float(rng.choice(rets))
            cap *= (1.0 + r)
            eq_paths[i, t] = cap
            peak = max(peak, cap)
            dd = 1.0 - cap / peak if peak > 0 else 0.0
            if dd > local_max_dd:
                local_max_dd = dd
        final_eq[i] = cap
        max_dd[i] = local_max_dd

    # Quantile fan
    q10 = np.percentile(eq_paths, 10, axis=0)
    q50 = np.percentile(eq_paths, 50, axis=0)
    q90 = np.percentile(eq_paths, 90, axis=0)

    if save_plot_path:
        try:
            plt.figure(figsize=(12, 6))
            x = np.arange(path_len)
            plt.plot(x, q50, label="Median path")
            plt.plot(x, q10, linestyle="--", label="P10 path")
            plt.plot(x, q90, linestyle="--", label="P90 path")
            plt.title("Monte Carlo Equity Paths (10/50/90 quantiles)")
            plt.xlabel("Trades")
            plt.ylabel("Equity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_plot_path)
            plt.close()
            logger.info(
                f"[MC] Saved Monte Carlo equity fan -> {save_plot_path}")
        except Exception as e:
            logger.warning(f"[MC] Failed to save Monte Carlo plot: {e}")

    stats = {
        "initial_capital": float(initial_capital),
        "n_paths": int(n_paths),
        "path_len": int(path_len),
        "median_ending_equity": float(np.median(final_eq)),
        "p10_ending_equity": float(np.percentile(final_eq, 10)),
        "p90_ending_equity": float(np.percentile(final_eq, 90)),
        "median_max_drawdown_pct": float(np.median(max_dd) * 100.0),
        "p90_max_drawdown_pct": float(np.percentile(max_dd, 90) * 100.0),
        "prob_ruin_70pct": float(np.mean(final_eq <= 0.7 * initial_capital)),
        "prob_ruin_50pct": float(np.mean(final_eq <= 0.5 * initial_capital)),
    }
    return stats


# ========================= CONFIG =========================
DEFAULT_CONFIG = {
    "data_dir": "./data",
    "instruments": ["EURUSD", "GBPUSD", "GBPJPY", "XAUUSD"],

    # capital & costs
    "initial_capital": 10000.0,
    "commission_per_trade": 0.0,
    "slippage_pct": 0.00010,     # ~1 pip majors

    # risk (tighter)
    "risk_per_trade": 0.003,     # 0.3%/trade
    "position_multiplier": 1.0,
    "min_units_default": 1000,
    "min_units_map": {"EURUSD": 10000, "GBPUSD": 10000, "GBPJPY": 1000, "XAUUSD": 10},
    "max_units_map": {"EURUSD": 40000, "GBPUSD": 40000, "GBPJPY": 8000, "XAUUSD": 50},

    # portfolio
    "max_open_trades": 2,
    "max_portfolio_risk": 0.03,  # 3% of equity across open risk

    # precision gate defaults (can tighten with --precision/--win-target)
    "precision_mode": True,
    "min_confidence": 0.55,
    "min_adx": 18,
    "max_adx": 55,
    "min_atr_std_ratio": 0.25,

    # exits
    "atr_stop_mult": 1.2,
    "tp_rr": 2.0,
    "breakeven_trigger_R": 0.5,
    "breakeven_offset_R": 0.05,
    "trailing_stop_mult": 1.2,
    "time_exit_bars": 100000,

    # runtime
    "dry_run": True,
    "oanda_practice": True,
    "oanda_api_url_practice": "https://api-fxpractice.oanda.com/v3",
    "oanda_api_url_live": "https://api-fxtrade.oanda.com/v3",
    "granularity": "M5",
    "poll_seconds": 5,

    # entry timing control
    "cooldown_bars": 12,
    "entry_rearm_bars": 12,

    # toggles
    "use_mmxm": True,
    "use_supertrend": True,

    # additional gates
    "session_start_hour": 6,      # UTC
    "session_end_hour": 20,       # UTC
    "min_confluence": 2,          # >= strategies agree
    "max_trades_per_day": 15,
}

STRAT_PARAMS = {
    "atr_period": 10,
    "donchian_period": 14, "breakout_k_atr": 0.0, "tb_rr": 2.0, "tb_sl_atr_mult": 1.2,
    "pb_lookback": 12, "pb_rr": 2.0, "pb_sl_atr_mult": 1.2,
    "mmxm_score_threshold": 0.3,
    "st_period": 10, "st_multiplier": 3.0,
    "boll_period": 20, "boll_dev": 2.0,
}


# ========================= LOGGING =========================
def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(
            "bot.log", encoding="utf-8"), logging.StreamHandler()],
    )


# ========================= DATA LOAD =========================
TF_TOKENS = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")


def _guess_dt_format(date_str: str, time_str: str) -> Optional[str]:
    ds, ts = str(date_str), str(time_str)
    if "." in ds:
        dfmt = "%Y.%m.%d"
    elif "-" in ds and "T" not in ds:
        dfmt = "%Y-%m-%d"
    elif "/" in ds:
        dfmt = "%Y/%m/%d"
    else:
        return None
    parts = ts.split(":")
    if len(parts) == 2:
        tfmt = "%H:%M"
    elif len(parts) == 3:
        tfmt = "%H:%M:%S"
    else:
        return None
    return f"{dfmt} {tfmt}"


def _clean_dt_index(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        cand = None
        for c in d.columns:
            if any(k in c.lower() for k in ["datetime", "timestamp", "time", "date"]):
                cand = c
                break
        if cand is None:
            raise ValueError("No datetime column")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d[cand] = pd.to_datetime(d[cand], utc=True, errors="coerce")
        d = d.set_index(cand)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx = pd.to_datetime(d.index, utc=True, errors="coerce")
    d = d.loc[~idx.isna()]
    d.index = pd.DatetimeIndex(idx[~idx.isna()])
    return d.sort_index()


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        cols = list(d.columns)
        dt_cols = [c for c in cols if any(
            k in c.lower() for k in ["datetime", "timestamp"])]
        if dt_cols:
            dc = dt_cols[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d[dc] = pd.to_datetime(d[dc], utc=True, errors="coerce")
            d = d.set_index(dc)
        else:
            dcols = [c for c in cols if "date" in c.lower()]
            tcols = [c for c in cols if "time" in c.lower()]
            if dcols and tcols and dcols[0] != tcols[0]:
                dc, tc = dcols[0], tcols[0]
                try:
                    fmt = _guess_dt_format(
                        str(d[dc].iloc[0]), str(d[tc].iloc[0]))
                except Exception:
                    fmt = None
                comb = d[dc].astype(str) + " " + d[tc].astype(str)
                if fmt:
                    dt = pd.to_datetime(
                        comb, format=fmt, utc=True, errors="coerce")
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dt = pd.to_datetime(comb, utc=True, errors="coerce")
                d = d.assign(__dt__=dt).set_index("__dt__")
            elif dcols:
                dc = dcols[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d[dc] = pd.to_datetime(d[dc], utc=True, errors="coerce")
                d = d.set_index(dc)
            elif tcols:
                tc = tcols[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d[tc] = pd.to_datetime(d[tc], utc=True, errors="coerce")
                d = d.set_index(tc)
            else:
                raise ValueError("No datetime column found")
    d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    rmap = {}
    for c in d.columns:
        cl = c.lower()
        if cl.startswith("open"):
            rmap[c] = "Open"
        elif cl.startswith("high"):
            rmap[c] = "High"
        elif cl.startswith("low"):
            rmap[c] = "Low"
        elif cl.startswith("close"):
            rmap[c] = "Close"
        elif cl.startswith("vol"):
            rmap[c] = "Volume"
    if rmap:
        d = d.rename(columns=rmap)
    keep = [c for c in ["Open", "High", "Low",
                        "Close", "Volume"] if c in d.columns]
    d = d[keep].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return d


def load_mt4_headerless(path: str, delim: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        if delim is None:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline()
            delim = ";" if first.count(";") > first.count(",") else ","
        df = pd.read_csv(
            path,
            header=None,
            sep=delim,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
            engine="python",
        )
        fmt = _guess_dt_format(
            str(df["Date"].iloc[0]), str(df["Time"].iloc[0]))
        comb = df["Date"].astype(str) + " " + df["Time"].astype(str)
        if fmt:
            dt = pd.to_datetime(comb, format=fmt, utc=True, errors="coerce")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(comb, utc=True, errors="coerce")
        df = df.assign(Datetime=dt).drop(columns=["Date", "Time"]).dropna()
        df = df.set_index("Datetime").sort_index()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if df.empty:
            return None
        logger.info(
            f"[PARSE] {os.path.basename(path)}: MT4 headerless OK ({len(df)} rows)")
        return df[
            ["Open", "High", "Low", "Close"]
            + (["Volume"] if "Volume" in df.columns else [])
        ]
    except Exception:
        return None


def infer_symbol_from_filename(path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(path))[0]
    clean = re.sub(r"[^A-Za-z]", "", base).upper()
    for tok in list(TF_TOKENS) + [
        "SYNTH",
        "SYNTHETIC",
        "CANDLE",
        "CANDLES",
        "DATA",
        "HIST",
        "OANDA",
    ]:
        clean = clean.replace(tok, "")
    known = [
        "EURUSD",
        "GBPUSD",
        "GBPJPY",
        "XAUUSD",
        "USDJPY",
        "AUDUSD",
        "US30",
        "NAS100",
        "SPX500",
        "XAGUSD",
    ]
    for k in known:
        if k in clean:
            return k
    m = re.search(r"([A-Z]{6})", clean)
    if m:
        return m.group(1)
    m2 = re.search(r"([A-Z]{3,5})([A-Z]{3,5})", clean)
    if m2:
        return (m2.group(1) + m2.group(2))[:10]
    return None


def _load_schema(schema_path: str) -> List[Tuple[str, Dict]]:
    out: List[Tuple[str, Dict]] = []
    if not os.path.isfile(schema_path):
        return out
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                for patt, cfg in obj.items():
                    out.append((str(patt), dict(cfg)))
                return out
        except Exception:
            pass
        blobs = re.findall(r"\{[\s\S]*?\}", txt)
        for blob in blobs:
            try:
                d = json.loads(blob)
                if isinstance(d, dict):
                    for patt, cfg in d.items():
                        out.append((str(patt), dict(cfg)))
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[SCHEMA] read fail: {e}")
    return out


def _apply_schema_to_path(
    path: str, schema: List[Tuple[str, Dict]]
) -> Tuple[Optional[str], Optional[str]]:
    base = os.path.basename(path)
    forced_sym, delim = None, None
    for patt, cfg in schema:
        try:
            if fnmatch.fnmatch(base, patt):
                forced_sym = cfg.get("symbol") or forced_sym
                delim = cfg.get("delimiter") or delim
        except Exception:
            continue
    return forced_sym, delim


def load_csv_file(path: str, schema: List[Tuple[str, Dict]]) -> Optional[Tuple[str, pd.DataFrame]]:
    forced_sym, forced_delim = _apply_schema_to_path(path, schema)
    sym = forced_sym or infer_symbol_from_filename(path)
    if not sym:
        return None
    try:
        df_try = pd.read_csv(path)
        try:
            df_fixed = ensure_datetime_index(df_try)
            if df_fixed is not None and not df_fixed.empty:
                logger.info(
                    f"[PARSE] {os.path.basename(path)}: headered OK ({len(df_fixed)} rows)"
                )
                return sym, df_fixed
        except Exception:
            pass
    except Exception:
        pass
    df_mt4 = load_mt4_headerless(path, forced_delim)
    if df_mt4 is not None and not df_mt4.empty:
        return sym, df_mt4
    logger.warning(f"[PARSE] Failed {path}")
    return None


def auto_discover_data(
    data_dir: str, schema: List[Tuple[str, Dict]], prefer_glob: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    data_map: Dict[str, pd.DataFrame] = {}
    base = os.path.abspath(data_dir)
    pats = ("*.csv", "*.CSV", "*.csv.csv", "*.CSV.CSV")
    cand: List[str] = []
    for p in pats:
        cand.extend(glob.glob(os.path.join(base, p)))
    if not cand:
        logger.warning(f"[DISCOVER] no CSVs in {base}")
        return {}
    logger.info(f"[DISCOVER] scanning {base} -> {len(cand)} files")
    best: Dict[str, Tuple[str, int, bool]] = {}
    for path in cand:
        res = load_csv_file(path, schema)
        if not res:
            continue
        sym, df = res
        rows = len(df)
        base_name = os.path.basename(path)
        prefers = bool(prefer_glob and fnmatch.fnmatch(base_name, prefer_glob))
        prev = best.get(sym)
        take = False
        if prev is None:
            take = True
        else:
            _, prev_rows, prev_pref = prev
            if prefers and not prev_pref:
                take = True
            elif prefers == prev_pref and rows > prev_rows:
                take = True
        if take:
            best[sym] = (path, rows, prefers)
            data_map[sym] = df
            tag = " [PREFERRED]" if prefers else ""
            logger.info(
                f"[DISCOVER] {sym}: -> {os.path.basename(path)} ({rows} rows){tag}"
            )
    if data_map:
        logger.info(f"[DISCOVER] Symbols: {sorted(data_map.keys())}")
    return data_map


def load_all_data(
    data_dir: str, symbols: List[str], prefer_glob: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    schema_path = os.path.join(os.path.abspath(data_dir), "csv_schema.json")
    schema = _load_schema(schema_path)
    if schema:
        logger.info(f"[SCHEMA] Using -> {schema_path}")
    discovered = auto_discover_data(data_dir, schema, prefer_glob)
    if discovered:
        return discovered
    data_map: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        cands = glob.glob(os.path.join(data_dir, f"*{sym}*.csv"))
        if not cands:
            logger.warning(f"No CSV for {sym}")
            continue
        best = max(cands, key=lambda p: os.path.getsize(p))
        res = load_csv_file(best, schema)
        if not res:
            continue
        _, df = res
        data_map[sym] = df
    if data_map:
        logger.info(f"[LOAD] Loaded: {sorted(data_map.keys())}")
    return data_map


# ========================= INDICATORS =========================
def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=max(2, int(span)), adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat(
        [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atrv = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * (plus_dm.ewm(alpha=1 / period,
                 adjust=False).mean() / (atrv + 1e-9))
    mdi = 100 * (minus_dm.ewm(alpha=1 / period,
                 adjust=False).mean() / (atrv + 1e-9))
    dx = ((pdi - mdi).abs() / (pdi + mdi + 1e-9)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-9)
    return 100 - (100 / (1 + rs))


# ========================= STRATEGIES =========================
class StrategyBase:
    def __init__(self, params: Dict):
        self.params = dict(params)

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        raise NotImplementedError


def donchian(df: pd.DataFrame, period: int = 20):
    up = df["High"].rolling(period, min_periods=period // 2).max()
    lo = df["Low"].rolling(period, min_periods=period // 2).min()
    return up, lo


class TrendBreakout(StrategyBase):
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        p = int(self.params.get("donchian_period", 14))
        if i < max(p, 220):
            return None
        close_i = df["Close"].iloc[i]
        high_i = df["High"].iloc[i]
        low_i = df["Low"].iloc[i]
        a = atr(df, int(self.params.get("atr_period", 10))).iloc[i]
        e200 = ema(df["Close"], 200)
        if pd.isna(e200.iloc[i]) or pd.isna(e200.iloc[i - 1]):
            return None
        slope = e200.diff().iloc[i]
        up, lo = donchian(df, p)
        band_up = up.iloc[i -
                          1] if not pd.isna(up.iloc[i - 1]) else up.iloc[:i].max()
        band_lo = lo.iloc[i -
                          1] if not pd.isna(lo.iloc[i - 1]) else lo.iloc[:i].min()
        if slope >= 0 and (high_i > band_up or close_i > band_up):
            sl = close_i - float(self.params.get("tb_sl_atr_mult", 1.2)) * a
            R = max(1e-9, close_i - sl)
            tp = close_i + float(self.params.get("tb_rr", 2.0)) * R
            return {
                "side": "buy",
                "confidence": 0.6,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "TB",
            }
        if slope <= 0 and (low_i < band_lo or close_i < band_lo):
            sl = close_i + float(self.params.get("tb_sl_atr_mult", 1.2)) * a
            R = max(1e-9, sl - close_i)
            tp = close_i - float(self.params.get("tb_rr", 2.0)) * R
            return {
                "side": "sell",
                "confidence": 0.6,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "TB",
            }
        return None


class MomentumEMA(StrategyBase):
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 220:
            return None
        c = df["Close"]
        e20 = ema(c, 20)
        e50 = ema(c, 50)
        e200 = ema(c, 200)
        if any(pd.isna(s.iloc[i]) for s in [e20, e50, e200]):
            return None
        slope200 = e200.diff().iloc[i]
        a = atr(df, int(self.params.get("atr_period", 10))).iloc[i]
        px = c.iloc[i]
        if e20.iloc[i] > e50.iloc[i] and slope200 >= 0:
            sl = px - 1.2 * a
            tp = px + 2.0 * (px - sl)
            return {
                "side": "buy",
                "confidence": 0.55,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "EMA",
            }
        if e20.iloc[i] < e50.iloc[i] and slope200 <= 0:
            sl = px + 1.2 * a
            tp = px - 2.0 * (sl - px)
            return {
                "side": "sell",
                "confidence": 0.55,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "EMA",
            }
        return None


class PullbackReentry(StrategyBase):
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        look = int(self.params.get("pb_lookback", 12))
        if i < max(look + 30, 60):
            return None
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        e20 = ema(c, 20)
        a = atr(df, int(self.params.get("atr_period", 10)))
        if any(pd.isna(s.iloc[i]) for s in [e20, a]):
            return None
        st = max(0, i - look + 1)
        pulled_long = any(l.iloc[j] <= (e20.iloc[j] - a.iloc[j])
                          for j in range(st, i + 1))
        pulled_short = any(h.iloc[j] >= (e20.iloc[j] + a.iloc[j])
                           for j in range(st, i + 1))
        cross_up = (c.iloc[i - 1] <= e20.iloc[i - 1]
                    ) and (c.iloc[i] > e20.iloc[i])
        cross_down = (c.iloc[i - 1] >= e20.iloc[i - 1]
                      ) and (c.iloc[i] < e20.iloc[i])
        px = c.iloc[i]
        if pulled_long and cross_up:
            sl = px - float(self.params.get("pb_sl_atr_mult", 1.2)) * a.iloc[i]
            R = max(1e-9, px - sl)
            tp = px + float(self.params.get("pb_rr", 2.0)) * R
            return {
                "side": "buy",
                "confidence": 0.50,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "PB",
            }
        if pulled_short and cross_down:
            sl = px + float(self.params.get("pb_sl_atr_mult", 1.2)) * a.iloc[i]
            R = max(1e-9, sl - px)
            tp = px - float(self.params.get("pb_rr", 2.0)) * R
            return {
                "side": "sell",
                "confidence": 0.50,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "PB",
            }
        return None


# Helpers for MMXMV2
def _to_lower_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    r = {}
    for c in d.columns:
        cl = c.lower()
        if cl.startswith("open"):
            r[c] = "open"
        elif cl.startswith("high"):
            r[c] = "high"
        elif cl.startswith("low"):
            r[c] = "low"
        elif cl.startswith("close"):
            r[c] = "close"
        elif cl.startswith("vol"):
            r[c] = "volume"
    d = d.rename(columns=r)
    return _clean_dt_index(d)[
        [c for c in ["open", "high", "low", "close", "volume"] if c in d.columns]
    ]


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    ohlc = {"open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum"}
    return df.resample(freq, label="right", closed="right").apply(ohlc).dropna()


def _fvg(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(False, index=df.index)
    for i in range(1, len(df) - 1):
        c0, c1 = df.iloc[i - 1], df.iloc[i]
        if (c0["close"] < c0["open"]) and (c1["close"] > c1["open"]) and (c1["low"] > c0["close"]):
            s.iloc[i] = True
    return s


class MMXMV2(StrategyBase):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.cache_len = -1
        self.sigs: Optional[pd.DataFrame] = None

    def _rebuild(self, df: pd.DataFrame):
        low = _to_lower_ohlcv(df)
        ex = _resample(low, "15min")
        ex["fvg"] = _fvg(ex)
        ctx = _resample(low, "1h")
        out = pd.DataFrame(index=low.index, data={"signal": 0, "score": 0.0})
        ex_idx = ex.index
        for t in ex_idx:
            pos = ex_idx.searchsorted(t, side="right")
            prev_t = (
                ex_idx[pos - 1]
                if pos > 0
                else pd.Timestamp.min.tz_localize("UTC")
            )
            if bool(ex.loc[t, "fvg"]):
                mask = (low.index > prev_t) & (low.index <= t)
                out.loc[mask, "score"] += 1.0
        ctx_idx = ctx.index
        for t in out.index:
            j = ctx_idx.searchsorted(t, side="right") - 1
            bias = 0
            if j >= 0:
                ht = ctx_idx[j]
                bias = 1 if low.loc[t, "close"] < ctx.loc[ht, "close"] else -1
            sc = float(out.at[t, "score"])
            thr = float(self.params.get("mmxm_score_threshold", 0.3))
            if sc >= thr and bias == 1:
                out.at[t, "signal"] = 1
            elif sc >= thr and bias == -1:
                out.at[t, "signal"] = -1
        self.sigs = out
        self.cache_len = len(df)

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 250 or not DEFAULT_CONFIG.get("use_mmxm", True):
            return None
        if self.sigs is None or len(df) != self.cache_len:
            self._rebuild(df)
        ts = df.index[i]
        if ts not in self.sigs.index:
            return None
        sig = int(self.sigs.at[ts, "signal"])
        if sig == 0:
            return None
        side = "buy" if sig == 1 else "sell"
        a = atr(df, int(self.params.get("atr_period", 10))).iloc[i]
        px = df["Close"].iloc[i]
        sl = px - 1.2 * a if side == "buy" else px + 1.2 * a
        R = abs(px - sl)
        tp = px + 2.0 * R if side == "buy" else px - 2.0 * R
        return {
            "side": side,
            "confidence": 0.5,
            "sl": float(sl),
            "tp": float(tp),
            "tag": "MMXM",
        }


class SuperTrendStrategy(StrategyBase):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.st_cache: Optional[Tuple[pd.Series, pd.Series]] = None

    def _calc_supertrend(self, df: pd.DataFrame):
        period = int(self.params.get("st_period", 10))
        mult = float(self.params.get("st_multiplier", 3.0))
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat(
            [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
        hl2 = (h + l) / 2.0
        upper_band = (hl2 + mult * atr_val).dropna()
        lower_band = (hl2 - mult * atr_val).dropna()
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()
        for i in range(1, len(df)):
            if c.iloc[i - 1] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = max(
                    upper_band.iloc[i], final_upper.iloc[i - 1])
            else:
                final_upper.iloc[i] = upper_band.iloc[i]
            if c.iloc[i - 1] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = min(
                    lower_band.iloc[i], final_lower.iloc[i - 1])
            else:
                final_lower.iloc[i] = lower_band.iloc[i]
        st = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=bool)
        st.iloc[0] = 0.0
        direction.iloc[0] = True
        for i in range(1, len(df)):
            if c.iloc[i] > final_lower.iloc[i - 1]:
                direction.iloc[i] = True
                st.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = False
                st.iloc[i] = final_upper.iloc[i]
        return st, direction

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 100:
            return None
        if self.st_cache is None or len(self.st_cache[0]) != len(df):
            self.st_cache = self._calc_supertrend(df)
        st_line, direction = self.st_cache
        prev_dir = direction.iloc[i - 1]
        curr_dir = direction.iloc[i]
        close_i = df["Close"].iloc[i]
        a = atr(df, int(self.params.get("atr_period", 10))).iloc[i]
        if not prev_dir and curr_dir:
            e200 = ema(df["Close"], 200)
            if pd.isna(e200.iloc[i]) or close_i < e200.iloc[i]:
                return None
            rsi_val = rsi(df["Close"], 14).iloc[i]
            if rsi_val >= 70:
                return None
            sl = close_i - 1.2 * a
            tp = close_i + 2.0 * (close_i - sl)
            return {
                "side": "buy",
                "confidence": 0.6,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "ST",
            }
        if prev_dir and not curr_dir:
            e200 = ema(df["Close"], 200)
            if pd.isna(e200.iloc[i]) or close_i > e200.iloc[i]:
                return None
            rsi_val = rsi(df["Close"], 14).iloc[i]
            if rsi_val <= 30:
                return None
            sl = close_i + 1.2 * a
            tp = close_i - 2.0 * (sl - close_i)
            return {
                "side": "sell",
                "confidence": 0.6,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "ST",
            }
        return None


class RangeReversion(StrategyBase):
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        period = int(self.params.get("boll_period", 20))
        if i < max(period, 220):
            return None
        adx_v = adx(df, 14).iloc[i]
        if adx_v >= 18:
            return None
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        e200 = ema(c, 200)
        roll = c.iloc[i - period + 1: i + 1]
        mid = roll.mean()
        std = roll.std()
        if pd.isna(std) or std == 0:
            return None
        upper, lower = mid + 2.0 * std, mid - 2.0 * std
        a = atr(df, 10).iloc[i]
        r = rsi(c, 14).iloc[i]
        px = c.iloc[i]
        if px > upper and r > 60 and px < e200.iloc[i]:
            sl = h.iloc[i] + 1.0 * a
            tp = mid
            return {
                "side": "sell",
                "confidence": 0.55,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "RR",
            }
        if px < lower and r < 40 and px > e200.iloc[i]:
            sl = l.iloc[i] - 1.0 * a
            tp = mid
            return {
                "side": "buy",
                "confidence": 0.55,
                "sl": float(sl),
                "tp": float(tp),
                "tag": "RR",
            }
        return None


# ========================= ENSEMBLE WITH CONFLUENCE =========================
class Ensemble:
    def __init__(self, strategies: List[StrategyBase], weights: Optional[List[float]] = None):
        self.strats = strategies
        self.w = list(weights) if weights else [1.0] * len(strategies)

    def aggregate(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        adx_v = adx(df, 14).iloc[i] if i >= 14 else 0
        regime = "trend" if adx_v >= 25 else (
            "range" if adx_v <= 20 else "neutral")
        votes: List[Dict] = []
        for strat, w in zip(self.strats, self.w):
            name = strat.__class__.__name__
            if regime == "trend" and name in ["RangeReversion"]:
                continue
            if regime == "range" and name in [
                "TrendBreakout",
                "MomentumEMA",
                "PullbackReentry",
                "MMXMV2",
                "SuperTrendStrategy",
            ]:
                continue
            sig = None
            try:
                sig = strat.generate_signal(df, i)
            except Exception as e:
                logger.debug(f"Strat error {name}: {e}")
            if sig:
                sig = dict(sig)
                sig["weight"] = w
                votes.append(sig)
        if not votes:
            return None
        buy = [v for v in votes if v["side"] == "buy"]
        sell = [v for v in votes if v["side"] == "sell"]
        side = (
            "buy"
            if sum(v["confidence"] * v["weight"] for v in buy)
            > sum(v["confidence"] * v["weight"] for v in sell)
            else "sell"
        )
        sel = buy if side == "buy" else sell
        if len(sel) < int(DEFAULT_CONFIG["min_confluence"]):
            return None
        conf = float(np.median([v["confidence"] for v in sel]))
        sls = [v["sl"] for v in sel]
        tps = [v["tp"] for v in sel]
        if not sls or not tps:
            return None
        return {
            "side": side,
            "confidence": conf,
            "sl": float(np.median(sls)),
            "tp": float(np.median(tps)),
            "tags": list({v.get("tag", "") for v in sel if v.get("tag")}),
        }


# ========================= GATES & SIZING =========================
def precision_gate(df: pd.DataFrame, i: int, side: str, conf: float, cfg: dict):
    if not cfg.get("precision_mode", False):
        return True, ["precision_off"]
    reasons: List[str] = []
    if conf < float(cfg.get("min_confidence", 0.55)):
        reasons.append(f"low_conf {conf:.2f}")
    try:
        adxv = adx(df, 14).iloc[i]
        if adxv < float(cfg.get("min_adx", 18)):
            reasons.append(f"low_adx {adxv:.1f}")
        if adxv > float(cfg.get("max_adx", 55)):
            reasons.append(f"high_adx {adxv:.1f}")
    except Exception:
        pass
    try:
        a = atr(df, 14).iloc[i]
        stdev = df["Close"].rolling(100, min_periods=50).std().iloc[i]
        if stdev > 0 and a / (stdev + 1e-9) < float(cfg.get("min_atr_std_ratio", 0.25)):
            reasons.append("low_atr_std_ratio")
    except Exception:
        pass
    e200 = ema(df["Close"], 200)
    if i >= 200 and not any(pd.isna(e200.iloc[i - 1: i + 1])):
        slope = e200.diff().iloc[i]
        if side == "buy" and slope <= 0:
            reasons.append("slope_down")
        if side == "sell" and slope >= 0:
            reasons.append("slope_up")
    ok = len(reasons) == 0
    return ok, (["pass"] if ok else reasons)


def compute_units(equity: float, risk_per_trade: float, entry: float, stop: float) -> float:
    dist_frac = abs(entry - stop) / max(1e-12, entry)
    dist_frac = max(dist_frac, 1e-5)
    risk_amt = max(0.0, equity * risk_per_trade)
    notional = risk_amt / dist_frac
    return max(0.0, notional / max(1e-12, entry))


def min_units_for(sym: str, cfg: dict) -> int:
    return int(cfg.get("min_units_map", {}).get(sym.upper(), cfg.get("min_units_default", 1)))


def max_units_for(sym: str, cfg: dict) -> int:
    return int(cfg.get("max_units_map", {}).get(sym.upper(), 10**12))


def session_ok(ts: pd.Timestamp, cfg: dict) -> bool:
    h = ts.hour
    return int(cfg["session_start_hour"]) <= h <= int(cfg["session_end_hour"])


# ========================= OANDA =========================
def oanda_symbol(sym: str) -> str:
    s = sym.upper().replace("/", "").replace("_", "")
    if s == "XAUUSD":
        return "XAU_USD"
    if s == "XAGUSD":
        return "XAG_USD"
    if s == "US30":
        return "US30_USD"
    if s == "SPX500":
        return "SPX500_USD"
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


def get_price_precision(instrument: str) -> int:
    if "JPY" in instrument:
        return 3
    if "XAU" in instrument or "XAG" in instrument:
        return 2
    if "US30" in instrument or "SPX500" in instrument:
        return 1
    return 5


class OandaClient:
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.base = (
            DEFAULT_CONFIG["oanda_api_url_practice"]
            if practice
            else DEFAULT_CONFIG["oanda_api_url_live"]
        )
        self.s = requests.Session()
        self.s.headers.update(
            {"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"}
        )
        self.account_id = account_id

    def get_candles(
        self, instrument: str, granularity: str = "M5", count: int = 600
    ) -> pd.DataFrame:
        url = f"{self.base}/instruments/{instrument}/candles"
        r = self.s.get(
            url,
            params={"count": int(count), "price": "M",
                    "granularity": granularity},
            timeout=30,
        )
        r.raise_for_status()
        rows = []
        for c in r.json().get("candles", []):
            if not c.get("complete", True):
                continue
            rows.append(
                {
                    "Datetime": c["time"],
                    "Open": float(c["mid"]["o"]),
                    "High": float(c["mid"]["h"]),
                    "Low": float(c["mid"]["l"]),
                    "Close": float(c["mid"]["c"]),
                    "Volume": int(c.get("volume", 0)),
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.loc[~dt.isna()].drop(columns=["Datetime"])
        df.index = pd.DatetimeIndex(dt[~dt.isna()])
        return df.sort_index()[["Open", "High", "Low", "Close", "Volume"]]

    def place_market(
        self,
        instrument: str,
        units: int,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ):
        url = f"{self.base}/accounts/{self.account_id}/orders"
        prec = get_price_precision(instrument)
        data: Dict = {
            "order": {
                "instrument": instrument,
                "units": str(int(units)),
                "type": "MARKET",
                "positionFill": "DEFAULT",
            }
        }
        if tp is not None:
            data["order"]["takeProfitOnFill"] = {"price": str(round(tp, prec))}
        if sl is not None:
            data["order"]["stopLossOnFill"] = {"price": str(round(sl, prec))}
        r = self.s.post(url, json=data, timeout=30)
        r.raise_for_status()
        return r.json()


# ========================= RUNNERS =========================
class Cooldown:
    def __init__(self, cooldown_bars: int, rearm_bars: int):
        self.cool = cooldown_bars
        self.rearm = rearm_bars
        self.last_entry: Dict[str, int] = {}
        self.last_exit: Dict[str, int] = {}

    def can_enter(self, sym: str, i: int) -> bool:
        le = self.last_entry.get(sym, -10**9)
        lx = self.last_exit.get(sym, -10**9)
        return ((i - le) >= self.rearm) and ((i - max(le, lx)) >= self.cool)

    def mark_entry(self, sym: str, i: int):
        self.last_entry[sym] = i

    def mark_exit(self, sym: str, i: int):
        self.last_exit[sym] = i


class LiveCSVRunner:
    def __init__(self, cfg: dict, params: dict):
        self.cfg = cfg
        self.params = params
        self.capital = float(cfg.get("initial_capital", 10000.0))
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_ts: List[Tuple[pd.Timestamp, float]] = []
        self.cool = Cooldown(
            int(cfg.get("cooldown_bars", 0)), int(
                cfg.get("entry_rearm_bars", 0))
        )
        self.risk_usage = 0.0
        self.daily_counts: Dict[Tuple[str, pd.Timestamp.date], int] = {}

    def _ensemble(self) -> Ensemble:
        strats: List[StrategyBase] = [
            TrendBreakout(STRAT_PARAMS),
            MomentumEMA(STRAT_PARAMS),
            PullbackReentry(STRAT_PARAMS),
        ]
        if self.cfg.get("use_mmxm", True):
            strats.append(MMXMV2(STRAT_PARAMS))
        if self.cfg.get("use_supertrend", True):
            strats.append(SuperTrendStrategy(STRAT_PARAMS))
        strats.append(RangeReversion(STRAT_PARAMS))
        weights: List[float] = [1.0, 1.0, 0.9]
        if self.cfg.get("use_mmxm", True):
            weights.append(0.6)
        if self.cfg.get("use_supertrend", True):
            weights.append(0.8)
        weights.append(0.7)
        return Ensemble(strats, weights)

    def _inc_day(self, sym: str, ts: pd.Timestamp):
        key = (sym, ts.date())
        self.daily_counts[key] = self.daily_counts.get(key, 0) + 1

    def _left_today(self, sym: str, ts: pd.Timestamp) -> int:
        return int(self.cfg.get("max_trades_per_day", 999)) - self.daily_counts.get(
            (sym, ts.date()), 0
        )

    def _manage(self, sym: str, df: pd.DataFrame, i: int, price: float):
        pos = self.positions.get(sym)
        if not pos:
            return
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        exit_px: Optional[float] = None
        reason: Optional[str] = None
        if pos["side"] == "buy":
            if low <= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            elif high >= pos["tp"]:
                exit_px, reason = pos["tp"], "TP"
        else:
            if high >= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            elif low <= pos["tp"]:
                exit_px, reason = pos["tp"], "TP"
        if exit_px is None:
            R0 = float(pos.get("R0", 0.0))
            if R0 > 0:
                trigger_R = float(self.cfg.get("breakeven_trigger_R", 0.5))
                offset_R = float(self.cfg.get("breakeven_offset_R", 0.05))
                favorable = (
                    price - pos["entry"]
                    if pos["side"] == "buy"
                    else (pos["entry"] - price)
                )
                if (not pos.get("be_active", False)) and favorable >= trigger_R * R0:
                    half_size = pos["size"] * 0.5
                    pnl_half = (
                        (price - pos["entry"]) * half_size
                        if pos["side"] == "buy"
                        else (pos["entry"] - price) * half_size
                    )
                    self.capital += pnl_half
                    pos["size"] -= half_size
                    pos["be_active"] = True
                    pos["partial"] = True
                    if "risk_frac" in pos:
                        self.risk_usage = max(
                            0.0, self.risk_usage - 0.5 * pos["risk_frac"]
                        )
                        pos["risk_frac"] *= 0.5
                    be_sl = (
                        pos["entry"] + offset_R * R0
                        if pos["side"] == "buy"
                        else pos["entry"] - offset_R * R0
                    )
                    if pos["side"] == "buy":
                        pos["sl"] = max(pos["sl"], be_sl)
                    else:
                        pos["sl"] = min(pos["sl"], be_sl)
                    self.trades.append(
                        {
                            "symbol": sym,
                            "side": pos["side"],
                            "entry_time": df.index[pos["entry_idx"]].isoformat(),
                            "exit_time": df.index[i].isoformat(),
                            "entry": pos["entry"],
                            "exit": price,
                            "size": half_size,
                            "pnl": pnl_half,
                            "reason": "PARTIAL",
                        }
                    )
                a = atr(df, int(STRAT_PARAMS.get("atr_period", 10))).iloc[i]
                offset = float(self.cfg.get("trailing_stop_mult", 1.2))
                be_sl = None
                if pos.get("be_active", False):
                    be_sl = (
                        pos["entry"] + offset_R * R0
                        if pos["side"] == "buy"
                        else pos["entry"] - offset_R * R0
                    )
                if pos["side"] == "buy":
                    trail = price - offset * a
                    new_sl = max(pos["sl"], trail)
                    if be_sl is not None:
                        new_sl = max(new_sl, be_sl)
                    pos["sl"] = new_sl
                else:
                    trail = price + offset * a
                    new_sl = min(pos["sl"], trail)
                    if be_sl is not None:
                        new_sl = min(new_sl, be_sl)
                    pos["sl"] = new_sl
            if (i - pos["entry_idx"]) >= int(self.cfg.get("time_exit_bars", 100000)):
                exit_px, reason = price, "TIME"
        if exit_px is not None:
            pnl = (
                (exit_px - pos["entry"]) * pos["size"]
                if pos["side"] == "buy"
                else (pos["entry"] - exit_px) * pos["size"]
            )
            pnl -= float(self.cfg.get("commission_per_trade", 0.0))
            self.capital += pnl
            self.trades.append(
                {
                    "symbol": sym,
                    "side": pos["side"],
                    "entry_time": df.index[pos["entry_idx"]].isoformat(),
                    "exit_time": df.index[i].isoformat(),
                    "entry": pos["entry"],
                    "exit": exit_px,
                    "size": pos["size"],
                    "pnl": pnl,
                    "reason": reason,
                }
            )
            self.cool.mark_exit(sym, i)
            self.risk_usage -= pos.get("risk_frac", 0.0)
            if self.risk_usage < 0:
                self.risk_usage = 0.0
            self.positions.pop(sym, None)

    def run(self, symbols: List[str], data_map: Dict[str, pd.DataFrame]):
        usable = [s for s in symbols if s in data_map]
        if not usable:
            logger.error("[CSV] No symbols loaded.")
            return
        min_len = min(len(data_map[s]) for s in usable)
        if min_len < 300:
            logger.error("[CSV] Not enough bars.")
            return
        ens_dict = {s: self._ensemble() for s in usable}
        warmup = 250
        logger.info(f"[CSV] Live simulation on {usable}")
        for i in range(warmup, min_len):
            ts = max(data_map[s].index[i] for s in usable)
            for sym in list(self.positions.keys()):
                price = data_map[sym]["Close"].iloc[i]
                self._manage(sym, data_map[sym], i, price)
            for sym in usable:
                if sym in self.positions:
                    continue
                if not session_ok(ts, self.cfg):
                    continue
                if not self.cool.can_enter(sym, i):
                    continue
                if self._left_today(sym, ts) <= 0:
                    continue
                if self.risk_usage >= float(self.cfg.get("max_portfolio_risk", 1.0)):
                    continue
                df = data_map[sym]
                price = df["Close"].iloc[i]
                sig = ens_dict[sym].aggregate(df, i)
                if not sig:
                    continue
                ok, _ = precision_gate(
                    df, i, sig["side"], sig["confidence"], self.cfg
                )
                if not ok:
                    continue
                entry_price = price * (
                    1
                    + float(self.cfg.get("slippage_pct", 0.0))
                    * (1 if sig["side"] == "buy" else -1)
                )
                use_risk = float(self.cfg.get("risk_per_trade", 0.003))
                units_f = compute_units(
                    self.capital, use_risk, entry_price, float(sig["sl"])
                )
                units_f *= float(self.cfg.get("position_multiplier", 1.0))
                units_i = int(round(abs(units_f)))
                units_i = max(
                    min_units_for(sym, self.cfg),
                    min(units_i, max_units_for(sym, self.cfg)),
                )
                if units_i <= 0:
                    continue
                risk_amt = abs(entry_price - sig["sl"]) * units_i
                risk_frac = risk_amt / max(1e-9, self.capital)
                if self.risk_usage + risk_frac > float(
                    self.cfg.get("max_portfolio_risk", 1.0)
                ):
                    continue
                self.positions[sym] = {
                    "side": sig["side"],
                    "entry": float(entry_price),
                    "size": float(units_i),
                    "sl": float(sig["sl"]),
                    "tp": float(sig["tp"]),
                    "entry_idx": i,
                    "risk_frac": risk_frac,
                    "R0": abs(float(entry_price) - float(sig["sl"])),
                    "be_active": False,
                    "partial": False,
                }
                self.risk_usage += risk_frac
                self.cool.mark_entry(sym, i)
                self._inc_day(sym, ts)
                logger.info(
                    f"[ENTRY] {sym} {sig['side']} units={units_i} "
                    f"px={entry_price:.5f} sl={sig['sl']:.5f} tp={sig['tp']:.5f} "
                    f"conf={sig['confidence']:.2f} tags={sig.get('tags')}"
                )
            self.equity_ts.append((ts, float(self.capital)))
        last_i = min_len - 1
        for sym, pos in list(self.positions.items()):
            df = data_map[sym]
            price = df["Close"].iloc[last_i]
            pnl = (
                (price - pos["entry"]) * pos["size"]
                if pos["side"] == "buy"
                else (pos["entry"] - price) * pos["size"]
            )
            pnl -= float(self.cfg.get("commission_per_trade", 0.0))
            self.capital += pnl
            self.trades.append(
                {
                    "symbol": sym,
                    "side": pos["side"],
                    "entry_time": df.index[pos["entry_idx"]].isoformat(),
                    "exit_time": df.index[last_i].isoformat(),
                    "entry": pos["entry"],
                    "exit": price,
                    "size": pos["size"],
                    "pnl": pnl,
                    "reason": "EOD",
                }
            )
        if self.trades:
            pd.DataFrame(self.trades).to_csv("trades_live.csv", index=False)
        if self.equity_ts:
            eq_ser = pd.Series({t: e for t, e in self.equity_ts}).sort_index()
            plt.figure(figsize=(12, 6))
            eq_ser.plot()
            plt.title("Equity Curve")
            plt.tight_layout()
            plt.savefig("equity_live.png")
        logger.info(
            f"[CSV] Done. Final equity {self.capital:.2f}, trades={len(self.trades)}"
        )


class Backtester:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.capital = float(cfg.get("initial_capital", 10000.0))
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_counts: Dict[Tuple[str, pd.Timestamp.date], int] = {}

    def _ensemble(self) -> Ensemble:
        strats: List[StrategyBase] = [
            TrendBreakout(STRAT_PARAMS),
            MomentumEMA(STRAT_PARAMS),
            PullbackReentry(STRAT_PARAMS),
        ]
        if self.cfg.get("use_mmxm", True):
            strats.append(MMXMV2(STRAT_PARAMS))
        if self.cfg.get("use_supertrend", True):
            strats.append(SuperTrendStrategy(STRAT_PARAMS))
        strats.append(RangeReversion(STRAT_PARAMS))
        weights: List[float] = [1.0, 1.0, 0.9]
        if self.cfg.get("use_mmxm", True):
            weights.append(0.6)
        if self.cfg.get("use_supertrend", True):
            weights.append(0.8)
        weights.append(0.7)
        return Ensemble(strats, weights)

    def _inc_day(self, sym: str, ts: pd.Timestamp):
        key = (sym, ts.date())
        self.daily_counts[key] = self.daily_counts.get(key, 0) + 1

    def _left_today(self, sym: str, ts: pd.Timestamp) -> int:
        return int(self.cfg.get("max_trades_per_day", 999)) - self.daily_counts.get(
            (sym, ts.date()), 0
        )

    def run(
        self,
        symbols: List[str],
        data_map: Dict[str, pd.DataFrame],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        if start or end:
            s = pd.to_datetime(start, utc=True) if start else None
            e = pd.to_datetime(end, utc=True) if end else None
            for k in list(data_map.keys()):
                df = data_map[k]
                if s is not None:
                    df = df[df.index >= s]
                if e is not None:
                    df = df[df.index <= e]
                data_map[k] = df
        usable = [s for s in symbols if s in data_map and len(
            data_map[s]) > 300]
        if not usable:
            logger.error("[BT] No usable symbols")
            return
        ens_dict = {s: self._ensemble() for s in usable}
        min_len = min(len(data_map[s]) for s in usable)
        start_i = 250
        end_i = min_len - 1
        if end_i <= start_i:
            logger.error("[BT] Not enough bars")
            return
        logger.info(
            f"[BT] Backtesting on {usable} | bars={end_i - start_i + 1}")
        for i in range(start_i, end_i + 1):
            ts = max(data_map[s].index[i] for s in usable)
            # manage
            for sym, pos in list(self.positions.items()):
                df = data_map[sym]
                high = df["High"].iloc[i]
                low = df["Low"].iloc[i]
                price = df["Close"].iloc[i]
                exit_px: Optional[float] = None
                reason: Optional[str] = None
                if pos["side"] == "buy":
                    if low <= pos["sl"]:
                        exit_px, reason = pos["sl"], "SL"
                    elif high >= pos["tp"]:
                        exit_px, reason = pos["tp"], "TP"
                else:
                    if high >= pos["sl"]:
                        exit_px, reason = pos["sl"], "SL"
                    elif low <= pos["tp"]:
                        exit_px, reason = pos["tp"], "TP"
                if exit_px is None:
                    R0 = float(pos.get("R0", 0.0))
                    if R0 > 0:
                        trigger_R = float(self.cfg.get(
                            "breakeven_trigger_R", 0.5))
                        offset_R = float(self.cfg.get(
                            "breakeven_offset_R", 0.05))
                        favorable = (
                            price - pos["entry"]
                            if pos["side"] == "buy"
                            else (pos["entry"] - price)
                        )
                        if (not pos.get("be_active", False)) and favorable >= trigger_R * R0:
                            half_size = pos["size"] * 0.5
                            pnl_half = (
                                (price - pos["entry"]) * half_size
                                if pos["side"] == "buy"
                                else (pos["entry"] - price) * half_size
                            )
                            self.capital += pnl_half - float(
                                self.cfg.get("commission_per_trade", 0.0)
                            )
                            pos["size"] -= half_size
                            pos["partial"] = True
                            pos["be_active"] = True
                            be_sl = (
                                pos["entry"] + offset_R * R0
                                if pos["side"] == "buy"
                                else pos["entry"] - offset_R * R0
                            )
                            if pos["side"] == "buy":
                                pos["sl"] = max(pos["sl"], be_sl)
                            else:
                                pos["sl"] = min(pos["sl"], be_sl)
                            self.trades.append(
                                {
                                    "symbol": sym,
                                    "side": pos["side"],
                                    "entry_time": pos["entry_time"].isoformat(),
                                    "exit_time": data_map[sym].index[i].isoformat(),
                                    "entry": pos["entry"],
                                    "exit": price,
                                    "size": half_size,
                                    "pnl": pnl_half,
                                    "reason": "PARTIAL",
                                }
                            )
                        a = atr(df, int(STRAT_PARAMS.get(
                            "atr_period", 10))).iloc[i]
                        offset = float(self.cfg.get("trailing_stop_mult", 1.2))
                        be_sl = None
                        if pos.get("be_active", False):
                            be_sl = (
                                pos["entry"] + offset_R * R0
                                if pos["side"] == "buy"
                                else pos["entry"] - offset_R * R0
                            )
                        if pos["side"] == "buy":
                            trail = price - offset * a
                            new_sl = max(pos["sl"], trail)
                            if be_sl is not None:
                                new_sl = max(new_sl, be_sl)
                            pos["sl"] = new_sl
                        else:
                            trail = price + offset * a
                            new_sl = min(pos["sl"], trail)
                            if be_sl is not None:
                                new_sl = min(new_sl, be_sl)
                            pos["sl"] = new_sl
                if exit_px is not None:
                    pnl = (
                        (exit_px - pos["entry"]) * pos["size"]
                        if pos["side"] == "buy"
                        else (pos["entry"] - exit_px) * pos["size"]
                    )
                    self.capital += pnl - float(
                        self.cfg.get("commission_per_trade", 0.0)
                    )
                    self.trades.append(
                        {
                            "symbol": sym,
                            "side": pos["side"],
                            "entry_time": pos["entry_time"].isoformat(),
                            "exit_time": data_map[sym].index[i].isoformat(),
                            "entry": pos["entry"],
                            "exit": exit_px,
                            "size": pos["size"],
                            "pnl": pnl,
                            "reason": reason,
                        }
                    )
                    self.positions.pop(sym, None)
            # entries
            for sym in usable:
                if sym in self.positions:
                    continue
                if not session_ok(ts, self.cfg):
                    continue
                if self._left_today(sym, ts) <= 0:
                    continue
                if len(self.positions) >= int(self.cfg.get("max_open_trades", 1e9)):
                    continue
                df = data_map[sym]
                price = df["Close"].iloc[i]
                sig = ens_dict[sym].aggregate(df, i)
                if not sig:
                    continue
                ok, _ = precision_gate(
                    df, i, sig["side"], sig["confidence"], self.cfg)
                if not ok:
                    continue
                entry_price = price
                use_risk = float(self.cfg.get("risk_per_trade", 0.003))
                units_f = compute_units(
                    self.capital, use_risk, entry_price, float(sig["sl"])
                )
                units_f *= float(self.cfg.get("position_multiplier", 1.0))
                units_i = int(round(abs(units_f)))
                units_i = max(
                    min_units_for(sym, self.cfg),
                    min(units_i, max_units_for(sym, self.cfg)),
                )
                if units_i == 0:
                    continue
                risk_amt = abs(entry_price - sig["sl"]) * units_i
                risk_frac = risk_amt / max(1e-9, self.capital)
                open_risk = sum(p.get("risk_frac", 0.0)
                                for p in self.positions.values())
                if open_risk + risk_frac > float(
                    self.cfg.get("max_portfolio_risk", 1.0)
                ):
                    continue
                self.positions[sym] = {
                    "side": sig["side"],
                    "entry": float(entry_price),
                    "size": float(units_i),
                    "sl": float(sig["sl"]),
                    "tp": float(sig["tp"]),
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "risk_frac": risk_frac,
                    "R0": abs(float(entry_price) - float(sig["sl"])),
                    "be_active": False,
                    "partial": False,
                }
                self._inc_day(sym, ts)
            self.equity_curve.append((ts, float(self.capital)))

        final_i = end_i
        for sym, pos in list(self.positions.items()):
            df = data_map[sym]
            price = df["Close"].iloc[final_i]
            pnl = (
                (price - pos["entry"]) * pos["size"]
                if pos["side"] == "buy"
                else (pos["entry"] - price) * pos["size"]
            )
            self.capital += pnl - \
                float(self.cfg.get("commission_per_trade", 0.0))
            self.trades.append(
                {
                    "symbol": sym,
                    "side": pos["side"],
                    "entry_time": pos["entry_time"].isoformat(),
                    "exit_time": df.index[final_i].isoformat(),
                    "entry": pos["entry"],
                    "exit": price,
                    "size": pos["size"],
                    "pnl": pnl,
                    "reason": "EOD",
                }
            )

        trades_df: Optional[pd.DataFrame] = None
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv("trades_backtest.csv", index=False)

        if self.equity_curve:
            eq = pd.Series({t: e for t, e in self.equity_curve}).sort_index()
            plt.figure(figsize=(12, 6))
            eq.plot()
            plt.title("Equity Curve")
            plt.tight_layout()
            plt.savefig("equity_curve.png")
            logger.info("Saved equity -> equity_curve.png")

        # Monte Carlo on backtest trades
        if trades_df is not None:
            try:
                mc_stats = run_monte_carlo_from_trades(
                    trades_df,
                    float(self.cfg.get("initial_capital", 10000.0)),
                    n_paths=int(self.cfg.get("mc_paths", 2000))
                    if "mc_paths" in self.cfg
                    else 2000,
                    path_len=None,
                    seed=42,
                    save_plot_path="equity_mc_quantiles.png",
                )
                if mc_stats is not None:
                    with open("mc_results.json", "w", encoding="utf-8") as f:
                        json.dump(mc_stats, f, indent=2)
                    logger.info(
                        "[MC] Monte Carlo summary: "
                        f"median_end={mc_stats['median_ending_equity']:.2f}, "
                        f"p10_end={mc_stats['p10_ending_equity']:.2f}, "
                        f"p90_end={mc_stats['p90_ending_equity']:.2f}, "
                        f"median_maxDD={mc_stats['median_max_drawdown_pct']:.1f}%, "
                        f"p90_maxDD={mc_stats['p90_max_drawdown_pct']:.1f}%"
                    )
            except Exception as e:
                logger.warning(f"[MC] Monte Carlo simulation failed: {e}")

        logger.info(
            f"[BT] Done. Final equity {self.capital:.2f} | trades={len(self.trades)}"
        )


class LiveOANDA:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.capital = float(cfg.get("initial_capital", 10000.0))
        self.positions: Dict[str, dict] = {}
        self.cool = Cooldown(
            int(cfg.get("cooldown_bars", 0)), int(
                cfg.get("entry_rearm_bars", 0))
        )
        self.client: Optional[OandaClient] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.ens: Dict[str, Ensemble] = {}
        self.risk_usage = 0.0

    def _setup(self):
        api = (
            getattr(keys, "OANDA_API_KEY", None)
            or getattr(keys, "ACCESS_TOKEN", None)
            if _HAS_KEYS
            else None
        )
        acc = (
            getattr(keys, "OANDA_ACCOUNT_ID", None)
            or getattr(keys, "ACCOUNT_ID", None)
            if _HAS_KEYS
            else None
        )
        api = api or os.environ.get(
            "OANDA_API_KEY") or os.environ.get("ACCESS_TOKEN")
        acc = acc or os.environ.get(
            "OANDA_ACCOUNT_ID") or os.environ.get("ACCOUNT_ID")
        if not api or not acc:
            raise RuntimeError("Missing OANDA credentials")
        self.client = OandaClient(
            api, acc, practice=self.cfg.get("oanda_practice", True))
        logger.info(
            f"OANDA connected (practice={self.cfg.get('oanda_practice', True)})")

    def _ensemble(self) -> Ensemble:
        strats: List[StrategyBase] = [
            TrendBreakout(STRAT_PARAMS),
            MomentumEMA(STRAT_PARAMS),
            PullbackReentry(STRAT_PARAMS),
        ]
        if self.cfg.get("use_mmxm", True):
            strats.append(MMXMV2(STRAT_PARAMS))
        if self.cfg.get("use_supertrend", True):
            strats.append(SuperTrendStrategy(STRAT_PARAMS))
        strats.append(RangeReversion(STRAT_PARAMS))
        weights: List[float] = [1.0, 1.0, 0.9]
        if self.cfg.get("use_mmxm", True):
            weights.append(0.6)
        if self.cfg.get("use_supertrend", True):
            weights.append(0.8)
        weights.append(0.7)
        return Ensemble(strats, weights)

    def _manage(self, sym: str, df: pd.DataFrame, i: int, price: float):
        pos = self.positions.get(sym)
        if not pos:
            return
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        exit_px: Optional[float] = None
        reason: Optional[str] = None
        if pos["side"] == "buy":
            if low <= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            elif high >= pos["tp"]:
                exit_px, reason = pos["tp"], "TP"
        else:
            if high >= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            elif low <= pos["tp"]:
                exit_px, reason = pos["tp"], "TP"
        if exit_px is not None:
            pnl = (
                (exit_px - pos["entry"]) * pos["size"]
                if pos["side"] == "buy"
                else (pos["entry"] - exit_px) * pos["size"]
            )
            self.capital += pnl - \
                float(self.cfg.get("commission_per_trade", 0.0))
            logger.info(
                f"[EXIT] {sym} {pos['side']} @ {exit_px:.5f} pnl={pnl:.2f} ({reason})"
            )
            self.cool.mark_exit(sym, i)
            self.risk_usage -= pos.get("risk_frac", 0.0)
            self.risk_usage = max(0.0, self.risk_usage)
            self.positions.pop(sym, None)

    def run(self, symbols: List[str], granularity: str = "M5", poll_seconds: int = 5):
        self._setup()
        for sym in symbols:
            o = oanda_symbol(sym)
            try:
                df = self.client.get_candles(
                    o, granularity=granularity, count=600)
            except Exception as e:
                logger.error(f"Init fetch fail {sym}: {e}")
                continue
            if df.empty:
                continue
            self.data[sym] = df
            self.ens[sym] = self._ensemble()
        if not self.data:
            logger.error("No symbols initialized")
            return
        logger.info(f"[LIVE] OANDA loop on {list(self.data.keys())}")
        while True:
            try:
                for sym in list(self.data.keys()):
                    o = oanda_symbol(sym)
                    df = self.client.get_candles(
                        o, granularity=granularity, count=350)
                    if df.empty:
                        continue
                    self.data[sym] = df
                    i = len(df) - 1
                    ts = df.index[i]
                    price = df["Close"].iloc[i]
                    self._manage(sym, df, i, price)
                    if sym in self.positions:
                        continue
                    if not session_ok(ts, self.cfg):
                        continue
                    if not self.cool.can_enter(sym, i):
                        continue
                    if self.risk_usage >= float(self.cfg.get("max_portfolio_risk", 1.0)):
                        continue
                    sig = self.ens[sym].aggregate(df, i)
                    if not sig:
                        continue
                    ok, _ = precision_gate(
                        df, i, sig["side"], sig["confidence"], self.cfg
                    )
                    if not ok:
                        continue
                    entry_price = price * (
                        1
                        + float(self.cfg.get("slippage_pct", 0.0))
                        * (1 if sig["side"] == "buy" else -1)
                    )
                    use_risk = float(self.cfg.get("risk_per_trade", 0.003))
                    units_f = compute_units(
                        self.capital, use_risk, entry_price, float(sig["sl"])
                    )
                    units_f *= float(self.cfg.get("position_multiplier", 1.0))
                    units_i = int(round(abs(units_f)))
                    units_i = max(
                        min_units_for(sym, self.cfg),
                        min(units_i, max_units_for(sym, self.cfg)),
                    )
                    if units_i == 0:
                        continue
                    risk_amt = abs(entry_price - sig["sl"]) * units_i
                    risk_frac = risk_amt / max(1e-9, self.capital)
                    if self.risk_usage + risk_frac > float(
                        self.cfg.get("max_portfolio_risk", 1.0)
                    ):
                        continue
                    send_units = units_i if sig["side"] == "buy" else -units_i
                    if self.cfg.get("dry_run", True):
                        logger.info(
                            f"[ENTRY] (DRY) {sym} {sig['side']} units={send_units} "
                            f"px={entry_price:.5f} sl={sig['sl']:.5f} "
                            f"tp={sig['tp']:.5f} conf={sig['confidence']:.2f}"
                        )
                    else:
                        try:
                            self.client.place_market(
                                o, send_units, tp=float(sig["tp"]), sl=float(sig["sl"])
                            )
                            logger.info(f"[ORDER] {sym} {send_units} placed")
                        except Exception as e:
                            logger.error(f"[ORDER FAIL] {sym}: {e}")
                            continue
                    self.positions[sym] = {
                        "side": sig["side"],
                        "entry": float(entry_price),
                        "size": float(abs(units_i)),
                        "sl": float(sig["sl"]),
                        "tp": float(sig["tp"]),
                        "entry_idx": i,
                        "risk_frac": risk_frac,
                    }
                    self.risk_usage += risk_frac
                    self.cool.mark_entry(sym, i)
                time.sleep(max(1, int(poll_seconds)))
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.exception(f"Live error: {e}")
                time.sleep(5)


# ========================= MAIN =========================
def main():
    ap = argparse.ArgumentParser(
        description="Selective FX Bot with Confluence & Risk Caps"
    )
    ap.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--live-csv", action="store_true")
    ap.add_argument(
        "--dry-run",
        default=None,
        type=lambda s: s.lower() in ["true", "1", "t", "yes"],
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--gran", default=None)
    ap.add_argument("--poll", default=None, type=int)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--prefer-csv", default=None)
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--risk", type=float, default=None)
    ap.add_argument("--pos-mult", type=float, default=None)
    ap.add_argument("--min-units-default", type=int, default=None)
    ap.add_argument("--win-target", type=float, default=None)
    # Optional override for Monte Carlo paths
    ap.add_argument("--mc-paths", type=int, default=None)
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = dict(DEFAULT_CONFIG)
    if args.dry_run is not None:
        cfg["dry_run"] = bool(args.dry_run)
    if args.gran:
        cfg["granularity"] = args.gran.upper()
    if args.poll is not None:
        cfg["poll_seconds"] = int(args.poll)
    if args.prefer_csv:
        cfg["prefer_csv_glob"] = args.prefer_csv
    if args.precision:
        cfg["precision_mode"] = True
    if args.risk is not None:
        cfg["risk_per_trade"] = max(0.0002, min(0.02, float(args.risk)))
    if args.pos_mult is not None:
        cfg["position_multiplier"] = max(0.1, float(args.pos_mult))
    if args.min_units_default is not None:
        cfg["min_units_default"] = max(1, int(args.min_units_default))
    if args.win_target is not None:
        try:
            t = float(args.win_target)
        except Exception:
            t = None
        if t and t >= 80.0:
            cfg.update(
                {
                    "precision_mode": True,
                    "min_confidence": 0.65,
                    "min_adx": 22,
                    "min_atr_std_ratio": 0.30,
                }
            )
    if args.mc_paths is not None:
        cfg["mc_paths"] = max(100, int(args.mc_paths))

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        logger.error(f"[PATH] data_dir does not exist -> {data_dir}")
        raise SystemExit(1)
    has_csvs = any(fn.lower().endswith(".csv") for fn in os.listdir(data_dir))
    logger.info(f"[PATH] {data_dir} ({'has CSVs' if has_csvs else 'exists'})")

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else []
    )
    data_map = load_all_data(
        data_dir, symbols or cfg["instruments"], prefer_glob=cfg.get("prefer_csv_glob")
    )

    if args.live_csv and not args.live and not args.backtest:
        if not data_map:
            logger.error("No data loaded. Put CSV files in data directory.")
            raise SystemExit(1)
        runner = LiveCSVRunner(cfg, STRAT_PARAMS)
        live_syms = symbols if symbols else list(data_map.keys())
        runner.run(live_syms, data_map)
        return

    if args.live:
        live_syms = symbols if symbols else cfg["instruments"]
        runner = LiveOANDA(cfg)
        runner.run(
            live_syms,
            granularity=cfg.get("granularity", "M5"),
            poll_seconds=int(cfg.get("poll_seconds", 5)),
        )
        return

    if args.backtest:
        if not data_map:
            logger.error("No data loaded for backtest.")
            raise SystemExit(1)
        bt = Backtester(cfg)
        bt.run(
            symbols if symbols else list(data_map.keys()),
            data_map,
            start=args.start,
            end=args.end,
        )
        return

    logger.info("Nothing selected. Use --backtest, --live, or --live-csv.")
    logger.info(
        'Example backtest:  python ultimate_bot_v4_4.py --backtest --symbols "EURUSD,GBPUSD,XAUUSD" --precision'
    )
    logger.info(
        'Example live-csv:  python ultimate_bot_v4_4.py --live-csv --symbols "EURUSD,GBPUSD" --precision'
    )


if __name__ == "__main__":
    main()
