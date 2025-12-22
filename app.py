import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

from curl_cffi import requests as c_requests

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Earnings Radar", page_icon="📡", layout="wide")
alt.data_transformers.disable_max_rows()

st.title("📡 Earnings Radar")
st.caption(
    "Upload a portfolio to see next earnings dates + quick price context, and review the last 4 earnings events "
    "(EPS estimate vs actual, surprise, and 1D/5D price reaction). ETFs typically have no earnings."
)

# -----------------------------
# Utils
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _fmt_num(x: Any, d: int = 2) -> str:
    v = _safe_float(x)
    return "—" if v is None else f"{v:,.{d}f}"


def _fmt_pct(x: Any, d: int = 1) -> str:
    v = _safe_float(x)
    return "—" if v is None else f"{v:+.{d}f}%"


def _normalize_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def today_utc_date() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def read_portfolio_file(uploaded) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)  # needs openpyxl
    raise ValueError("Unsupported file type. Upload .csv, .xlsx, or .xls")


def find_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Ticker", "Symbol", "Ticker Symbol", "Trading Symbol", "Security", "Instrument"]
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    lower_map = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


# -----------------------------
# Yahoo quoteSummary (reliable earnings source)
# -----------------------------
@st.cache_resource
def yahoo_session() -> c_requests.Session:
    # Chrome impersonation improves response consistency
    return c_requests.Session(impersonate="chrome")


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def quote_summary(ticker: str, modules: List[str]) -> Dict[str, Any]:
    """
    Calls Yahoo quoteSummary endpoint and returns the JSON result dict (or empty dict).
    """
    tk = _normalize_ticker(ticker)
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{tk}"
    params = {"modules": ",".join(modules)}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    try:
        r = yahoo_session().get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return {}
        j = r.json()
        res = j.get("quoteSummary", {}).get("result")
        if not res:
            return {}
        return res[0] or {}
    except Exception:
        return {}


def _is_etf_like(summary: Dict[str, Any]) -> bool:
    # Use quoteType if present
    try:
        qt = (summary.get("quoteType", {}) or {}).get("quoteType")
        if isinstance(qt, str) and qt.upper() in {"ETF", "MUTUALFUND", "FUND", "INDEX"}:
            return True
    except Exception:
        pass
    return False


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_next_earnings_from_yahoo(ticker: str) -> Optional[pd.Timestamp]:
    """
    Next earnings date from quoteSummary calendarEvents.
    """
    s = quote_summary(ticker, modules=["calendarEvents", "quoteType"])
    if not s:
        return None

    if _is_etf_like(s.get("quoteType", {}) or {}):
        return None

    try:
        earnings = (s.get("calendarEvents", {}) or {}).get("earnings", {}) or {}
        dates = earnings.get("earningsDate") or []
        # dates is usually list of dicts with 'raw' epoch seconds
        now = pd.Timestamp(today_utc_date(), tz="UTC")
        candidates = []
        for d in dates:
            raw = None
            if isinstance(d, dict):
                raw = d.get("raw")
            if raw is None:
                continue
            ts = pd.to_datetime(int(raw), unit="s", utc=True, errors="coerce")
            if pd.notna(ts) and ts >= now:
                candidates.append(ts)
        if candidates:
            return pd.Timestamp(min(candidates))
    except Exception:
        pass

    return None


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_past_quarterly_earnings_from_yahoo(ticker: str, limit: int = 8) -> pd.DataFrame:
    """
    Past quarterly earnings events from quoteSummary earningsHistory.quarterlyEarnings
    Returns a dataframe with:
      - earnings_date (UTC)
      - epsEstimate, epsActual, surprisePercent
    """
    s = quote_summary(ticker, modules=["earningsHistory", "quoteType"])
    if not s:
        return pd.DataFrame()

    if _is_etf_like(s.get("quoteType", {}) or {}):
        return pd.DataFrame()

    try:
        eh = (s.get("earningsHistory", {}) or {})
        q = eh.get("history") or []
        rows = []
        for item in q[: max(1, limit)]:
            # item fields: quarter (raw), epsEstimate/epsActual/surprisePercent etc
            quarter = (item.get("quarter") or {}).get("raw")
            # Some tickers have quarter but no earnings date; quarter works as event anchor (best available)
            if quarter is None:
                continue

            ts = pd.to_datetime(int(quarter), unit="s", utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            eps_est = (item.get("epsEstimate") or {}).get("raw")
            eps_act = (item.get("epsActual") or {}).get("raw")
            surprise = (item.get("surprisePercent") or {}).get("raw")

            rows.append(
                {
                    "Earnings Date (UTC)": ts,
                    "EPS Estimate": _safe_float(eps_est),
                    "Reported EPS": _safe_float(eps_act),
                    "Surprise(%)": _safe_float(surprise) * 100.0 if _safe_float(surprise) is not None else None,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("Earnings Date (UTC)")
        return df
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Price history + reaction context
# -----------------------------
@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=_normalize_ticker(ticker),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        return df
    except Exception:
        return pd.DataFrame()


def event_reaction_metrics(hist: pd.DataFrame, event_ts: pd.Timestamp) -> Optional[Dict[str, Any]]:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna()
    if close.empty:
        return None

    event_day = pd.to_datetime(event_ts, utc=True, errors="coerce")
    if pd.isna(event_day):
        return None
    event_day = event_day.normalize()

    idx_norm = close.index.normalize()
    pos = np.where(idx_norm <= event_day)[0]
    if len(pos) == 0:
        return None
    p0 = int(pos[-1])

    prev_close = _safe_float(close.iloc[p0])
    next1 = _safe_float(close.iloc[p0 + 1]) if p0 + 1 < len(close) else None
    next5 = _safe_float(close.iloc[p0 + 5]) if p0 + 5 < len(close) else None

    move_1d = (next1 / prev_close - 1.0) * 100.0 if (prev_close not in (None, 0) and next1 is not None) else None
    move_5d = (next5 / prev_close - 1.0) * 100.0 if (prev_close not in (None, 0) and next5 is not None) else None

    roll_hi = close.rolling(window=252, min_periods=20).max()
    roll_lo = close.rolling(window=252, min_periods=20).min()

    hi = _safe_float(roll_hi.iloc[p0])
    lo = _safe_float(roll_lo.iloc[p0])

    dist_hi = (prev_close - hi) / hi * 100.0 if (prev_close is not None and hi not in (None, 0)) else None
    dist_lo = (prev_close - lo) / lo * 100.0 if (prev_close is not None and lo not in (None, 0)) else None

    return {
        "Event Close (prev)": prev_close,
        "1D Move (%)": move_1d,
        "5D Move (%)": move_5d,
        "Dist vs 52W High (%)": dist_hi,
        "Dist vs 52W Low (%)": dist_lo,
    }


# -----------------------------
# 1Y price metrics for the overview table
# -----------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_1y_price_metrics(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    results = []
    for batch in chunk_list(tickers, 120):
        try:
            data = yf.download(
                tickers=batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            data = pd.DataFrame()

        if data is None or data.empty:
            for tk in batch:
                results.append({"Ticker": tk})
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for tk in batch:
                if tk not in data.columns.get_level_values(0):
                    results.append({"Ticker": tk})
                    continue
                closes = data[(tk, "Close")].dropna()
                highs = data[(tk, "High")].dropna()
                lows = data[(tk, "Low")].dropna()

                cur = _safe_float(closes.iloc[-1]) if not closes.empty else None
                hi = _safe_float(highs.max()) if not highs.empty else None
                lo = _safe_float(lows.min()) if not lows.empty else None

                d_hi = (cur - hi) / hi * 100.0 if (cur is not None and hi not in (None, 0)) else None
                d_lo = (cur - lo) / lo * 100.0 if (cur is not None and lo not in (None, 0)) else None
                rng = (hi - lo) / lo * 100.0 if (hi is not None and lo not in (None, 0)) else None

                results.append(
                    {
                        "Ticker": tk,
                        "Current": cur,
                        "52W High": hi,
                        "52W Low": lo,
                        "Δ vs 52W High (%)": d_hi,
                        "Δ vs 52W Low (%)": d_lo,
                        "52W Range (%)": rng,
                    }
                )
        else:
            tk = batch[0]
            closes = data["Close"].dropna()
            highs = data["High"].dropna()
            lows = data["Low"].dropna()

            cur = _safe_float(closes.iloc[-1]) if not closes.empty else None
            hi = _safe_float(highs.max()) if not highs.empty else None
            lo = _safe_float(lows.min()) if not lows.empty else None

            d_hi = (cur - hi) / hi * 100.0 if (cur is not None and hi not in (None, 0)) else None
            d_lo = (cur - lo) / lo * 100.0 if (cur is not None and lo not in (None, 0)) else None
            rng = (hi - lo) / lo * 100.0 if (hi is not None and lo not in (None, 0)) else None

            results.append(
                {
                    "Ticker": tk,
                    "Current": cur,
                    "52W High": hi,
                    "52W Low": lo,
                    "Δ vs 52W High (%)": d_hi,
                    "Δ vs 52W Low (%)": d_lo,
                    "52W Range (%)": rng,
                }
            )

    return pd.DataFrame(results).drop_duplicates(subset=["Ticker"])


# -----------------------------
# Portfolio UI (primary)
# -----------------------------
st.subheader("📤 Upload Portfolio (Primary)")
st.markdown("Upload a **CSV or Excel** file with a ticker column (e.g. `Ticker` or `Symbol`).")

portfolio_file = st.file_uploader("Portfolio file", type=["csv", "xlsx", "xls"])

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    max_tickers = st.slider("Max tickers to process (speed)", 5, 150, 40, 5)
with c2:
    only_with_next = st.checkbox("Only tickers with next earnings", value=False)
with c3:
    show_debug = st.checkbox("Debug (show why missing)", value=False)

run = st.button("Run Earnings Radar", type="primary")

with st.popover("ℹ️ Help"):
    st.markdown(
        """
**Why some rows can be missing?**
- ETFs/funds typically have **no earnings**.
- Some tickers (ADR/OTC) may have incomplete Yahoo event data.
- If Yahoo blocks a request temporarily, you may see blanks — rerun later.

This app avoids `yfinance.get_earnings_dates()` and pulls earnings events from Yahoo quoteSummary instead.
        """.strip()
    )

if run:
    if portfolio_file is None:
        st.error("Upload a portfolio file first.")
        st.stop()

    try:
        pf = read_portfolio_file(portfolio_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    tcol = find_ticker_column(pf)
    if tcol is None:
        st.error("No ticker column found. Rename your column to `Ticker` or `Symbol` and retry.")
        st.stop()

    tickers = pf[tcol].astype(str).apply(_normalize_ticker).dropna().tolist()
    tickers = [t for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        st.error("No valid tickers found.")
        st.stop()

    if len(tickers) > max_tickers:
        st.warning(f"Portfolio has {len(tickers)} tickers — processing first {max_tickers} for speed.")
        tickers = tickers[:max_tickers]

    st.divider()
    st.subheader("📦 Portfolio Overview")

    with st.spinner("Fetching price metrics..."):
        pm = fetch_1y_price_metrics(tickers)

    # Next earnings (quoteSummary)
    bar = st.progress(0.0)
    next_rows = []
    for i, tk in enumerate(tickers, start=1):
        nxt = fetch_next_earnings_from_yahoo(tk)
        next_rows.append({"Ticker": tk, "Next Earnings (UTC)": nxt})
        bar.progress(i / max(1, len(tickers)))
    bar.empty()

    ed = pd.DataFrame(next_rows)
    ov = pm.merge(ed, on="Ticker", how="left")

    if only_with_next:
        ov = ov[ov["Next Earnings (UTC)"].notna()].reset_index(drop=True)

    disp = ov.copy()
    disp["Next Earnings (UTC)"] = pd.to_datetime(disp["Next Earnings (UTC)"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    disp["Current"] = disp["Current"].apply(lambda x: _fmt_num(x, 2))
    for c in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
        disp[c] = disp[c].apply(lambda x: _fmt_pct(x, 1))

    disp = disp.sort_values("Next Earnings (UTC)", na_position="last")

    st.dataframe(
        disp[["Ticker", "Next Earnings (UTC)", "Current", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Portfolio Overview (CSV)",
        ov.to_csv(index=False),
        file_name="portfolio_earnings_overview.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🧾 Past 4 Earnings + Price Reaction (per holding)")

    for tk in tickers:
        with st.expander(tk, expanded=False):
            # Pull last earnings events from Yahoo quoteSummary
            e = fetch_past_quarterly_earnings_from_yahoo(tk, limit=12)
            if e.empty:
                st.info("No earnings history returned by Yahoo for this ticker (or it’s an ETF/fund / incomplete listing).")
                if show_debug:
                    raw = quote_summary(tk, modules=["quoteType", "calendarEvents", "earningsHistory"])
                    st.write("Debug raw keys:", list(raw.keys()) if isinstance(raw, dict) else raw)
                continue

            # Keep past 4 (most recent)
            e = e.sort_values("Earnings Date (UTC)").tail(4).reset_index(drop=True)

            hist = fetch_price_history(tk, period="3y")
            rows = []
            for _, r in e.iterrows():
                ts = r["Earnings Date (UTC)"]
                ctx = event_reaction_metrics(hist, ts) if hist is not None and not hist.empty else None
                row = {
                    "Earnings Date (UTC)": pd.to_datetime(ts, utc=True).strftime("%Y-%m-%d"),
                    "EPS Estimate": r.get("EPS Estimate"),
                    "Reported EPS": r.get("Reported EPS"),
                    "Surprise(%)": r.get("Surprise(%)"),
                }
                if ctx:
                    row.update(ctx)
                rows.append(row)

            out = pd.DataFrame(rows)

            # Format
            disp2 = out.copy()
            for col in ["EPS Estimate", "Reported EPS", "Event Close (prev)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_num(x, 2))
            for col in ["Surprise(%)", "1D Move (%)", "5D Move (%)", "Dist vs 52W High (%)", "Dist vs 52W Low (%)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_pct(x, 1))

            order = [
                "Earnings Date (UTC)",
                "EPS Estimate",
                "Reported EPS",
                "Surprise(%)",
                "Event Close (prev)",
                "1D Move (%)",
                "5D Move (%)",
                "Dist vs 52W High (%)",
                "Dist vs 52W Low (%)",
            ]
            order = [c for c in order if c in disp2.columns]

            st.dataframe(disp2[order], use_container_width=True, hide_index=True)
