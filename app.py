import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from curl_cffi import requests as c_requests


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Earnings Radar", page_icon="📡", layout="wide")
st.title("📡 Earnings Radar")
st.caption(
    "Upload a portfolio and get: next earnings dates + 52W context, and last 4 earnings events with "
    "EPS (est vs actual), surprise, and 1D/5D price reaction. "
    "Note: ETFs typically have no earnings."
)


# -----------------------------
# Helpers
# -----------------------------
def _normalize_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


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


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def today_utc_date() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).normalize()


def read_portfolio_file(uploaded) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
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
# Yahoo quoteSummary (hardened)
# -----------------------------
@st.cache_resource
def yahoo_session() -> c_requests.Session:
    """
    Persistent session with a warm-up request to set Yahoo cookies.
    This improves reliability on Streamlit Cloud.
    """
    sess = c_requests.Session(impersonate="chrome")
    try:
        # Warm cookies / consent flow (best-effort)
        sess.get(
            "https://finance.yahoo.com/",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
    except Exception:
        pass
    return sess


def _qs_request(url: str, params: Dict[str, str]) -> Dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
        "Connection": "keep-alive",
    }

    # retry a couple times (Yahoo sometimes returns transient empties)
    for attempt in range(3):
        try:
            r = yahoo_session().get(url, params=params, headers=headers, timeout=20)
            if r.status_code != 200:
                time.sleep(0.4 * (attempt + 1))
                continue

            j = r.json()
            res = j.get("quoteSummary", {}).get("result")
            if res and isinstance(res, list) and res[0]:
                return res[0]
            time.sleep(0.4 * (attempt + 1))
        except Exception:
            time.sleep(0.4 * (attempt + 1))

    return {}


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def quote_summary(ticker: str, modules: List[str]) -> Dict[str, Any]:
    """
    Uses query1 + query2 fallback. Returns quoteSummary result dict or {}.
    """
    tk = _normalize_ticker(ticker)
    params = {"modules": ",".join(modules)}

    url1 = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{tk}"
    url2 = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{tk}"

    out = _qs_request(url1, params)
    if out:
        return out

    out = _qs_request(url2, params)
    return out or {}


def _quote_type(summary: Dict[str, Any]) -> Optional[str]:
    try:
        qt = (summary.get("quoteType") or {}).get("quoteType")
        if isinstance(qt, str):
            return qt.upper()
    except Exception:
        pass
    return None


def _is_fund_like(summary: Dict[str, Any]) -> bool:
    qt = _quote_type(summary)
    return qt in {"ETF", "MUTUALFUND", "FUND", "INDEX"}


# -----------------------------
# Earnings: Next + Past
# -----------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def next_earnings_date(ticker: str) -> Tuple[Optional[pd.Timestamp], str]:
    """
    Returns (next_earnings_ts_utc, reason)
    """
    tk = _normalize_ticker(ticker)

    # 1) Yahoo quoteSummary calendarEvents (best)
    s = quote_summary(tk, modules=["calendarEvents", "quoteType"])
    if s:
        if _is_fund_like(s):
            return None, "ETF/Fund (no earnings)"
        try:
            earnings = (s.get("calendarEvents") or {}).get("earnings") or {}
            dates = earnings.get("earningsDate") or []
            now = today_utc_date()
            candidates = []
            for d in dates:
                raw = d.get("raw") if isinstance(d, dict) else None
                if raw is None:
                    continue
                ts = pd.to_datetime(int(raw), unit="s", utc=True, errors="coerce")
                if pd.notna(ts) and ts.normalize() >= now:
                    candidates.append(ts)
            if candidates:
                return pd.Timestamp(min(candidates)), "Yahoo calendarEvents"
        except Exception:
            pass

    # 2) yfinance fallback (sometimes works when Yahoo blocks quoteSummary)
    try:
        t = yf.Ticker(tk)
        cal = t.calendar
        # calendar can be dict-like or df-like depending on yfinance version
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # common layout: a single column with index labels
            # try to find "Earnings Date"
            idx = [str(i).lower() for i in cal.index]
            if any("earnings date" in i for i in idx):
                row = cal.loc[[cal.index[idx.index(next(i for i in idx if "earnings date" in i))]]]
                # may contain list / Timestamp
                val = row.iloc[0, 0]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    dt = pd.to_datetime(val[0], utc=True, errors="coerce")
                else:
                    dt = pd.to_datetime(val, utc=True, errors="coerce")
                if pd.notna(dt):
                    return pd.Timestamp(dt), "yfinance calendar"
        elif isinstance(cal, dict):
            # sometimes "Earnings Date" key exists
            for k in cal.keys():
                if "earn" in str(k).lower():
                    dt = pd.to_datetime(cal[k], utc=True, errors="coerce")
                    if pd.notna(dt):
                        return pd.Timestamp(dt), "yfinance calendar"
    except Exception:
        pass

    return None, "No next earnings from Yahoo/yfinance"


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def past_earnings_events(ticker: str, limit: int = 12) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df, reason). df columns:
      Earnings Date (UTC), EPS Estimate, Reported EPS, Surprise(%)
    """
    tk = _normalize_ticker(ticker)

    # 1) Yahoo quoteSummary earningsHistory (preferred)
    s = quote_summary(tk, modules=["earningsHistory", "quoteType"])
    if s:
        if _is_fund_like(s):
            return pd.DataFrame(), "ETF/Fund (no earnings)"
        try:
            eh = s.get("earningsHistory") or {}
            hist = eh.get("history") or []
            rows = []
            for item in hist[: max(1, limit)]:
                quarter_raw = (item.get("quarter") or {}).get("raw")
                if quarter_raw is None:
                    continue
                ts = pd.to_datetime(int(quarter_raw), unit="s", utc=True, errors="coerce")
                if pd.isna(ts):
                    continue

                eps_est = (item.get("epsEstimate") or {}).get("raw")
                eps_act = (item.get("epsActual") or {}).get("raw")
                surprise = (item.get("surprisePercent") or {}).get("raw")

                rows.append(
                    {
                        "Earnings Date (UTC)": pd.Timestamp(ts),
                        "EPS Estimate": _safe_float(eps_est),
                        "Reported EPS": _safe_float(eps_act),
                        "Surprise(%)": (_safe_float(surprise) * 100.0) if _safe_float(surprise) is not None else None,
                    }
                )
            if rows:
                df = pd.DataFrame(rows).sort_values("Earnings Date (UTC)")
                return df, "Yahoo earningsHistory"
        except Exception:
            pass

    # 2) yfinance fallback: get_earnings_dates (can be flaky but helps some tickers)
    try:
        t = yf.Ticker(tk)
        df = t.get_earnings_dates(limit=limit)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df.copy()
            # Ensure a standard schema
            # yfinance typically uses index as DatetimeIndex
            if not isinstance(out.index, pd.DatetimeIndex):
                # try common columns
                for c in out.columns:
                    if "earn" in str(c).lower() and "date" in str(c).lower():
                        out.index = pd.to_datetime(out[c], utc=True, errors="coerce")
                        break

            out = out[~out.index.isna()].sort_index()
            out = out.reset_index().rename(columns={"index": "Earnings Date (UTC)"})
            out["Earnings Date (UTC)]"] = out["Earnings Date (UTC)"]

            # normalize column names
            col_map = {}
            for c in out.columns:
                lc = str(c).lower()
                if "eps estimate" in lc:
                    col_map[c] = "EPS Estimate"
                if "reported eps" in lc or (lc == "epsactual"):
                    col_map[c] = "Reported EPS"
                if "surprise" in lc and "%" in lc:
                    col_map[c] = "Surprise(%)"
                if "earnings date" in lc:
                    col_map[c] = "Earnings Date (UTC)"
            out = out.rename(columns=col_map)

            keep = ["Earnings Date (UTC)", "EPS Estimate", "Reported EPS", "Surprise(%)"]
            keep = [c for c in keep if c in out.columns]
            out = out[keep].copy()

            # index may already be UTC; force UTC
            out["Earnings Date (UTC)"] = pd.to_datetime(out["Earnings Date (UTC)"], utc=True, errors="coerce")
            out = out.dropna(subset=["Earnings Date (UTC)"]).sort_values("Earnings Date (UTC)")
            if not out.empty:
                return out, "yfinance get_earnings_dates"
    except Exception:
        pass

    return pd.DataFrame(), "No earnings history from Yahoo/yfinance"


# -----------------------------
# Prices + 52W context
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


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
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


def event_reaction_metrics(hist: pd.DataFrame, event_ts: pd.Timestamp) -> Dict[str, Any]:
    """
    Uses the last close on/before event date and computes 1D and 5D change after that.
    """
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}

    close = hist["Close"].dropna()
    if close.empty:
        return {}

    event_day = pd.to_datetime(event_ts, utc=True, errors="coerce")
    if pd.isna(event_day):
        return {}
    event_day = event_day.normalize()

    idx_norm = close.index.normalize()
    pos = np.where(idx_norm <= event_day)[0]
    if len(pos) == 0:
        return {}
    p0 = int(pos[-1])

    prev_close = _safe_float(close.iloc[p0])
    next1 = _safe_float(close.iloc[p0 + 1]) if p0 + 1 < len(close) else None
    next5 = _safe_float(close.iloc[p0 + 5]) if p0 + 5 < len(close) else None

    move_1d = (next1 / prev_close - 1.0) * 100.0 if (prev_close not in (None, 0) and next1 is not None) else None
    move_5d = (next5 / prev_close - 1.0) * 100.0 if (prev_close not in (None, 0) and next5 is not None) else None

    return {
        "Event Close (prev)": prev_close,
        "1D Move (%)": move_1d,
        "5D Move (%)": move_5d,
    }


# -----------------------------
# UI: Upload is PRIMARY
# -----------------------------
st.subheader("📤 Upload Portfolio (Primary)")
st.markdown("Upload a **CSV or Excel** file with a ticker column (like `Ticker` or `Symbol`).")

portfolio_file = st.file_uploader("Portfolio file", type=["csv", "xlsx", "xls"])

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    max_tickers = st.slider("Max tickers to process (speed)", 5, 200, 40, 5)
with c2:
    only_with_next = st.checkbox("Only tickers with next earnings", value=False)
with c3:
    show_source_cols = st.checkbox("Show data-source columns", value=True)

with st.popover("ℹ️ Why is earnings sometimes missing?"):
    st.markdown(
        """
- **ETFs/Funds** usually have **no earnings** (QQQ, VOO, SOXX etc.).
- Yahoo can **rate-limit** or **block** certain endpoints occasionally, especially on cloud hosting.
- This app uses a **hardened Yahoo request**, and then a **yfinance fallback** for both next earnings and history.
        """.strip()
    )

run = st.button("Run Earnings Radar", type="primary")

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

    with st.spinner("Fetching 52-week price metrics..."):
        pm = fetch_1y_price_metrics(tickers)

    # Next earnings per ticker with reasons
    bar = st.progress(0.0)
    next_rows = []
    for i, tk in enumerate(tickers, start=1):
        nxt, reason = next_earnings_date(tk)
        next_rows.append({"Ticker": tk, "Next Earnings (UTC)": nxt, "Next Earnings Source": reason})
        bar.progress(i / max(1, len(tickers)))
    bar.empty()

    ed = pd.DataFrame(next_rows)
    ov = pm.merge(ed, on="Ticker", how="left")

    if only_with_next:
        ov = ov[ov["Next Earnings (UTC)"].notna()].reset_index(drop=True)

    # Display formatting
    disp = ov.copy()
    disp["Next Earnings (UTC)"] = pd.to_datetime(disp["Next Earnings (UTC)"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    disp["Current"] = disp["Current"].apply(lambda x: _fmt_num(x, 2))
    for c in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda x: _fmt_pct(x, 1))

    # sort: earnings date first
    disp = disp.sort_values("Next Earnings (UTC)", na_position="last")

    cols = ["Ticker", "Next Earnings (UTC)", "Current", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]
    if show_source_cols:
        cols.append("Next Earnings Source")

    st.dataframe(disp[cols], use_container_width=True, hide_index=True)

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
            e_df, e_src = past_earnings_events(tk, limit=16)
            if e_df.empty:
                st.info(f"No earnings history returned for {tk}. Source: {e_src}")
                continue

            # Keep last 4
            e_df = e_df.sort_values("Earnings Date (UTC)").tail(4).reset_index(drop=True)

            hist = fetch_price_history(tk, period="2y")

            rows = []
            for _, r in e_df.iterrows():
                ts = r["Earnings Date (UTC)"]
                ctx = event_reaction_metrics(hist, ts)

                rows.append(
                    {
                        "Earnings Date (UTC)": pd.to_datetime(ts, utc=True).strftime("%Y-%m-%d"),
                        "EPS Estimate": r.get("EPS Estimate"),
                        "Reported EPS": r.get("Reported EPS"),
                        "Surprise(%)": r.get("Surprise(%)"),
                        **ctx,
                    }
                )

            out = pd.DataFrame(rows)

            # Format
            disp2 = out.copy()
            for col in ["EPS Estimate", "Reported EPS", "Event Close (prev)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_num(x, 2))
            for col in ["Surprise(%)", "1D Move (%)", "5D Move (%)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_pct(x, 1))

            st.caption(f"Source: {e_src}")
            st.dataframe(disp2, use_container_width=True, hide_index=True)
