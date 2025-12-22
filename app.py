import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Earnings Radar", page_icon="📡", layout="wide")
alt.data_transformers.disable_max_rows()

st.title("📡 Earnings Radar")
st.caption(
    "Portfolio-first earnings monitoring. Upload holdings to get next earnings + price context, "
    "and review last 4 earnings reactions."
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    if v is None:
        return "—"
    return f"{v:,.{d}f}"


def _fmt_pct(x: Any, d: int = 1) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    return f"{v:+.{d}f}%"


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def today_utc_date() -> "datetime":
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _normalize_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def read_portfolio_file(uploaded) -> pd.DataFrame:
    """
    Reads CSV or Excel based on file extension.
    """
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)  # requires openpyxl for .xlsx
    raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")


def find_ticker_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to locate a ticker column from common names.
    """
    candidates = [
        "Ticker",
        "Symbol",
        "Ticker Symbol",
        "Trading Symbol",
        "Security",
        "Instrument",
    ]
    cols = list(df.columns)

    # exact match first
    for c in candidates:
        if c in cols:
            return c

    # case-insensitive match
    lower_map = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


# -----------------------------------------------------------------------------
# Universe loading (for tagging)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_universe_csv(path: str = "sp500_universe.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Ticker", "Company", "Sector", "Industry"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Universe CSV missing columns: {missing}. Required: {sorted(required)}")

    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).apply(_normalize_ticker)
    df["Company"] = df["Company"].fillna("").astype(str).str.strip()
    df["Sector"] = df["Sector"].fillna("").astype(str).str.strip()
    df["Industry"] = df["Industry"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Price metrics (batched) via yf.download
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_1y_price_metrics(tickers: List[str]) -> pd.DataFrame:
    """
    Returns per ticker:
      - current (last close)
      - 52w high / low
      - Δ vs 52w high (%)
      - Δ vs 52w low (%)
      - 52w range (%)
    """
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


# -----------------------------------------------------------------------------
# Earnings (cached)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_next_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    now = today_utc_date()

    # best attempt: earnings table
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=12)
        if df is not None and not df.empty:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce").dropna()
            future = idx[idx >= pd.Timestamp(now)]
            if len(future) > 0:
                return pd.Timestamp(future.min())
    except Exception:
        pass

    # fallback: info['earningsDate']
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        ed = info.get("earningsDate")
        if isinstance(ed, list) and len(ed) > 0:
            candidates = []
            for x in ed:
                ts = pd.to_datetime(x, utc=True, errors="coerce")
                if pd.notna(ts):
                    candidates.append(ts)
            candidates = [c for c in candidates if c >= pd.Timestamp(now)]
            if candidates:
                return pd.Timestamp(min(candidates))
        else:
            ts = pd.to_datetime(ed, utc=True, errors="coerce")
            if pd.notna(ts) and ts >= pd.Timestamp(now):
                return pd.Timestamp(ts)
    except Exception:
        pass

    return None


def fetch_earnings_for_list(tickers: List[str]) -> pd.DataFrame:
    bar = st.progress(0.0)
    rows = []
    n = len(tickers)
    for i, tk in enumerate(tickers, start=1):
        rows.append({"Ticker": tk, "Next Earnings (UTC)": fetch_next_earnings_date(tk)})
        bar.progress(i / max(1, n))
    bar.empty()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Past 4 earnings + context (portfolio)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_earnings_history_table(ticker: str, limit: int = 16) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()].sort_index()
        return out
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_price_history_for_events(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert("UTC")
        df = df[~df.index.isna()]
        return df
    except Exception:
        return pd.DataFrame()


def _event_context_from_history(hist: pd.DataFrame, event_ts: pd.Timestamp) -> Optional[Dict[str, Any]]:
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


def portfolio_last4_rows(ticker: str) -> pd.DataFrame:
    e = fetch_earnings_history_table(ticker, limit=16)
    if e.empty:
        return pd.DataFrame()

    now = today_utc_date()
    e = e.sort_index()
    past = e[e.index < pd.Timestamp(now)]
    if past.empty:
        return pd.DataFrame()

    last4 = past.tail(4).copy()
    hist = fetch_price_history_for_events(ticker, period="3y")

    rows = []
    for ts, row in last4.iterrows():
        ctx = _event_context_from_history(hist, ts) if not hist.empty else None
        base = {"Ticker": ticker, "Earnings Date (UTC)": ts.strftime("%Y-%m-%d")}
        for col in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
            base[col] = _safe_float(row.get(col)) if col in last4.columns else None
        if ctx:
            base.update(ctx)
        rows.append(base)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Load universe (for tags)
# -----------------------------------------------------------------------------
try:
    universe = load_universe_csv("sp500_universe.csv")
except Exception as e:
    st.error(str(e))
    st.stop()

meta = universe[["Ticker", "Company", "Sector", "Industry"]].drop_duplicates(subset=["Ticker"]).set_index("Ticker")

# -----------------------------------------------------------------------------
# Portfolio-first UI
# -----------------------------------------------------------------------------
st.subheader("📤 Upload Portfolio (Primary)")
st.markdown("Upload a **CSV or Excel** file containing a ticker column (e.g. `Ticker` or `Symbol`).")

portfolio_file = st.file_uploader("Portfolio file", type=["csv", "xlsx", "xls"])

pf_max_names = st.slider("Max tickers to process (speed control)", 5, 120, 40, 5)
pf_only_next_earnings = st.checkbox("Show only tickers with a next earnings date", value=False)
pf_run = st.button("Run Portfolio Earnings Radar", type="primary")

with st.popover("ℹ️ File format help"):
    st.markdown(
        """
Accepted files:
- `.csv`
- `.xlsx` / `.xls`

We auto-detect ticker columns like:
`Ticker`, `Symbol`, `Ticker Symbol`, etc.
        """.strip()
    )

if pf_run:
    if portfolio_file is None:
        st.error("Upload a portfolio file first.")
        st.stop()

    try:
        pf = read_portfolio_file(portfolio_file)
    except Exception as e:
        st.error(f"Could not read portfolio file: {e}")
        st.stop()

    ticker_col = find_ticker_column(pf)
    if ticker_col is None:
        st.error("Could not find a ticker column. Rename your column to `Ticker` or `Symbol` and try again.")
        st.stop()

    pf_tickers = pf[ticker_col].astype(str).apply(_normalize_ticker).dropna().tolist()
    pf_tickers = [t for t in pf_tickers if t]
    pf_tickers = list(dict.fromkeys(pf_tickers))

    if not pf_tickers:
        st.error("No valid tickers found in your file.")
        st.stop()

    if len(pf_tickers) > pf_max_names:
        st.warning(f"Portfolio has {len(pf_tickers)} tickers — showing first {pf_max_names} for speed.")
        pf_tickers = pf_tickers[:pf_max_names]

    st.divider()
    st.subheader("📦 Portfolio Overview")

    with st.spinner("Fetching 1Y price metrics..."):
        pm = fetch_1y_price_metrics(pf_tickers)

    with st.spinner("Fetching next earnings dates (cached)..."):
        ed = fetch_earnings_for_list(pf_tickers)

    ov = pm.merge(ed, on="Ticker", how="left")
    ov["Next Earnings (UTC)"] = pd.to_datetime(ov["Next Earnings (UTC)"], utc=True, errors="coerce")
    ov["Company"] = ov["Ticker"].apply(lambda t: meta.loc[t, "Company"] if t in meta.index else "")
    ov["Sector"] = ov["Ticker"].apply(lambda t: meta.loc[t, "Sector"] if t in meta.index else "")
    ov["Industry"] = ov["Ticker"].apply(lambda t: meta.loc[t, "Industry"] if t in meta.index else "")

    if pf_only_next_earnings:
        ov = ov[ov["Next Earnings (UTC)"].notna()].reset_index(drop=True)

    disp = ov.copy()
    disp["Next Earnings (UTC)"] = disp["Next Earnings (UTC)"].dt.strftime("%Y-%m-%d")
    disp["Current"] = disp["Current"].apply(lambda x: _fmt_num(x, 2))
    disp["52W High"] = disp["52W High"].apply(lambda x: _fmt_num(x, 2))
    disp["52W Low"] = disp["52W Low"].apply(lambda x: _fmt_num(x, 2))
    for c in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
        disp[c] = disp[c].apply(lambda x: _fmt_pct(x, 1))

    disp = disp.sort_values("Next Earnings (UTC)", ascending=True, na_position="last")

    st.dataframe(
        disp[
            [
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Next Earnings (UTC)",
                "Current",
                "Δ vs 52W High (%)",
                "Δ vs 52W Low (%)",
                "52W Range (%)",
            ]
        ],
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
    st.subheader("🧾 Past 4 Earnings (per holding)")

    for tk in pf_tickers:
        company = meta.loc[tk, "Company"] if tk in meta.index else ""
        exp_title = f"{tk} — {company}" if company else tk
        with st.expander(exp_title, expanded=False):
            out = portfolio_last4_rows(tk)
            if out.empty:
                st.info("No past earnings events returned by Yahoo for this ticker.")
                continue

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
