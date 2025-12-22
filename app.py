import math
from datetime import datetime, timezone, timedelta
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

st.title("📡 Earnings Radar (S&P 500)")
st.caption(
    "Sector/Industry comes from your CSV (stable). Prices & earnings dates come from Yahoo via yfinance "
    "(earnings calls are cached and limited for speed)."
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


def _fmt_mcap(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    if v >= 1e12:
        return f"{v/1e12:.2f}T"
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    return f"{v:,.0f}"


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


def today_utc_date() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _normalize_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


# -----------------------------------------------------------------------------
# Universe loading
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
    Uses yf.download in chunks.
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
# Market caps (optional)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_market_caps(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for batch in chunk_list(tickers, 60):
        for tk in batch:
            mcap = None
            try:
                t = yf.Ticker(tk)
                try:
                    fast = getattr(t, "fast_info", None)
                    if fast is not None:
                        mcap = _safe_float(getattr(fast, "market_cap", None))
                except Exception:
                    mcap = None

                if mcap is None:
                    info = {}
                    try:
                        info = t.info or {}
                    except Exception:
                        info = {}
                    mcap = _safe_float(info.get("marketCap"))
            except Exception:
                mcap = None
            rows.append({"Ticker": tk, "Market Cap": mcap})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Earnings (slowest; cached; limited)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_next_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    now = today_utc_date()

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


def fetch_earnings_for_list(tickers: List[str], max_n: int) -> pd.DataFrame:
    tickers = tickers[:max_n]
    bar = st.progress(0.0)
    rows = []
    n = len(tickers)
    for i, tk in enumerate(tickers, start=1):
        dt = fetch_next_earnings_date(tk)
        rows.append({"Ticker": tk, "Next Earnings (UTC)": dt})
        bar.progress(i / max(1, n))
    bar.empty()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Portfolio: past 4 quarters earnings + price context + consolidated table
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


def build_portfolio_section(pf_tickers: List[str], universe: pd.DataFrame, max_names: int) -> None:
    if not pf_tickers:
        st.info("No tickers found in uploaded portfolio.")
        return

    pf_tickers = list(dict.fromkeys([_normalize_ticker(t) for t in pf_tickers if str(t).strip()]))
    if len(pf_tickers) > max_names:
        st.warning(f"Portfolio has {len(pf_tickers)} tickers — showing first {max_names} for speed.")
        pf_tickers = pf_tickers[:max_names]

    # consolidated table (one row per ticker)
    st.subheader("📦 Portfolio Overview (next earnings + price context)")
    st.caption("One row per holding. Then expand below for past 4 earnings.")

    meta = universe[["Ticker", "Company", "Sector", "Industry"]].drop_duplicates(subset=["Ticker"])
    m = meta.set_index("Ticker")

    # prices for context
    with st.spinner("Fetching 1Y price metrics for portfolio tickers..."):
        pm = fetch_1y_price_metrics(pf_tickers)
    df = pm.copy()

    # earnings next
    with st.spinner("Fetching next earnings dates (cached) for portfolio tickers..."):
        ed = fetch_earnings_for_list(pf_tickers, max_n=len(pf_tickers))
    df = df.merge(ed, on="Ticker", how="left")
    df["Next Earnings (UTC)"] = pd.to_datetime(df["Next Earnings (UTC)"], utc=True, errors="coerce")

    # attach meta
    df["Company"] = df["Ticker"].apply(lambda t: m.loc[t, "Company"] if t in m.index else "")
    df["Sector"] = df["Ticker"].apply(lambda t: m.loc[t, "Sector"] if t in m.index else "")
    df["Industry"] = df["Ticker"].apply(lambda t: m.loc[t, "Industry"] if t in m.index else "")

    # display
    disp = df.copy()
    disp["Next Earnings (UTC)"] = disp["Next Earnings (UTC)"].dt.strftime("%Y-%m-%d")
    disp["Current"] = disp["Current"].apply(lambda x: _fmt_num(x, 2))
    disp["52W High"] = disp["52W High"].apply(lambda x: _fmt_num(x, 2))
    disp["52W Low"] = disp["52W Low"].apply(lambda x: _fmt_num(x, 2))
    for c in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
        disp[c] = disp[c].apply(lambda x: _fmt_pct(x, 1))

    st.dataframe(
        disp[
            ["Ticker", "Company", "Sector", "Industry", "Next Earnings (UTC)", "Current", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]
        ].sort_values("Next Earnings (UTC)", ascending=True, na_position="last"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("📦 Portfolio: Past 4 Earnings (with event + price context)")
    with st.popover("ℹ️ What does “price context” mean here?"):
        st.markdown(
            """
For each of the last 4 earnings events we show:
- **Event Close (prev)**: last trading close on/before the earnings date  
- **1D / 5D Move**: close-to-close reaction after that point (approx proxy for earnings reaction)  
- **Dist vs 52W High/Low**: where the stock was vs its trailing 252-trading-day high/low *at that time*
            """.strip()
        )

    for tk in pf_tickers:
        company = m.loc[tk, "Company"] if tk in m.index else ""
        sector = m.loc[tk, "Sector"] if tk in m.index else ""
        industry = m.loc[tk, "Industry"] if tk in m.index else ""

        exp_title = f"{tk} — {company}" if company else tk
        with st.expander(exp_title, expanded=False):
            st.caption(f"{sector} • {industry}".strip(" •"))

            out = portfolio_last4_rows(tk)
            if out.empty:
                st.info("No past earnings events returned by Yahoo for this ticker.")
                continue

            # format
            disp2 = out.copy()
            # numeric formatting
            for col in ["EPS Estimate", "Reported EPS", "Event Close (prev)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_num(x, 2))
            for col in ["Surprise(%)", "1D Move (%)", "5D Move (%)", "Dist vs 52W High (%)", "Dist vs 52W Low (%)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_pct(x, 1))

            # column order
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


# -----------------------------------------------------------------------------
# Small chart: earnings counts by day
# -----------------------------------------------------------------------------
def earnings_count_chart(dates: pd.Series) -> Optional[alt.Chart]:
    s = pd.to_datetime(dates, utc=True, errors="coerce").dropna()
    if s.empty:
        return None
    d = pd.DataFrame({"date": s.dt.date})
    agg = d.groupby("date").size().reset_index(name="count")
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Earnings date (UTC)"),
            y=alt.Y("count:Q", title="Number of companies"),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=220)
    )


# -----------------------------------------------------------------------------
# Load universe
# -----------------------------------------------------------------------------
try:
    universe = load_universe_csv("sp500_universe.csv")
except Exception as e:
    st.error(str(e))
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Universe Filters")

    sectors = sorted([s for s in universe["Sector"].dropna().unique().tolist() if s])
    sel_sectors = st.multiselect("Sector", sectors, default=sectors)

    ind_pool = universe[universe["Sector"].isin(sel_sectors)]["Industry"].dropna().unique().tolist()
    industries = sorted([i for i in ind_pool if i])
    sel_industries = st.multiselect("Industry", industries, default=industries)

    st.divider()

    st.header("Radar Features")
    include_earnings = st.checkbox("Include next earnings date", value=True)
    earnings_window_days = st.slider("Only show earnings within (days)", 0, 90, 21, 1) if include_earnings else 21
    max_earnings_fetch = st.slider(
        "Max tickers to fetch earnings for",
        25, 500, 200, 25,
        help="Earnings calls are slow. This cap keeps the app usable."
    ) if include_earnings else 0

    st.divider()

    st.header("Near-High / Near-Low Filters")
    near_high_on = st.checkbox("Only near 52W High", value=False)
    near_high_pct = st.slider("Within X% of 52W High", 1, 50, 10, 1, disabled=not near_high_on)
    near_low_on = st.checkbox("Only near 52W Low", value=False)
    near_low_pct = st.slider("Within X% of 52W Low", 1, 100, 15, 1, disabled=not near_low_on)
    st.caption("Near-high: Δ vs 52W High between 0% and -X%. Near-low: Δ vs 52W Low between 0% and +X%.")

    st.divider()

    apply_mcap = st.checkbox("Enable market cap filter (slower)", value=False)
    if apply_mcap:
        min_mcap_b, max_mcap_b = st.slider(
            "Market cap range ($B)",
            min_value=0.0,
            max_value=10000.0,
            value=(5.0, 10000.0),
            step=5.0,
        )
    else:
        min_mcap_b, max_mcap_b = 0.0, 10000.0

    st.divider()

    st.header("Portfolio (optional)")
    portfolio_file = st.file_uploader("Upload portfolio CSV (Ticker column)", type=["csv"])
    show_portfolio = st.checkbox("Show portfolio overview + past 4 earnings", value=False)
    portfolio_max_names = st.slider("Max portfolio tickers to process", 5, 80, 30, 5, disabled=not show_portfolio)

    st.divider()
    rows_to_show = st.slider("Rows to show", 25, 500, 150, 25)

    st.divider()
    run = st.button("Run Radar", type="primary")


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if not run:
    st.info("Set your filters and click **Run Radar**.")
    st.stop()

# Filter universe list (fast)
u = universe.copy()
u = u[u["Sector"].isin(sel_sectors)]
u = u[u["Industry"].isin(sel_industries)]
tickers = u["Ticker"].tolist()

if not tickers:
    st.warning("No tickers match those Sector/Industry filters.")
    st.stop()

st.write(f"Tickers after Sector/Industry filters: **{len(tickers)}**")

# Price metrics (fast-ish)
with st.spinner(f"Fetching 1Y price metrics for {len(tickers)} tickers..."):
    pm = fetch_1y_price_metrics(tickers)
df = u.merge(pm, on="Ticker", how="left")

# Near-high / near-low filters
if near_high_on and not df.empty:
    v = pd.to_numeric(df["Δ vs 52W High (%)"], errors="coerce")
    df = df[(v <= 0.0) & (v >= -float(near_high_pct))].reset_index(drop=True)

if near_low_on and not df.empty:
    v = pd.to_numeric(df["Δ vs 52W Low (%)"], errors="coerce")
    df = df[(v >= 0.0) & (v <= float(near_low_pct))].reset_index(drop=True)

# Optional market caps (slow)
if apply_mcap and not df.empty:
    with st.spinner("Fetching market caps (cached) ..."):
        mcap_df = fetch_market_caps(df["Ticker"].tolist())
    df = df.merge(mcap_df, on="Ticker", how="left")

    cap_min = min_mcap_b * 1e9
    cap_max = max_mcap_b * 1e9

    def cap_ok(x: Any) -> bool:
        v = _safe_float(x)
        if v is None:
            return True
        return cap_min <= v <= cap_max

    df = df[df["Market Cap"].apply(cap_ok)].reset_index(drop=True)
else:
    df["Market Cap"] = np.nan

# Earnings (slowest)
if include_earnings and not df.empty:
    with st.spinner("Fetching next earnings dates (cached) ..."):
        ed = fetch_earnings_for_list(df["Ticker"].tolist(), max_n=int(max_earnings_fetch))
    df = df.merge(ed, on="Ticker", how="left")

    now = today_utc_date()
    end = now + timedelta(days=int(earnings_window_days))
    df["Next Earnings (UTC)"] = pd.to_datetime(df["Next Earnings (UTC)"], utc=True, errors="coerce")
    df = df[(df["Next Earnings (UTC)"].isna()) | ((df["Next Earnings (UTC)"] >= pd.Timestamp(now)) & (df["Next Earnings (UTC)"] <= pd.Timestamp(end)))]
else:
    df["Next Earnings (UTC)"] = pd.NaT

# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows after all filters", int(df.shape[0]))
c2.metric("Earnings included", "Yes" if include_earnings else "No")
c3.metric("With earnings date", int(df["Next Earnings (UTC)"].notna().sum()) if include_earnings else 0)
c4.metric("Market cap filter", "On" if apply_mcap else "Off")

st.divider()

# Earnings cluster chart
if include_earnings:
    st.subheader("Earnings Cluster View")
    chart = earnings_count_chart(df["Next Earnings (UTC)"])
    if chart is None:
        st.info("No earnings dates available for the current selection/window.")
    else:
        st.altair_chart(chart, use_container_width=True)
    st.divider()

# Table controls
s1, s2, s3 = st.columns([0.40, 0.30, 0.30])
with s1:
    sort_by = st.selectbox(
        "Sort by",
        ["Next Earnings (UTC)", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)", "Market Cap", "Ticker"],
        index=0 if include_earnings else 1,
    )
with s2:
    ascending = st.checkbox("Ascending", value=True)
with s3:
    only_with_earnings = st.checkbox("Only rows with earnings date", value=False) if include_earnings else False

df_disp = df.copy()
df_disp["Company Display"] = df_disp["Company"].fillna("").astype(str)
df_disp.loc[df_disp["Company Display"].str.len() == 0, "Company Display"] = df_disp["Ticker"]

if only_with_earnings and include_earnings:
    df_disp = df_disp[df_disp["Next Earnings (UTC)"].notna()].reset_index(drop=True)

df_disp = df_disp.sort_values(sort_by, ascending=ascending, na_position="last").head(int(rows_to_show))

# Display table
table = df_disp[
    [
        "Ticker",
        "Company Display",
        "Sector",
        "Industry",
        "Market Cap",
        "Current",
        "52W High",
        "52W Low",
        "Δ vs 52W High (%)",
        "Δ vs 52W Low (%)",
        "52W Range (%)",
        "Next Earnings (UTC)",
    ]
].copy()

table["Market Cap"] = table["Market Cap"].apply(_fmt_mcap)
for col in ["Current", "52W High", "52W Low"]:
    table[col] = table[col].apply(lambda x: _fmt_num(x, 2))
for col in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
    table[col] = table[col].apply(lambda x: _fmt_pct(x, 1))
table["Next Earnings (UTC)"] = pd.to_datetime(table["Next Earnings (UTC)"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

st.subheader("Radar Table")
st.dataframe(table, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Export")
export_raw = df.copy()
export_raw["Next Earnings (UTC)"] = pd.to_datetime(export_raw["Next Earnings (UTC)"], utc=True, errors="coerce")
st.download_button(
    "⬇️ Download Radar (CSV)",
    export_raw.to_csv(index=False),
    file_name="earnings_radar.csv",
    mime="text/csv",
)

# -----------------------------------------------------------------------------
# Portfolio section (optional)
# -----------------------------------------------------------------------------
if show_portfolio:
    st.divider()
    if portfolio_file is None:
        st.info("Upload a portfolio CSV in the sidebar (must include a `Ticker` column).")
    else:
        try:
            pf = pd.read_csv(portfolio_file)
            if "Ticker" not in pf.columns:
                st.error("Portfolio CSV must include a column named `Ticker`.")
            else:
                pf_tickers = pf["Ticker"].astype(str).apply(_normalize_ticker).dropna().tolist()
                pf_tickers = [t for t in pf_tickers if t]
                build_portfolio_section(pf_tickers, universe=universe, max_names=int(portfolio_max_names))
        except Exception as e:
            st.error(f"Could not read portfolio CSV: {e}")
