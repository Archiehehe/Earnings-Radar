import os
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import finnhub
except Exception as e:
    finnhub = None


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Earnings Radar (Portfolio)",
    page_icon="📅",
    layout="wide",
)

st.title("📅 Earnings Radar — Portfolio Upload")
st.caption(
    "Upload a portfolio (CSV/Excel) and get next earnings dates + past 4 earnings with simple price-reaction context. "
    "Powered by Finnhub (free API key required)."
)

# -----------------------------
# Helpers
# -----------------------------
def get_api_key() -> Optional[str]:
    # Streamlit Cloud: put this into Secrets as FINNHUB_API_KEY
    # Locally: export FINNHUB_API_KEY=...
    key = None
    if "FINNHUB_API_KEY" in st.secrets:
        key = st.secrets["FINNHUB_API_KEY"]
    if not key:
        key = os.getenv("FINNHUB_API_KEY")
    return key


def get_client() -> "finnhub.Client":
    if finnhub is None:
        raise RuntimeError("finnhub-python is not installed. Check requirements.txt.")
    key = get_api_key()
    if not key:
        raise RuntimeError("Missing FINNHUB_API_KEY. Add it in Streamlit Secrets or as an env var.")
    return finnhub.Client(api_key=key)


def symbol_variants(sym: str) -> List[str]:
    """
    Finnhub often accepts dot tickers (BRK.B). Some users upload dash tickers (BRK-B).
    Try both.
    """
    s = (sym or "").strip().upper()
    if not s:
        return []
    vars_ = [s]
    if "." in s:
        vars_.append(s.replace(".", "-"))
    if "-" in s:
        vars_.append(s.replace("-", "."))
    # de-dup while keeping order
    out = []
    for v in vars_:
        if v not in out:
            out.append(v)
    return out


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if np.isnan(x):
                return None
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def to_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return 100.0 * x


def fmt_pct(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:.1f}%"


def fmt_num(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:,.2f}"


def fmt_date(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, date):
        return x.strftime("%Y-%m-%d")
    return str(x)


# -----------------------------
# Rate limit (gentle throttle)
# -----------------------------
def throttle(min_interval_s: float = 0.15) -> None:
    """
    Finnhub free tier is commonly ~60 calls/min. We'll keep a gentle delay.
    Caching reduces calls a lot.
    """
    last = st.session_state.get("_fh_last_call_ts", 0.0)
    now = time.time()
    wait = min_interval_s - (now - last)
    if wait > 0:
        time.sleep(wait)
    st.session_state["_fh_last_call_ts"] = time.time()


# -----------------------------
# Finnhub API wrappers (cached)
# -----------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fh_profile2(symbol: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Returns (profile, used_symbol)
    """
    c = get_client()
    for sym in symbol_variants(symbol):
        try:
            throttle()
            prof = c.company_profile2(symbol=sym)
            if isinstance(prof, dict) and (prof.get("name") or prof.get("ticker") or prof.get("finnhubIndustry")):
                return prof, sym
        except Exception:
            continue
    return None, None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fh_company_earnings(symbol: str, limit: int = 4) -> Tuple[List[dict], Optional[str]]:
    """
    Past earnings surprises (usually quarterly). Returns (list, used_symbol).
    """
    c = get_client()
    for sym in symbol_variants(symbol):
        try:
            throttle()
            rows = c.company_earnings(sym, limit=limit)
            if isinstance(rows, list) and len(rows) > 0:
                return rows, sym
            # if it's a valid response but empty, still accept symbol
            if isinstance(rows, list):
                return [], sym
        except Exception:
            continue
    return [], None


@st.cache_data(ttl=2 * 60 * 60, show_spinner=False)
def fh_earnings_calendar_next(symbol: str, lookahead_days: int = 180) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    """
    Try to get the next upcoming earnings date from the earnings calendar.
    """
    c = get_client()
    start = date.today()
    end = start + timedelta(days=int(lookahead_days))

    for sym in symbol_variants(symbol):
        try:
            throttle()
            cal = c.earnings_calendar(_from=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"), symbol=sym, international=False)
            # Finnhub response style: {'earningsCalendar': [...]}
            events = None
            if isinstance(cal, dict):
                events = cal.get("earningsCalendar")
            if isinstance(events, list) and events:
                # pick the earliest event date
                ds = []
                for ev in events:
                    d = ev.get("date") or ev.get("datetime") or ev.get("epsReportDate")
                    if d:
                        try:
                            ds.append(pd.to_datetime(d).normalize())
                        except Exception:
                            pass
                if ds:
                    return min(ds), sym
            # valid but empty is still ok
            if isinstance(events, list):
                return None, sym
        except Exception:
            continue

    return None, None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fh_daily_candles(symbol: str, years: int = 2) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch daily candles for last N years and return a DF indexed by date.
    Used for current price, 52W highs/lows, and event reaction.
    """
    c = get_client()
    to_ts = int(time.time())
    from_ts = int((datetime.utcnow() - timedelta(days=365 * years)).timestamp())

    for sym in symbol_variants(symbol):
        try:
            throttle()
            res = c.stock_candles(sym, "D", from_ts, to_ts)
            # Typical: {'c':[], 'h':[], 'l':[], 'o':[], 's':'ok', 't':[], 'v':[]}
            if not isinstance(res, dict) or res.get("s") != "ok":
                continue
            ts = res.get("t") or []
            if not ts:
                continue

            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
                    "open": res.get("o", []),
                    "high": res.get("h", []),
                    "low": res.get("l", []),
                    "close": res.get("c", []),
                    "volume": res.get("v", []),
                }
            ).dropna(subset=["date", "close"])

            df = df.sort_values("date").drop_duplicates("date").set_index("date")
            return df, sym
        except Exception:
            continue

    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), None


# -----------------------------
# Analytics
# -----------------------------
def compute_52w_metrics(candles: pd.DataFrame) -> Dict[str, Optional[float]]:
    if candles is None or candles.empty or "close" not in candles.columns:
        return {"current": None, "high_52w": None, "low_52w": None, "vs_52w_high": None, "vs_52w_low": None, "range_52w": None}

    closes = candles["close"].dropna()
    if closes.empty:
        return {"current": None, "high_52w": None, "low_52w": None, "vs_52w_high": None, "vs_52w_low": None, "range_52w": None}

    # Use last 252 trading days for "52W"
    last_52w = closes.tail(252)
    current = float(closes.iloc[-1])
    high_52w = float(last_52w.max()) if not last_52w.empty else None
    low_52w = float(last_52w.min()) if not last_52w.empty else None

    vs_high = None
    vs_low = None
    rng = None
    if high_52w and high_52w != 0:
        vs_high = (current - high_52w) / high_52w
    if low_52w and low_52w != 0:
        vs_low = (current - low_52w) / low_52w
        if high_52w is not None:
            rng = (high_52w - low_52w) / low_52w

    return {
        "current": current,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "vs_52w_high": to_percent(vs_high),
        "vs_52w_low": to_percent(vs_low),
        "range_52w": to_percent(rng),
    }


def price_reaction_from_candles(candles: pd.DataFrame, event_date: pd.Timestamp) -> Dict[str, Optional[float]]:
    """
    Compute simple post-earnings reaction using daily closes:
      pre_close = last close strictly BEFORE event_date
      post_close_1d = first close strictly AFTER event_date
      post_close_3d = close 3 trading days AFTER event_date

    This approximates "after-hours earnings → next trading day reaction" pretty well.
    """
    out = {
        "pre_date": None,
        "pre_close": None,
        "post1_date": None,
        "post1_close": None,
        "ret_1d": None,
        "post3_date": None,
        "post3_close": None,
        "ret_3d": None,
    }
    if candles is None or candles.empty:
        return out

    idx = candles.index
    if not isinstance(event_date, pd.Timestamp):
        try:
            event_date = pd.to_datetime(event_date).normalize()
        except Exception:
            return out
    else:
        event_date = event_date.normalize()

    # pre: last date < event_date
    pre_mask = idx < event_date
    if not pre_mask.any():
        return out
    pre_date = idx[pre_mask][-1]
    pre_close = safe_float(candles.loc[pre_date, "close"])

    # post1: first date > event_date
    post_mask = idx > event_date
    if not post_mask.any():
        out.update({"pre_date": pre_date, "pre_close": pre_close})
        return out

    post_dates = idx[post_mask]
    post1_date = post_dates[0]
    post1_close = safe_float(candles.loc[post1_date, "close"])

    ret_1d = None
    if pre_close and post1_close:
        ret_1d = (post1_close - pre_close) / pre_close

    # post3: 3 trading days after event_date (3rd element)
    post3_date = None
    post3_close = None
    ret_3d = None
    if len(post_dates) >= 3:
        post3_date = post_dates[2]
        post3_close = safe_float(candles.loc[post3_date, "close"])
        if pre_close and post3_close:
            ret_3d = (post3_close - pre_close) / pre_close

    out.update(
        {
            "pre_date": pre_date,
            "pre_close": pre_close,
            "post1_date": post1_date,
            "post1_close": post1_close,
            "ret_1d": to_percent(ret_1d),
            "post3_date": post3_date,
            "post3_close": post3_close,
            "ret_3d": to_percent(ret_3d),
        }
    )
    return out


# -----------------------------
# Portfolio parsing
# -----------------------------
def load_portfolio(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload CSV or Excel.")
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    return df


def extract_tickers(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    cols = [c.lower() for c in df.columns]
    ticker_col = None
    for candidate in ["ticker", "symbol", "symbols", "tickers"]:
        if candidate in cols:
            ticker_col = df.columns[cols.index(candidate)]
            break
    if ticker_col is None:
        # fallback: first column
        ticker_col = df.columns[0]

    tickers = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
        .dropna()
        .tolist()
    )

    # de-dup order-preserving
    out = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


# -----------------------------
# UI: Upload first (very visible)
# -----------------------------
st.subheader("1) Upload your portfolio (CSV or Excel)")
uploaded = st.file_uploader(
    "Upload a portfolio file with a column named Ticker or Symbol (or put tickers in the first column).",
    type=["csv", "xlsx", "xls"],
    help="Examples: Ticker, Symbol. Other columns are ignored for this app.",
)

manual_tickers = []
portfolio_df = None
tickers = []

colA, colB = st.columns([1, 1])
with colA:
    st.markdown("**No file?** Paste tickers (comma-separated):")
    raw = st.text_input("Tickers", value="", placeholder="AAPL, MSFT, NVDA")
    if raw.strip():
        manual_tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

with colB:
    st.info(
        "🔑 **You must set a Finnhub API key** in Streamlit Secrets:\n\n"
        "`FINNHUB_API_KEY = \"your_key\"`\n\n"
        "Finnhub’s earnings calendar + earnings history are much more reliable than Yahoo."
    )

if uploaded is not None:
    try:
        portfolio_df = load_portfolio(uploaded)
        tickers = extract_tickers(portfolio_df)
        st.success(f"Loaded {len(tickers)} unique tickers from file.")
    except Exception as e:
        st.error(f"Could not read portfolio: {e}")

if not tickers and manual_tickers:
    tickers = manual_tickers
    st.success(f"Using {len(tickers)} tickers from manual input.")

if not tickers:
    st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")
lookahead_days = st.sidebar.slider("Next earnings lookahead (days)", 30, 365, 180, 15)
max_holdings = st.sidebar.slider("Max tickers to process", 5, 200, min(60, len(tickers)), 5)
show_only_upcoming = st.sidebar.checkbox("Show only holdings with an upcoming earnings date", value=False)
upcoming_within_days = st.sidebar.slider("Upcoming earnings within (days)", 7, 365, 60, 7)

st.sidebar.divider()
st.sidebar.caption("Tip: If you upload 200 tickers, processing will take longer even with caching.")

run = st.sidebar.button("🚀 Run Earnings Radar", use_container_width=True)

if not run:
    st.stop()


# -----------------------------
# Main processing
# -----------------------------
st.subheader("2) Portfolio Overview")

tickers = tickers[:max_holdings]

progress = st.progress(0, text="Starting…")
rows = []
errors = []

for i, t in enumerate(tickers, start=1):
    try:
        prof, used_sym_prof = fh_profile2(t)
        candles, used_sym_candles = fh_daily_candles(t, years=2)
        next_dt, used_sym_next = fh_earnings_calendar_next(t, lookahead_days=lookahead_days)

        name = None
        sector = None
        industry = None
        if prof:
            name = prof.get("name") or prof.get("ticker")
            # Finnhub profile2 fields vary; best-effort:
            sector = prof.get("gicsSector") or prof.get("sector")
            industry = prof.get("finnhubIndustry") or prof.get("gicsIndustry") or prof.get("industry")

        m = compute_52w_metrics(candles)

        rows.append(
            {
                "Ticker": t,
                "Company": name,
                "Sector": sector,
                "Industry": industry,
                "Next Earnings": next_dt,
                "Current": m["current"],
                "Δ vs 52W High (%)": m["vs_52w_high"],
                "Δ vs 52W Low (%)": m["vs_52w_low"],
                "52W Range (%)": m["range_52w"],
                "_sym_used_profile": used_sym_prof,
                "_sym_used_candles": used_sym_candles,
                "_sym_used_next": used_sym_next,
            }
        )
    except Exception as e:
        errors.append((t, str(e)))

    progress.progress(i / len(tickers), text=f"Processing {i}/{len(tickers)}: {t}")

progress.empty()

overview = pd.DataFrame(rows)

# Clean types
if not overview.empty:
    overview["Next Earnings"] = pd.to_datetime(overview["Next Earnings"], errors="coerce")
    for c in ["Current", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
        overview[c] = pd.to_numeric(overview[c], errors="coerce")

# Filters
filtered = overview.copy()
if show_only_upcoming:
    filtered = filtered.dropna(subset=["Next Earnings"])
if not filtered.empty:
    cutoff = pd.Timestamp(date.today() + timedelta(days=int(upcoming_within_days)))
    filtered["Days to Earnings"] = (filtered["Next Earnings"] - pd.Timestamp(date.today())).dt.days
    filtered = filtered[(filtered["Next Earnings"].isna()) | (filtered["Next Earnings"] <= cutoff)]

# Display
display_df = filtered.copy()
if "Days to Earnings" not in display_df.columns and "Next Earnings" in display_df.columns:
    display_df["Days to Earnings"] = (display_df["Next Earnings"] - pd.Timestamp(date.today())).dt.days

# Column order
cols = [
    "Ticker", "Company", "Sector", "Industry",
    "Next Earnings", "Days to Earnings",
    "Current", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)",
]
cols = [c for c in cols if c in display_df.columns]

st.dataframe(
    display_df[cols].sort_values(["Next Earnings", "Ticker"], na_position="last"),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Next Earnings": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
        "Days to Earnings": st.column_config.NumberColumn(format="%d"),
        "Current": st.column_config.NumberColumn(format="%.2f"),
        "Δ vs 52W High (%)": st.column_config.NumberColumn(format="%.1f"),
        "Δ vs 52W Low (%)": st.column_config.NumberColumn(format="%.1f"),
        "52W Range (%)": st.column_config.NumberColumn(format="%.1f"),
    },
)

# Download
csv_bytes = display_df[cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Portfolio Overview (CSV)",
    data=csv_bytes,
    file_name="earnings_radar_overview.csv",
    mime="text/csv",
)

if errors:
    with st.expander("⚠️ Errors (some tickers may be ETFs/funds or have missing coverage)"):
        for t, msg in errors[:200]:
            st.write(f"**{t}** — {msg}")


# -----------------------------
# Past earnings + reaction
# -----------------------------
st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")

st.caption(
    "This uses Finnhub company earnings history + your 2-year daily candles. "
    "Reaction is measured from the last close **before** the earnings date to the 1st/3rd close **after** it."
)

if overview.empty:
    st.info("No companies processed.")
    st.stop()

# Use the same tickers order
for _, row in overview.iterrows():
    t = row["Ticker"]
    company = row.get("Company") or t

    with st.expander(f"{t} — {company}", expanded=False):
        earnings, used_sym_earn = fh_company_earnings(t, limit=4)
        candles, used_sym_c = fh_daily_candles(t, years=2)

        if not earnings:
            st.info("No earnings history returned for this ticker (often ETFs/funds or incomplete coverage).")
            continue

        # Build earnings table with reactions
        out_rows = []
        for ev in earnings:
            # Finnhub usually provides date like '2024-02-01'
            ev_date_raw = ev.get("date") or ev.get("period")
            try:
                ev_date = pd.to_datetime(ev.get("date")).normalize()
            except Exception:
                ev_date = None

            eps_a = safe_float(ev.get("epsActual"))
            eps_e = safe_float(ev.get("epsEstimate"))
            surprise = safe_float(ev.get("epsSurprise"))
            surprise_pct = safe_float(ev.get("epsSurprisePercent"))

            rx = {}
            if ev_date is not None and not candles.empty:
                rx = price_reaction_from_candles(candles, ev_date)
            else:
                rx = {}

            out_rows.append(
                {
                    "Earnings Date": ev.get("date"),
                    "Quarter": ev.get("period") or ev.get("quarter"),
                    "EPS Actual": eps_a,
                    "EPS Est.": eps_e,
                    "Surprise": surprise,
                    "Surprise %": surprise_pct,
                    "Pre Close Date": rx.get("pre_date"),
                    "Post Close (1D) Date": rx.get("post1_date"),
                    "1D Reaction %": rx.get("ret_1d"),
                    "Post Close (3D) Date": rx.get("post3_date"),
                    "3D Reaction %": rx.get("ret_3d"),
                }
            )

        evdf = pd.DataFrame(out_rows)
        if not evdf.empty:
            # format dates
            for c in ["Pre Close Date", "Post Close (1D) Date", "Post Close (3D) Date"]:
                evdf[c] = pd.to_datetime(evdf[c], errors="coerce")

            st.dataframe(
                evdf.sort_values("Earnings Date", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "EPS Actual": st.column_config.NumberColumn(format="%.2f"),
                    "EPS Est.": st.column_config.NumberColumn(format="%.2f"),
                    "Surprise": st.column_config.NumberColumn(format="%.2f"),
                    "Surprise %": st.column_config.NumberColumn(format="%.1f"),
                    "1D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                    "3D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                    "Pre Close Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                    "Post Close (1D) Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                    "Post Close (3D) Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                },
            )
        else:
            st.info("No earnings rows to display.")


st.divider()
st.caption(
    "If a ticker shows no earnings: it may be an ETF/fund, a non-US listing, or the provider may not cover it. "
    "This app will still show 52-week price context when candles are available."
)
