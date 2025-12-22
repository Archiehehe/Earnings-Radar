import os
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import finnhub
except Exception:
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
    # Streamlit Cloud: Settings -> Secrets
    # Locally: export FINNHUB_API_KEY=...
    key = None
    if hasattr(st, "secrets") and "FINNHUB_API_KEY" in st.secrets:
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
    Finnhub often accepts dot tickers (BRK.B). Some people use dash tickers (BRK-B).
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


# -----------------------------
# Gentle throttle (helps Finnhub free tier)
# -----------------------------
def throttle(min_interval_s: float = 0.20) -> None:
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
def fh_profile2(symbol: str) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """
    Returns (profile, used_symbol, error)
    """
    c = get_client()
    last_err = None
    for sym in symbol_variants(symbol):
        try:
            throttle()
            prof = c.company_profile2(symbol=sym)
            if isinstance(prof, dict) and (prof.get("name") or prof.get("ticker") or prof.get("finnhubIndustry")):
                return prof, sym, None
            # if dict but empty-ish, still accept as no-data (not an error)
            if isinstance(prof, dict):
                return prof, sym, None
        except Exception as e:
            last_err = str(e)
            continue
    return None, None, last_err


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fh_company_earnings(symbol: str, limit: int = 4) -> Tuple[List[dict], Optional[str], Optional[str]]:
    """
    Past earnings surprises. Returns (list, used_symbol, error)
    """
    c = get_client()
    last_err = None
    for sym in symbol_variants(symbol):
        try:
            throttle()
            rows = c.company_earnings(sym, limit=limit)
            if isinstance(rows, list):
                return rows, sym, None
        except Exception as e:
            last_err = str(e)
            continue
    return [], None, last_err


@st.cache_data(ttl=2 * 60 * 60, show_spinner=False)
def fh_earnings_calendar_next(symbol: str, lookahead_days: int = 180) -> Tuple[Optional[pd.Timestamp], Optional[str], Optional[str]]:
    """
    Try to get the next upcoming earnings date from the earnings calendar.
    Returns (next_date, used_symbol, error)
    """
    c = get_client()
    start = date.today()
    end = start + timedelta(days=int(lookahead_days))
    last_err = None

    for sym in symbol_variants(symbol):
        try:
            throttle()
            cal = c.earnings_calendar(
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                symbol=sym,
                international=False,
            )
            events = cal.get("earningsCalendar") if isinstance(cal, dict) else None
            if isinstance(events, list) and events:
                ds = []
                for ev in events:
                    d = ev.get("date") or ev.get("datetime") or ev.get("epsReportDate")
                    if d:
                        try:
                            ds.append(pd.to_datetime(d).normalize())
                        except Exception:
                            pass
                if ds:
                    return min(ds), sym, None
                return None, sym, None
            if isinstance(events, list):
                return None, sym, None
        except Exception as e:
            last_err = str(e)
            continue

    return None, None, last_err


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fh_daily_candles(symbol: str, years: int = 2) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Fetch daily candles for last N years and return a DF indexed by date.
    Returns (df, used_symbol, error)
    """
    c = get_client()
    to_ts = int(time.time())
    from_ts = int((datetime.utcnow() - timedelta(days=365 * years)).timestamp())
    last_err = None

    for sym in symbol_variants(symbol):
        try:
            throttle()
            res = c.stock_candles(sym, "D", from_ts, to_ts)
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
            return df, sym, None
        except Exception as e:
            last_err = str(e)
            continue

    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), None, last_err


# -----------------------------
# Analytics
# -----------------------------
def compute_52w_metrics(candles: pd.DataFrame) -> Dict[str, Optional[float]]:
    out = {
        "current": None,
        "high_52w": None,
        "low_52w": None,
        "vs_52w_high": None,
        "vs_52w_low": None,
        "range_52w": None,
    }
    if candles is None or candles.empty or "close" not in candles.columns:
        return out

    closes = candles["close"].dropna()
    if closes.empty:
        return out

    last_52w = closes.tail(252)  # ~52 weeks trading days
    current = float(closes.iloc[-1])
    high_52w = float(last_52w.max()) if not last_52w.empty else None
    low_52w = float(last_52w.min()) if not last_52w.empty else None

    out["current"] = current
    out["high_52w"] = high_52w
    out["low_52w"] = low_52w

    if high_52w and high_52w != 0:
        out["vs_52w_high"] = to_percent((current - high_52w) / high_52w)
    if low_52w and low_52w != 0:
        out["vs_52w_low"] = to_percent((current - low_52w) / low_52w)
        if high_52w is not None:
            out["range_52w"] = to_percent((high_52w - low_52w) / low_52w)

    return out


def price_reaction_from_candles(candles: pd.DataFrame, event_date: pd.Timestamp) -> Dict[str, Optional[float]]:
    """
    Reaction using daily closes:
      pre_close = last close strictly BEFORE event_date
      post_close_1d = first close strictly AFTER event_date
      post_close_3d = close 3 trading days AFTER event_date
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
    event_date = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event_date):
        return out
    event_date = event_date.normalize()

    pre_mask = idx < event_date
    if not pre_mask.any():
        return out
    pre_date = idx[pre_mask][-1]
    pre_close = safe_float(candles.loc[pre_date, "close"])

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

    out = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


# -----------------------------
# Upload section (big + visible)
# -----------------------------
st.subheader("1) Upload your portfolio (CSV or Excel)")
uploaded = st.file_uploader(
    "Upload a portfolio file with a column named Ticker or Symbol (or put tickers in the first column).",
    type=["csv", "xlsx", "xls"],
    help="Examples: Ticker, Symbol. Other columns are ignored for this app.",
)

manual_tickers = []
portfolio_df = None
tickers: List[str] = []

colA, colB = st.columns([1, 1])
with colA:
    st.markdown("**No file?** Paste tickers (comma-separated):")
    raw = st.text_input("Tickers", value="", placeholder="AAPL, MSFT, NVDA")
    if raw.strip():
        manual_tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

with colB:
    st.info(
        "🔑 **Streamlit Cloud**: Manage app → Settings → Secrets\n\n"
        "Add:\n"
        '`FINNHUB_API_KEY = "your_key"`'
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

# Hard fail early if no key (prevents confusing downstream errors)
if not get_api_key():
    st.error("Missing FINNHUB_API_KEY. Add it in Streamlit Secrets and reload.")
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
st.sidebar.caption("Tip: more tickers = more API calls. Caching helps a lot after the first run.")

run = st.sidebar.button("🚀 Run Earnings Radar", use_container_width=True)

if not run:
    st.stop()


# -----------------------------
# Main processing
# -----------------------------
st.subheader("2) Portfolio Overview")

tickers = tickers[:max_holdings]

progress = st.progress(0, text="Starting…")

# Ensure columns always exist (prevents KeyError even if rows == [])
OVERVIEW_COLS = [
    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "Next Earnings",
    "Current",
    "Δ vs 52W High (%)",
    "Δ vs 52W Low (%)",
    "52W Range (%)",
]

rows: List[Dict] = []
errors: List[Tuple[str, str]] = []

for i, t in enumerate(tickers, start=1):
    # Default row (so table is stable even on errors)
    row_out = {
        "Ticker": t,
        "Company": None,
        "Sector": None,
        "Industry": None,
        "Next Earnings": pd.NaT,
        "Current": np.nan,
        "Δ vs 52W High (%)": np.nan,
        "Δ vs 52W Low (%)": np.nan,
        "52W Range (%)": np.nan,
    }

    try:
        prof, used_prof, err_prof = fh_profile2(t)
        candles, used_candles, err_candles = fh_daily_candles(t, years=2)
        next_dt, used_next, err_next = fh_earnings_calendar_next(t, lookahead_days=lookahead_days)

        # Profile fields are best-effort
        if isinstance(prof, dict) and prof:
            row_out["Company"] = prof.get("name") or prof.get("ticker") or row_out["Company"]
            row_out["Sector"] = prof.get("gicsSector") or prof.get("sector") or row_out["Sector"]
            row_out["Industry"] = (
                prof.get("finnhubIndustry") or prof.get("gicsIndustry") or prof.get("industry") or row_out["Industry"]
            )

        if isinstance(next_dt, (pd.Timestamp, datetime, date)):
            row_out["Next Earnings"] = pd.to_datetime(next_dt, errors="coerce")

        m = compute_52w_metrics(candles)
        row_out["Current"] = m["current"] if m["current"] is not None else np.nan
        row_out["Δ vs 52W High (%)"] = m["vs_52w_high"] if m["vs_52w_high"] is not None else np.nan
        row_out["Δ vs 52W Low (%)"] = m["vs_52w_low"] if m["vs_52w_low"] is not None else np.nan
        row_out["52W Range (%)"] = m["range_52w"] if m["range_52w"] is not None else np.nan

        # Record underlying API issues as warnings (not fatal)
        if err_prof:
            errors.append((t, f"profile2: {err_prof}"))
        if err_candles:
            errors.append((t, f"candles: {err_candles}"))
        if err_next:
            errors.append((t, f"earnings_calendar: {err_next}"))

    except Exception as e:
        errors.append((t, str(e)))

    rows.append(row_out)
    progress.progress(i / len(tickers), text=f"Processing {i}/{len(tickers)}: {t}")

progress.empty()

overview = pd.DataFrame(rows, columns=OVERVIEW_COLS)

# Normalize types
overview["Next Earnings"] = pd.to_datetime(overview["Next Earnings"], errors="coerce")
for c in ["Current", "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
    overview[c] = pd.to_numeric(overview[c], errors="coerce")

# Filters (safe even if everything is NaT)
filtered = overview.copy()
if show_only_upcoming:
    # safe because column always exists now
    filtered = filtered.dropna(subset=["Next Earnings"])

# "Within X days" filter
if not filtered.empty:
    today = pd.Timestamp(date.today())
    cutoff = today + pd.Timedelta(days=int(upcoming_within_days))
    # keep NaT rows unless user enabled show_only_upcoming
    if not show_only_upcoming:
        mask = filtered["Next Earnings"].isna() | (filtered["Next Earnings"] <= cutoff)
    else:
        mask = filtered["Next Earnings"] <= cutoff
    filtered = filtered[mask].copy()

# Add Days to Earnings
today = pd.Timestamp(date.today())
filtered["Days to Earnings"] = (filtered["Next Earnings"] - today).dt.days

# Display
display_cols = [
    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "Next Earnings",
    "Days to Earnings",
    "Current",
    "Δ vs 52W High (%)",
    "Δ vs 52W Low (%)",
    "52W Range (%)",
]

st.dataframe(
    filtered[display_cols].sort_values(["Next Earnings", "Ticker"], na_position="last"),
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

csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Portfolio Overview (CSV)",
    data=csv_bytes,
    file_name="earnings_radar_overview.csv",
    mime="text/csv",
)

if errors:
    with st.expander("⚠️ Errors / Missing Coverage (normal for ETFs/funds / some listings)"):
        # de-dupe identical messages to keep it readable
        seen = set()
        for t, msg in errors:
            key = (t, msg)
            if key in seen:
                continue
            seen.add(key)
            st.write(f"**{t}** — {msg}")


# -----------------------------
# Past earnings + reaction
# -----------------------------
st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")
st.caption(
    "Earnings history comes from Finnhub. Price reaction uses daily candles: "
    "last close before earnings date → first close after (1D) and 3 trading days after (3D)."
)

for _, r in overview.iterrows():
    t = r["Ticker"]
    company = r.get("Company") if pd.notna(r.get("Company")) else None
    label = f"{t} — {company}" if company else t

    with st.expander(label, expanded=False):
        earnings, used_sym_e, err_e = fh_company_earnings(t, limit=4)
        candles, used_sym_c, err_c = fh_daily_candles(t, years=2)

        if err_e:
            st.caption(f"Note: earnings endpoint error: {err_e}")
        if err_c:
            st.caption(f"Note: price candles endpoint error: {err_c}")

        if not earnings:
            st.info("No earnings history returned for this ticker (common for ETFs/funds or incomplete coverage).")
            continue

        out_rows = []
        for ev in earnings:
            ev_date_str = ev.get("date")
            ev_date = pd.to_datetime(ev_date_str, errors="coerce")
            eps_a = safe_float(ev.get("epsActual"))
            eps_e = safe_float(ev.get("epsEstimate"))
            surprise = safe_float(ev.get("epsSurprise"))
            surprise_pct = safe_float(ev.get("epsSurprisePercent"))

            rx = {}
            if pd.notna(ev_date) and candles is not None and not candles.empty:
                rx = price_reaction_from_candles(candles, ev_date)

            out_rows.append(
                {
                    "Earnings Date": ev_date_str,
                    "Period": ev.get("period") or ev.get("quarter"),
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

        # Cast date columns safely
        evdf["Pre Close Date"] = pd.to_datetime(evdf["Pre Close Date"], errors="coerce")
        evdf["Post Close (1D) Date"] = pd.to_datetime(evdf["Post Close (1D) Date"], errors="coerce")
        evdf["Post Close (3D) Date"] = pd.to_datetime(evdf["Post Close (3D) Date"], errors="coerce")

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

st.divider()
st.caption(
    "If a ticker has no earnings: it may be an ETF/fund, a non-US listing, or not covered by the provider. "
    "The app will still show price/52-week context when candles are available."
)
