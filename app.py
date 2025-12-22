import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Earnings Radar (Portfolio)",
    page_icon="📅",
    layout="wide",
)

# -----------------------------
# Constants
# -----------------------------
SP500_UNIVERSE_CSV = "sp500_universe.csv"  # put this in your repo root
FINNHUB_BASE = "https://finnhub.io/api/v1"


# -----------------------------
# Helpers
# -----------------------------
def _to_date(x) -> Optional[date]:
    """Best-effort parse to date (handles common Finnhub strings)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Handle ISO timestamps like 2025-01-30T00:00:00.000Z
        s2 = s.replace("T", " ").replace("Z", "")
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
        ):
            try:
                return datetime.strptime(s2, fmt).date()
            except Exception:
                pass
    return None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating)):
            if np.isnan(x):
                return None
            return float(x)
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in ("none", "nan"):
            return None
        return float(s)
    except Exception:
        return None


def _fmt_money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{v/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{v/1e6:.2f}M"
    return f"{v:.0f}"


def _clean_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")  # yfinance uses BRK-B style


def _extract_tickers_from_df(df: pd.DataFrame) -> List[str]:
    """Try common column names first; else use first column."""
    if df is None or df.empty:
        return []
    cols = [c.lower().strip() for c in df.columns]
    ticker_col = None
    for name in ["ticker", "symbol", "symbols", "tickers"]:
        if name in cols:
            ticker_col = df.columns[cols.index(name)]
            break
    if ticker_col is None:
        ticker_col = df.columns[0]
    tickers = df[ticker_col].astype(str).map(_clean_ticker).tolist()
    tickers = [t for t in tickers if t and t not in ("NAN", "NONE")]
    # de-dupe preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@st.cache_data(ttl=24 * 3600)
def load_sp500_universe_map() -> pd.DataFrame:
    """Load S&P 500 mapping if file exists (Ticker -> Company/Sector/Industry)."""
    try:
        df = pd.read_csv(SP500_UNIVERSE_CSV)
        rename = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ("symbol", "ticker"):
                rename[c] = "Ticker"
            elif cl in ("company", "company name", "name"):
                rename[c] = "Company"
            elif cl == "sector":
                rename[c] = "Sector"
            elif cl == "industry":
                rename[c] = "Industry"
        df = df.rename(columns=rename)
        if "Ticker" in df.columns:
            df["Ticker"] = df["Ticker"].astype(str).map(_clean_ticker)
        keep = [c for c in ["Ticker", "Company", "Sector", "Industry"] if c in df.columns]
        df = df[keep].drop_duplicates(subset=["Ticker"]) if "Ticker" in df.columns else df
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Company", "Sector", "Industry"])


def get_finnhub_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        key = None
    if isinstance(key, str):
        key = key.strip()
        if key == "":
            return None
        return key
    return None


def finnhub_get(endpoint: str, params: Dict, api_key: str) -> Tuple[int, dict]:
    url = f"{FINNHUB_BASE}{endpoint}"
    p = dict(params or {})
    p["token"] = api_key
    try:
        r = requests.get(url, params=p, timeout=20)
        status = r.status_code
        try:
            js = r.json()
        except Exception:
            js = {}
        return status, js
    except Exception:
        return 0, {}


@st.cache_data(ttl=6 * 3600)
def finnhub_next_earnings_date(symbol: str, lookahead_days: int, api_key: str) -> Optional[date]:
    """Get the next earnings date from Finnhub earnings calendar within lookahead window."""
    sym = _clean_ticker(symbol)
    today = date.today()
    to_d = today + timedelta(days=int(lookahead_days))
    status, js = finnhub_get(
        "/calendar/earnings",
        {"from": today.isoformat(), "to": to_d.isoformat(), "symbol": sym},
        api_key=api_key,
    )
    if status != 200 or not isinstance(js, dict):
        return None
    data = js.get("earningsCalendar", [])
    if not isinstance(data, list) or len(data) == 0:
        return None

    dates = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if _clean_ticker(item.get("symbol", "")) != sym:
            continue
        d = _to_date(item.get("date"))
        if d is not None and d >= today:
            dates.append(d)
    if not dates:
        return None
    return min(dates)


@st.cache_data(ttl=12 * 3600)
def finnhub_stock_earnings(symbol: str, limit: int, api_key: str) -> pd.DataFrame:
    """Past earnings fundamentals (EPS actual/estimate) from Finnhub /stock/earnings."""
    sym = _clean_ticker(symbol)
    status, js = finnhub_get("/stock/earnings", {"symbol": sym, "limit": int(limit)}, api_key=api_key)
    if status != 200 or not isinstance(js, list):
        return pd.DataFrame()

    rows = []
    for item in js:
        if not isinstance(item, dict):
            continue
        # fields: period, actual, estimate, surprise, surprisePercent, year, quarter
        rows.append(
            {
                "Year": item.get("year"),
                "Quarter": item.get("quarter"),
                "Period": item.get("period"),
                "EPS Actual": _safe_float(item.get("actual")),
                "EPS Est.": _safe_float(item.get("estimate")),
                "Surprise": _safe_float(item.get("surprise")),
                "Surprise %": _safe_float(item.get("surprisePercent")),
            }
        )
    df = pd.DataFrame(rows)
    return df


@st.cache_data(ttl=12 * 3600)
def finnhub_past_earnings_dates_from_calendar(symbol: str, api_key: str, limit: int = 4) -> pd.DataFrame:
    """
    Get past earnings EVENT dates via Finnhub calendar reliably by paging backwards
    in smaller windows (some accounts/APIs return empty if you request huge ranges).
    """
    sym = _clean_ticker(symbol)
    today = date.today()

    # Keep windows small to avoid API limitations (works more reliably than 900-day in one shot)
    window_days = 120
    max_windows = 10  # 10*120 ~ 1200 days back

    events: List[dict] = []
    seen_keys = set()

    end = today
    for _ in range(max_windows):
        start = end - timedelta(days=window_days)
        status, js = finnhub_get(
            "/calendar/earnings",
            {"from": start.isoformat(), "to": end.isoformat(), "symbol": sym},
            api_key=api_key,
        )
        if status == 200 and isinstance(js, dict):
            items = js.get("earningsCalendar", [])
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    if _clean_ticker(it.get("symbol", "")) != sym:
                        continue
                    d = _to_date(it.get("date"))
                    if d is None or d > today:
                        continue
                    y = it.get("year")
                    q = it.get("quarter")
                    key = (d, y, q)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    events.append(
                        {
                            "Earnings Date": d,
                            "Year": y,
                            "Quarter": q,
                            "EPS Actual": _safe_float(it.get("epsActual")),
                            "EPS Est.": _safe_float(it.get("epsEstimate")),
                            "Surprise": _safe_float(it.get("epsSurprise")),
                            "Surprise %": _safe_float(it.get("epsSurprisePercent")),
                        }
                    )

        # move window back
        end = start - timedelta(days=1)

        # early stop if we already have enough events
        if len(events) >= limit:
            break

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    df = df.sort_values("Earnings Date", ascending=False).head(int(limit)).reset_index(drop=True)
    return df


def merge_earnings_dates_with_eps(symbol: str, api_key: str, limit: int = 4) -> pd.DataFrame:
    """
    Goal:
      - Earnings Date: from Finnhub calendar (event date)
      - EPS fields: from calendar if present, else backfill from /stock/earnings by (Year, Quarter).
    """
    dates_df = finnhub_past_earnings_dates_from_calendar(symbol, api_key=api_key, limit=limit)

    # If calendar has nothing, fall back to /stock/earnings (still shows “past 4 quarters”)
    if dates_df is None or dates_df.empty:
        se = finnhub_stock_earnings(symbol, limit=limit, api_key=api_key)
        if se is None or se.empty:
            return pd.DataFrame()
        # Use Period as the date-ish field (not perfect, but better than nothing)
        se = se.copy()
        se["Earnings Date"] = se["Period"].apply(_to_date)
        se = se[["Earnings Date", "Year", "Quarter", "Period", "EPS Actual", "EPS Est.", "Surprise", "Surprise %"]]
        se = se.sort_values("Earnings Date", ascending=False).head(int(limit)).reset_index(drop=True)
        return se

    # Backfill EPS from /stock/earnings when calendar EPS fields are missing
    se = finnhub_stock_earnings(symbol, limit=12, api_key=api_key)
    se_map = {}
    if se is not None and not se.empty:
        for _, r in se.iterrows():
            se_map[(r.get("Year"), r.get("Quarter"))] = r

    out_rows = []
    for _, r in dates_df.iterrows():
        y, q = r.get("Year"), r.get("Quarter")
        row = dict(r)

        # If any EPS fields missing, try /stock/earnings match
        need = any(row.get(k) is None or (isinstance(row.get(k), float) and np.isnan(row.get(k)))
                   for k in ["EPS Actual", "EPS Est.", "Surprise", "Surprise %"])
        if need and (y, q) in se_map:
            rr = se_map[(y, q)]
            row["EPS Actual"] = row.get("EPS Actual") if row.get("EPS Actual") is not None else rr.get("EPS Actual")
            row["EPS Est."] = row.get("EPS Est.") if row.get("EPS Est.") is not None else rr.get("EPS Est.")
            row["Surprise"] = row.get("Surprise") if row.get("Surprise") is not None else rr.get("Surprise")
            row["Surprise %"] = row.get("Surprise %") if row.get("Surprise %") is not None else rr.get("Surprise %")

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out = out.sort_values("Earnings Date", ascending=False).head(int(limit)).reset_index(drop=True)
    return out


@st.cache_data(ttl=24 * 3600)
def yf_fast_info(ticker: str) -> Dict:
    """Fast info per ticker (market cap etc). Cached heavily."""
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi is None:
            return {}
        return dict(fi)
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600)
def yf_history_1y_single(ticker: str) -> pd.DataFrame:
    """1Y daily history fallback (fills missing Current/52W fields)."""
    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            threads=False,
            progress=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index).date
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=6 * 3600)
def yf_history_2y_single(ticker: str) -> pd.DataFrame:
    """2Y daily history for one ticker (for reactions around last 4 earnings)."""
    try:
        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            threads=False,
            progress=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index).date
        return df
    except Exception:
        return pd.DataFrame()


def _get_current_high_low_52w(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Robust: try yfinance fast_info first, then fallback to 1Y history.
    """
    fi = yf_fast_info(ticker)

    current = _safe_float(fi.get("last_price") or fi.get("lastPrice") or fi.get("regular_market_price"))
    high52 = _safe_float(fi.get("year_high") or fi.get("yearHigh"))
    low52 = _safe_float(fi.get("year_low") or fi.get("yearLow"))

    if current is not None and high52 is not None and low52 is not None:
        return current, high52, low52

    hist = yf_history_1y_single(ticker)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return current, high52, low52

    close = hist["Close"].dropna()
    if close.empty:
        return current, high52, low52

    current = current if current is not None else float(close.iloc[-1])
    high52 = high52 if high52 is not None else float(close.max())
    low52 = low52 if low52 is not None else float(close.min())
    return current, high52, low52


def compute_reaction_from_history(hist: pd.DataFrame, event_date: date) -> Dict:
    """
    Given daily history indexed by date, compute:
    - pre_close: last close strictly before event_date
    - post_close_1d: first close strictly after event_date
    - post_close_3d: third trading close strictly after event_date
    """
    out = {
        "Pre Close Date": None,
        "Pre Close": None,
        "Post Close (1D) Date": None,
        "Post Close (1D)": None,
        "1D Reaction %": None,
        "Post Close (3D) Date": None,
        "Post Close (3D)": None,
        "3D Reaction %": None,
    }
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out

    idx = sorted(list(hist.index))
    closes = hist["Close"].to_dict()

    pre_dates = [d for d in idx if d < event_date and d in closes and pd.notna(closes[d])]
    post_dates = [d for d in idx if d > event_date and d in closes and pd.notna(closes[d])]

    if not pre_dates or not post_dates:
        return out

    pre_d = pre_dates[-1]
    post1_d = post_dates[0]
    out["Pre Close Date"] = pre_d
    out["Pre Close"] = float(closes[pre_d])
    out["Post Close (1D) Date"] = post1_d
    out["Post Close (1D)"] = float(closes[post1_d])

    if out["Pre Close"] and out["Post Close (1D)"]:
        out["1D Reaction %"] = (out["Post Close (1D)"] / out["Pre Close"] - 1.0) * 100.0

    if len(post_dates) >= 3:
        post3_d = post_dates[2]
        out["Post Close (3D) Date"] = post3_d
        out["Post Close (3D)"] = float(closes[post3_d])
        if out["Pre Close"] and out["Post Close (3D)"]:
            out["3D Reaction %"] = (out["Post Close (3D)"] / out["Pre Close"] - 1.0) * 100.0

    return out


# -----------------------------
# UI: Header + Upload (visible)
# -----------------------------
st.title("📅 Earnings Radar (Portfolio)")
st.caption(
    "Upload your holdings (CSV/XLSX) or paste tickers. "
    "Next earnings + past 4 earnings event dates (Finnhub). "
    "Market cap + price + 52-week stats + reactions (Yahoo Finance via yfinance)."
)

api_key = get_finnhub_key()
if api_key is None:
    st.error("Missing FINNHUB_API_KEY in Streamlit Secrets. Add it and rerun.")
    st.stop()

col_u1, col_u2 = st.columns([2, 3], vertical_alignment="top")
with col_u1:
    uploaded = st.file_uploader(
        "Upload holdings file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Must contain a ticker column (Ticker/Symbol) or have tickers in the first column.",
    )

with col_u2:
    pasted = st.text_input(
        "No file? Paste tickers (comma-separated)",
        placeholder="AAPL, MSFT, NVDA",
    )

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")
lookahead_days = st.sidebar.slider("Next earnings lookahead (days)", 30, 365, 180, 10)
max_tickers = st.sidebar.slider("Max tickers to process", 5, 300, 60, 5)

only_with_upcoming = st.sidebar.checkbox("Show only holdings with an upcoming earnings date", value=True)
within_days = st.sidebar.slider("Upcoming earnings within (days)", 1, min(lookahead_days, 365), 60, 1)

# IMPORTANT: gating heavy per-ticker 2Y history calls
compute_reactions = st.sidebar.checkbox("Compute price reaction (1D/3D) in Past Earnings section (slower)", value=True)

st.sidebar.caption("Tip: fewer tickers = faster. Caching helps a lot after the first run.")
run = st.sidebar.button("🚀 Run Earnings Radar", use_container_width=True)

# -----------------------------
# Load tickers
# -----------------------------
tickers: List[str] = []
holdings_df = None

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            holdings_df = pd.read_csv(uploaded)
        else:
            holdings_df = pd.read_excel(uploaded)
        tickers = _extract_tickers_from_df(holdings_df)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if pasted and pasted.strip():
    pasted_list = [_clean_ticker(x) for x in pasted.split(",") if x.strip()]
    seen = set(tickers)
    for t in pasted_list:
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)

tickers = tickers[: int(max_tickers)]

if tickers:
    st.success(f"Loaded {len(tickers)} unique tickers.")
else:
    st.info("Upload a file or paste tickers to begin.")
    st.stop()

# -----------------------------
# Main compute
# -----------------------------
if run:
    sp500_map = load_sp500_universe_map()
    sp500_lookup = {}
    if not sp500_map.empty and "Ticker" in sp500_map.columns:
        sp500_lookup = sp500_map.set_index("Ticker").to_dict(orient="index")

    rows = []
    progress = st.progress(0, text="Fetching earnings + price stats...")
    total = len(tickers)

    for i, t in enumerate(tickers):
        # SP500 metadata
        meta = sp500_lookup.get(t, {})
        company = meta.get("Company", None)
        sector = meta.get("Sector", None)
        industry = meta.get("Industry", None)

        # Next earnings from Finnhub
        next_e = finnhub_next_earnings_date(t, lookahead_days=lookahead_days, api_key=api_key)
        days_to = (next_e - date.today()).days if next_e is not None else None

        # yfinance: market cap + current + 52W (robust)
        fi = yf_fast_info(t)
        mcap = _safe_float(fi.get("market_cap")) or _safe_float(fi.get("marketCap"))
        current, high52, low52 = _get_current_high_low_52w(t)

        # Fallback company/sector/industry from yfinance if missing
        if not company or not sector or not industry:
            try:
                info = yf.Ticker(t).get_info()
                company = company or info.get("shortName") or info.get("longName")
                sector = sector or info.get("sector")
                industry = industry or info.get("industry")
            except Exception:
                pass

        # Derived %s
        d_vs_high = None
        d_vs_low = None
        r52 = None
        if current is not None and high52 is not None and high52 != 0:
            d_vs_high = (current - high52) / high52 * 100.0
        if current is not None and low52 is not None and low52 != 0:
            d_vs_low = (current - low52) / low52 * 100.0
        if high52 is not None and low52 is not None and low52 != 0:
            r52 = (high52 - low52) / low52 * 100.0

        rows.append(
            {
                "Ticker": t,
                "Company": company,
                "Sector": sector,
                "Industry": industry,
                "Market Cap": mcap,
                "Next Earnings": next_e,
                "Days to Earnings": days_to,
                "Current": current,
                "52W High": high52,
                "52W Low": low52,
                "Δ vs 52W High (%)": d_vs_high,
                "Δ vs 52W Low (%)": d_vs_low,
                "52W Range (%)": r52,
            }
        )

        progress.progress((i + 1) / total, text=f"Fetched {i+1}/{total}: {t}")

    progress.empty()

    overview = pd.DataFrame(rows)

    # Filtering
    filtered = overview.copy()
    if only_with_upcoming:
        filtered = filtered[filtered["Next Earnings"].notna()].copy()

    if not filtered.empty:
        filtered = filtered[filtered["Days to Earnings"].notna()].copy()
        filtered = filtered[(filtered["Days to Earnings"] >= 0) & (filtered["Days to Earnings"] <= within_days)].copy()

    # Sort
    if not filtered.empty:
        filtered = filtered.sort_values(["Days to Earnings", "Ticker"], ascending=[True, True])

    # -----------------------------
    # Display: Portfolio Overview
    # -----------------------------
    st.subheader("2) Portfolio Overview")

    if filtered.empty:
        st.warning("No holdings matched your filters (try turning off 'Show only holdings with an upcoming earnings date').")
    else:
        view = filtered.copy()
        view["Market Cap"] = view["Market Cap"].apply(_fmt_money)
        view["Next Earnings"] = view["Next Earnings"].astype("object")

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Next Earnings": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Current": st.column_config.NumberColumn(format="%.2f"),
                "52W High": st.column_config.NumberColumn(format="%.2f"),
                "52W Low": st.column_config.NumberColumn(format="%.2f"),
                "Δ vs 52W High (%)": st.column_config.NumberColumn(format="%.1f"),
                "Δ vs 52W Low (%)": st.column_config.NumberColumn(format="%.1f"),
                "52W Range (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        csv_bytes = view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Portfolio Overview (CSV)",
            data=csv_bytes,
            file_name="earnings_radar_portfolio_overview.csv",
            mime="text/csv",
        )

    # -----------------------------
    # Past 4 earnings + price reaction
    # -----------------------------
    st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")
    st.caption(
        "Earnings event dates are pulled from Finnhub calendar (paged in small windows for reliability). "
        "EPS data is backfilled from Finnhub /stock/earnings if needed. "
        "Price reaction uses daily closes from Yahoo Finance (yfinance)."
    )

    # iterate ORIGINAL tickers list (not filtered)
    row_by_ticker = {r["Ticker"]: r for r in rows}

    for t in tickers:
        company = (row_by_ticker.get(t) or {}).get("Company", None)
        label = f"{t}" + (f" — {company}" if company else "")

        with st.expander(label, expanded=False):
            # IMPORTANT: past 4 quarters must show again
            earn_df = merge_earnings_dates_with_eps(t, api_key=api_key, limit=4)

            if earn_df is None or earn_df.empty:
                st.info("No earnings history returned (common for ETFs/funds or incomplete listings).")
                continue

            # If user disabled reactions, just show the table
            if not compute_reactions:
                st.dataframe(
                    earn_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Earnings Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                        "EPS Actual": st.column_config.NumberColumn(format="%.3f"),
                        "EPS Est.": st.column_config.NumberColumn(format="%.3f"),
                        "Surprise": st.column_config.NumberColumn(format="%.3f"),
                        "Surprise %": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
                continue

            hist2y = yf_history_2y_single(t)
            if hist2y is None or hist2y.empty:
                st.warning("Could not fetch 2Y daily price history from Yahoo Finance for reactions.")
                st.dataframe(earn_df, use_container_width=True, hide_index=True)
                continue

            enriched = []
            for _, row in earn_df.iterrows():
                ed = row.get("Earnings Date")
                if isinstance(ed, pd.Timestamp):
                    ed = ed.date()
                if not isinstance(ed, date):
                    ed = _to_date(ed)

                base = dict(row)
                if ed is None:
                    enriched.append(base)
                    continue

                base.update(compute_reaction_from_history(hist2y, ed))
                enriched.append(base)

            out_df = pd.DataFrame(enriched)

            # clean order (keep Period if it exists from fallback)
            keep_cols = [
                "Earnings Date",
                "Year",
                "Quarter",
                "Period",
                "EPS Actual",
                "EPS Est.",
                "Surprise",
                "Surprise %",
                "Pre Close Date",
                "Pre Close",
                "Post Close (1D) Date",
                "Post Close (1D)",
                "1D Reaction %",
                "Post Close (3D) Date",
                "Post Close (3D)",
                "3D Reaction %",
            ]
            keep_cols = [c for c in keep_cols if c in out_df.columns]
            out_df = out_df[keep_cols].copy()

            st.dataframe(
                out_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Earnings Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Pre Close Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Post Close (1D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Post Close (3D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "EPS Actual": st.column_config.NumberColumn(format="%.3f"),
                    "EPS Est.": st.column_config.NumberColumn(format="%.3f"),
                    "Surprise": st.column_config.NumberColumn(format="%.3f"),
                    "Surprise %": st.column_config.NumberColumn(format="%.1f"),
                    "Pre Close": st.column_config.NumberColumn(format="%.2f"),
                    "Post Close (1D)": st.column_config.NumberColumn(format="%.2f"),
                    "Post Close (3D)": st.column_config.NumberColumn(format="%.2f"),
                    "1D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                    "3D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                },
            )
else:
    st.info("Click **Run Earnings Radar** in the sidebar to fetch data.")
