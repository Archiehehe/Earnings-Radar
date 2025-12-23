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
SP500_UNIVERSE_CSV = "sp500_universe.csv"  # keep in repo root
FINNHUB_BASE = "https://finnhub.io/api/v1"


# -----------------------------
# Ticker normalization (IMPORTANT: Finnhub vs yfinance)
# -----------------------------
def _base_ticker(t: str) -> str:
    return str(t).strip().upper()


def _yf_ticker(base: str) -> str:
    # yfinance prefers BRK-B format
    return base.replace(".", "-")


def _finnhub_ticker(base: str) -> str:
    # Finnhub often uses BRK.B format. Heuristic:
    # if it's like BRK-B or RDS-A => convert last '-' to '.'
    if "." in base:
        return base
    if "-" in base:
        parts = base.split("-")
        if len(parts) == 2 and len(parts[1]) == 1 and parts[0].isalpha():
            return parts[0] + "." + parts[1]
    return base


# -----------------------------
# Helpers
# -----------------------------
def _to_date(x) -> Optional[date]:
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


def _extract_tickers_from_df(df: pd.DataFrame) -> List[str]:
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

    raw = df[ticker_col].astype(str).tolist()
    tickers = []
    for t in raw:
        b = _base_ticker(t)
        if b and b not in ("NAN", "NONE"):
            tickers.append(b)

    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@st.cache_data(ttl=24 * 3600)
def load_sp500_universe_map() -> pd.DataFrame:
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
            df["Ticker"] = df["Ticker"].astype(str).map(_base_ticker)
        keep = [c for c in ["Ticker", "Company", "Sector", "Industry"] if c in df.columns]
        df = df[keep].drop_duplicates(subset=["Ticker"]) if "Ticker" in df.columns else df
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Company", "Sector", "Industry"])


def get_finnhub_key() -> Optional[str]:
    try:
        key = st.secrets.get("FINNHUB_API_KEY", None)
    except Exception:
        key = None
    if isinstance(key, str):
        key = key.strip()
        return key if key else None
    return None


def finnhub_get(endpoint: str, params: Dict, api_key: str) -> Tuple[int, dict]:
    url = f"{FINNHUB_BASE}{endpoint}"
    p = dict(params or {})
    p["token"] = api_key
    try:
        r = requests.get(url, params=p, timeout=25)
        status = r.status_code
        try:
            js = r.json()
        except Exception:
            js = {}
        return status, js
    except Exception:
        return 0, {}


@st.cache_data(ttl=6 * 3600)
def finnhub_calendar_range(frm: date, to: date, api_key: str) -> pd.DataFrame:
    """
    Single call (or as close as possible) to fetch earnings calendar for a range,
    then we filter locally for your tickers. This avoids rate-limiting.
    """
    status, js = finnhub_get(
        "/calendar/earnings",
        {"from": frm.isoformat(), "to": to.isoformat()},
        api_key=api_key,
    )
    if status != 200 or not isinstance(js, dict):
        return pd.DataFrame()

    items = js.get("earningsCalendar", [])
    if not isinstance(items, list) or len(items) == 0:
        return pd.DataFrame()

    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = it.get("symbol")
        d = _to_date(it.get("date"))
        if not sym or d is None:
            continue
        rows.append(
            {
                "symbol": _base_ticker(sym),
                "date": d,
                "hour": it.get("hour"),
                "year": it.get("year"),
                "quarter": it.get("quarter"),
                "epsActual": _safe_float(it.get("epsActual")),
                "epsEstimate": _safe_float(it.get("epsEstimate")),
                "epsSurprise": _safe_float(it.get("epsSurprise")),
                "epsSurprisePercent": _safe_float(it.get("epsSurprisePercent")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)
    return df


@st.cache_data(ttl=6 * 3600)
def yf_fast_info(yf_symbol: str) -> Dict:
    try:
        t = yf.Ticker(yf_symbol)
        fi = getattr(t, "fast_info", None)
        return dict(fi) if fi is not None else {}
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600)
def yf_history_1y_single(yf_symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            yf_symbol,
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
def yf_history_2y_single(yf_symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            yf_symbol,
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


def _get_current_high_low_52w(yf_symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    fi = yf_fast_info(yf_symbol)

    current = _safe_float(fi.get("last_price") or fi.get("lastPrice") or fi.get("regular_market_price"))
    high52 = _safe_float(fi.get("year_high") or fi.get("yearHigh"))
    low52 = _safe_float(fi.get("year_low") or fi.get("yearLow"))

    if current is not None and high52 is not None and low52 is not None:
        return current, high52, low52

    hist = yf_history_1y_single(yf_symbol)
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
    Robust reaction:
      - Pre Close = last close before event_date
      - Post Close (0D) = close on event_date if trading day exists
      - Post Close (1D) = first close after event_date
      - Post Close (3D) = third close after event_date
    """
    out = {
        "Pre Close Date": None,
        "Pre Close": None,
        "Post Close (0D) Date": None,
        "Post Close (0D)": None,
        "0D Reaction %": None,
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

    # pre: strictly before
    pre_dates = [d for d in idx if d < event_date and d in closes and pd.notna(closes[d])]
    if pre_dates:
        pre_d = pre_dates[-1]
        out["Pre Close Date"] = pre_d
        out["Pre Close"] = float(closes[pre_d])

    # post0: same day if exists
    if event_date in closes and pd.notna(closes[event_date]):
        out["Post Close (0D) Date"] = event_date
        out["Post Close (0D)"] = float(closes[event_date])
        if out["Pre Close"]:
            out["0D Reaction %"] = (out["Post Close (0D)"] / out["Pre Close"] - 1.0) * 100.0

    # post list: strictly after
    post_dates = [d for d in idx if d > event_date and d in closes and pd.notna(closes[d])]
    if post_dates and out["Pre Close"]:
        post1 = post_dates[0]
        out["Post Close (1D) Date"] = post1
        out["Post Close (1D)"] = float(closes[post1])
        out["1D Reaction %"] = (out["Post Close (1D)"] / out["Pre Close"] - 1.0) * 100.0

        if len(post_dates) >= 3:
            post3 = post_dates[2]
            out["Post Close (3D) Date"] = post3
            out["Post Close (3D)"] = float(closes[post3])
            out["3D Reaction %"] = (out["Post Close (3D)"] / out["Pre Close"] - 1.0) * 100.0

    return out


# -----------------------------
# UI: Header + Upload (visible)
# -----------------------------
st.title("📅 Earnings Radar (Portfolio)")
st.caption(
    "Upload your holdings (CSV/XLSX) or paste tickers. "
    "Earnings dates + EPS from Finnhub (calendar). "
    "Market cap + price + 52-week stats + price reaction from Yahoo (yfinance)."
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

compute_reactions = st.sidebar.checkbox(
    "Compute price reaction (0D/1D/3D) in Past Earnings section (slower)",
    value=True,
)

st.sidebar.caption("Tip: fewer tickers = faster. Caching helps a lot after the first run.")
run = st.sidebar.button("🚀 Run Earnings Radar", use_container_width=True)

# -----------------------------
# Load tickers
# -----------------------------
tickers_base: List[str] = []
holdings_df = None

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            holdings_df = pd.read_csv(uploaded)
        else:
            holdings_df = pd.read_excel(uploaded)
        tickers_base = _extract_tickers_from_df(holdings_df)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if pasted and pasted.strip():
    pasted_list = [_base_ticker(x) for x in pasted.split(",") if x.strip()]
    seen = set(tickers_base)
    for t in pasted_list:
        if t and t not in seen:
            tickers_base.append(t)
            seen.add(t)

tickers_base = tickers_base[: int(max_tickers)]

if tickers_base:
    st.success(f"Loaded {len(tickers_base)} unique tickers.")
else:
    st.info("Upload a file or paste tickers to begin.")
    st.stop()

# build symbol maps
symbol_rows = []
for b in tickers_base:
    symbol_rows.append(
        {
            "base": b,
            "yf": _yf_ticker(b),
            "fh": _finnhub_ticker(b),
        }
    )
sym_df = pd.DataFrame(symbol_rows)
base_to_yf = dict(zip(sym_df["base"], sym_df["yf"]))
base_to_fh = dict(zip(sym_df["base"], sym_df["fh"]))

# -----------------------------
# Main compute
# -----------------------------
if run:
    today = date.today()

    # Pull Finnhub calendar ONCE for future + past (avoids rate limit nuking)
    future_df = finnhub_calendar_range(today, today + timedelta(days=int(lookahead_days)), api_key=api_key)
    past_df = finnhub_calendar_range(today - timedelta(days=600), today, api_key=api_key)

    # Finnhub symbols in calendar might come back with dots for class shares, etc.
    wanted_fh = set(base_to_fh.values())

    future_df = future_df[future_df["symbol"].isin(wanted_fh)].copy() if not future_df.empty else future_df
    past_df = past_df[past_df["symbol"].isin(wanted_fh)].copy() if not past_df.empty else past_df

    # Helpers to get next/past events per ticker from the cached tables
    def _next_earnings_for(base: str) -> Optional[date]:
        fh = base_to_fh[base]
        if future_df is None or future_df.empty:
            return None
        df = future_df[future_df["symbol"] == fh]
        if df.empty:
            return None
        # next upcoming >= today
        dts = [d for d in df["date"].tolist() if isinstance(d, date) and d >= today]
        return min(dts) if dts else None

    def _past_events_for(base: str, n: int = 4) -> pd.DataFrame:
        fh = base_to_fh[base]
        if past_df is None or past_df.empty:
            return pd.DataFrame()
        df = past_df[past_df["symbol"] == fh].copy()
        if df.empty:
            return df

        df = df.sort_values("date", ascending=False).head(int(n))
        # build nice output
        out = pd.DataFrame(
            {
                "Earnings Date": df["date"],
                "Year": df["year"],
                "Quarter": df["quarter"],
                "Hour": df["hour"],
                "EPS Actual": df["epsActual"],
                "EPS Est.": df["epsEstimate"],
                "Surprise": df["epsSurprise"],
                "Surprise %": df["epsSurprisePercent"],
            }
        )
        return out.reset_index(drop=True)

    # SP500 metadata
    sp500_map = load_sp500_universe_map()
    sp500_lookup = {}
    if not sp500_map.empty and "Ticker" in sp500_map.columns:
        sp500_lookup = sp500_map.set_index("Ticker").to_dict(orient="index")

    # Build portfolio overview
    rows = []
    progress = st.progress(0, text="Fetching portfolio stats...")
    total = len(tickers_base)

    for i, b in enumerate(tickers_base):
        yf_sym = base_to_yf[b]
        fh_sym = base_to_fh[b]

        meta = sp500_lookup.get(b, {})
        company = meta.get("Company")
        sector = meta.get("Sector")
        industry = meta.get("Industry")

        next_e = _next_earnings_for(b)
        days_to = (next_e - today).days if next_e else None

        # yfinance data fill
        fi = yf_fast_info(yf_sym)
        mcap = _safe_float(fi.get("market_cap")) or _safe_float(fi.get("marketCap"))
        current, high52, low52 = _get_current_high_low_52w(yf_sym)

        # yfinance metadata fallback
        if not company or not sector or not industry:
            try:
                info = yf.Ticker(yf_sym).get_info()
                company = company or info.get("shortName") or info.get("longName")
                sector = sector or info.get("sector")
                industry = industry or info.get("industry")
            except Exception:
                pass

        d_vs_high = None
        d_vs_low = None
        r52 = None
        if current is not None and high52 not in (None, 0):
            d_vs_high = (current - high52) / high52 * 100.0
        if current is not None and low52 not in (None, 0):
            d_vs_low = (current - low52) / low52 * 100.0
        if high52 is not None and low52 not in (None, 0):
            r52 = (high52 - low52) / low52 * 100.0

        rows.append(
            {
                "Ticker": b,
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
                "_yf": yf_sym,
                "_fh": fh_sym,
            }
        )

        progress.progress((i + 1) / total, text=f"Fetched {i+1}/{total}: {b}")

    progress.empty()

    overview = pd.DataFrame(rows)

    # Filtering for overview
    filtered = overview.copy()
    if only_with_upcoming:
        filtered = filtered[filtered["Next Earnings"].notna()].copy()
    if not filtered.empty:
        filtered = filtered[filtered["Days to Earnings"].notna()].copy()
        filtered = filtered[(filtered["Days to Earnings"] >= 0) & (filtered["Days to Earnings"] <= within_days)].copy()
    if not filtered.empty:
        filtered = filtered.sort_values(["Days to Earnings", "Ticker"], ascending=[True, True])

    # -----------------------------
    # Display: Portfolio Overview
    # -----------------------------
    st.subheader("2) Portfolio Overview")

    if filtered.empty:
        st.warning("No holdings matched your filters (try turning off 'Show only holdings with an upcoming earnings date').")
    else:
        view = filtered.drop(columns=["_yf", "_fh"], errors="ignore").copy()
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

        st.download_button(
            "⬇️ Download Portfolio Overview (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="earnings_radar_portfolio_overview.csv",
            mime="text/csv",
        )

    # -----------------------------
    # Past 4 earnings + reaction
    # -----------------------------
    st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")
    st.caption(
        "Past 4 earnings come from Finnhub earnings calendar (pulled once for the whole portfolio). "
        "Price reaction uses Yahoo daily closes (yfinance)."
    )

    row_by_ticker = {r["Ticker"]: r for r in rows}

    for b in tickers_base:
        company = (row_by_ticker.get(b) or {}).get("Company", None)
        yf_sym = (row_by_ticker.get(b) or {}).get("_yf", _yf_ticker(b))
        label = f"{b}" + (f" — {company}" if company else "")

        with st.expander(label, expanded=False):
            earn_df = _past_events_for(b, n=4)

            if earn_df is None or earn_df.empty:
                st.info("No earnings history returned (common for ETFs/funds or some non-US listings).")
                continue

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

            hist2y = yf_history_2y_single(yf_sym)
            if hist2y is None or hist2y.empty:
                st.warning("Could not fetch 2Y daily price history from Yahoo Finance for reactions.")
                st.dataframe(earn_df, use_container_width=True, hide_index=True)
                continue

            enriched = []
            for _, r in earn_df.iterrows():
                ed = r.get("Earnings Date")
                if isinstance(ed, pd.Timestamp):
                    ed = ed.date()
                if not isinstance(ed, date):
                    ed = _to_date(ed)

                base = dict(r)
                if ed is None:
                    enriched.append(base)
                    continue

                base.update(compute_reaction_from_history(hist2y, ed))
                enriched.append(base)

            out_df = pd.DataFrame(enriched)

            keep_cols = [
                "Earnings Date",
                "Year",
                "Quarter",
                "Hour",
                "EPS Actual",
                "EPS Est.",
                "Surprise",
                "Surprise %",
                "Pre Close Date",
                "Pre Close",
                "Post Close (0D) Date",
                "Post Close (0D)",
                "0D Reaction %",
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
                    "Post Close (0D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Post Close (1D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Post Close (3D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "EPS Actual": st.column_config.NumberColumn(format="%.3f"),
                    "EPS Est.": st.column_config.NumberColumn(format="%.3f"),
                    "Surprise": st.column_config.NumberColumn(format="%.3f"),
                    "Surprise %": st.column_config.NumberColumn(format="%.1f"),
                    "Pre Close": st.column_config.NumberColumn(format="%.2f"),
                    "Post Close (0D)": st.column_config.NumberColumn(format="%.2f"),
                    "Post Close (1D)": st.column_config.NumberColumn(format="%.2f"),
                    "Post Close (3D)": st.column_config.NumberColumn(format="%.2f"),
                    "0D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                    "1D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                    "3D Reaction %": st.column_config.NumberColumn(format="%.1f"),
                },
            )

else:
    st.info("Click **Run Earnings Radar** in the sidebar to fetch data.")
