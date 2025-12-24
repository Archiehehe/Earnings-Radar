import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
MAX_WORKERS = 8

st.set_page_config(page_title="Earnings Calendar Tracker", layout="wide")

# =========================
# FORMATTERS
# =========================
def fmt_big(n):
    if n is None:
        return None
    n = float(n)
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:.2f}"


def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return round((a - b) / b * 100, 2)


# =========================
# FINNHUB (PAST EARNINGS)
# =========================
def finnhub_past_earnings(ticker, limit=4):
    try:
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r[:limit])
        if not df.empty:
            df["date"] = pd.to_datetime(df["period"])
            return df
    except Exception:
        pass
    return pd.DataFrame()


# =========================
# NEXT EARNINGS (Finnhub)
# =========================
def finnhub_next_earnings(ticker, lookahead_days=90):
    try:
        today = datetime.utcnow().date()
        end = today + timedelta(days=lookahead_days)
        url = f"https://finnhub.io/api/v1/calendar/earnings?from={today}&to={end}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        for e in r.get("earningsCalendar", []):
            if e.get("symbol") == ticker:
                return pd.to_datetime(e["date"]).date()
    except Exception:
        return None
    return None


# =========================
# YFINANCE (CACHED)
# =========================
@st.cache_data(ttl=3600)
def yf_prices(tickers, period):
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
    )


def market_cap(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        return fi.get("market_cap") or fi.get("marketCap")
    except Exception:
        return None


# =========================
# PRICE REACTION
# =========================
def reaction(price_df, date, days):
    try:
        d = pd.to_datetime(date).normalize()
        pre = price_df.loc[:d].iloc[-1]["Close"]
        post = price_df.loc[d + timedelta(days=days):].iloc[0]["Close"]
        return round((post - pre) / pre * 100, 2)
    except Exception:
        return None


# =========================
# MAIN FETCH
# =========================
def fetch_all(tickers, lookahead_days, progress):
    rows = []

    prices_1y = yf_prices(tickers, "1y")
    prices_2y = yf_prices(tickers, "2y")

    # Market cap (parallel)
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        mcaps = dict(zip(tickers, ex.map(market_cap, tickers)))

    # Past earnings (parallel)
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        past = dict(zip(tickers, ex.map(finnhub_past_earnings, tickers)))

    # Next earnings (parallel)
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        next_earn = dict(zip(tickers, ex.map(lambda t: finnhub_next_earnings(t, lookahead_days), tickers)))

    for i, t in enumerate(tickers):
        try:
            p1 = prices_1y[t]
            p2 = prices_2y[t]

            current = p1["Close"].iloc[-1]
            high52 = p1["High"].max()
            low52 = p1["Low"].min()

            df = past.get(t, pd.DataFrame())

            if df.empty:
                rows.append({
                    "Ticker": t,
                    "Market Cap": fmt_big(mcaps.get(t)),
                    "Current Price": round(current, 2),
                    "52W High": round(high52, 2),
                    "52W Low": round(low52, 2),
                    "Δ vs 52W High %": pct(current, high52),
                    "Δ vs 52W Low %": pct(current, low52),
                    "Next Earnings": next_earn.get(t) or "N/A",
                })
            else:
                for _, r in df.iterrows():
                    rows.append({
                        "Ticker": t,
                        "Market Cap": fmt_big(mcaps.get(t)),
                        "Current Price": round(current, 2),
                        "52W High": round(high52, 2),
                        "52W Low": round(low52, 2),
                        "Δ vs 52W High %": pct(current, high52),
                        "Δ vs 52W Low %": pct(current, low52),
                        "Next Earnings": next_earn.get(t) or "N/A",
                        "Earnings Date": r["date"].date(),
                        "EPS Actual": r.get("actual"),
                        "EPS Est.": r.get("estimate"),
                        "Surprise": r.get("surprise"),
                        "1D Reaction %": reaction(p2, r["date"], 1),
                        "3D Reaction %": reaction(p2, r["date"], 3),
                    })

        except Exception:
            rows.append({"Ticker": t})

        progress.progress((i + 1) / len(tickers))

    return rows


# =========================
# UI
# =========================
st.title("📅 Earnings Calendar Tracker")

uploaded_files = st.file_uploader(
    "Upload CSV or Excel (Ticker / Symbol column)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

text = st.text_area("Enter tickers", "AAPL\nMSFT\nNVDA\nGOOGL")

lookahead_days = st.slider("Lookahead days", 30, 180, 90)

tickers = set()

if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_excel(f) if f.name.endswith(("xls", "xlsx")) else pd.read_csv(f)
            for c in ["Ticker", "Symbol", "ticker", "symbol"]:
                if c in df.columns:
                    tickers.update(df[c].dropna().astype(str).str.upper())
        except Exception:
            st.warning(f"Could not read {f.name}")

tickers.update(t.strip().upper() for t in text.replace(",", "\n").split() if t.strip())
tickers = sorted(tickers)

if st.button("Fetch Earnings"):
    if not tickers:
        st.warning("No tickers provided")
    else:
        progress = st.progress(0.0)
        data = fetch_all(tickers, lookahead_days, progress)
        st.dataframe(pd.DataFrame(data), use_container_width=True)
