import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# =========================
# CONFIG
# =========================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
MAX_WORKERS = 8

st.set_page_config(page_title="Earnings Radar", layout="wide")

# =========================
# HELPERS
# =========================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (a - b) / b * 100

def is_future(date_obj):
    """Check if a date is today or in the future"""
    if date_obj is None:
        return False
    return date_obj >= datetime.now().date()

def format_market_cap(val):
    """Truncate large market cap numbers to M, B, or T"""
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    if val >= 1e12:
        return f"{val / 1e12:.2f}T"
    elif val >= 1e9:
        return f"{val / 1e9:.2f}B"
    elif val >= 1e6:
        return f"{val / 1e6:.2f}M"
    return f"{val:.2f}"

# =========================
# NEXT EARNINGS (MULTIPLE METHODS)
# =========================
def get_next_earnings_yahoo_scrape(ticker):
    """Scrape next earnings from Yahoo Finance page"""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        text = response.text
        if 'Earnings Date' in text:
            import re
            date_pattern = r'(\w{3}\s+\d{1,2},\s+\d{4})'
            match = re.search(date_pattern, text)
            if match:
                date_str = match.group(1)
                dt = pd.to_datetime(date_str).date()
                if is_future(dt):
                    return dt
    except Exception:
        pass
    return None

def get_next_earnings_yf_info(ticker):
    """Try yfinance info method"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        for field in ['earningsDate', 'earningsTimestamp', 'nextEarningsDate']:
            if field in info and info[field]:
                date_val = info[field][0] if isinstance(info[field], list) else info[field]
                dt = pd.to_datetime(date_val, unit='s' if isinstance(date_val, (int, float)) else None).date()
                if is_future(dt):
                    return dt
    except Exception:
        pass
    return None

def get_next_earnings_yf_calendar(ticker):
    """Try yfinance calendar"""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is not None and not cal.empty:
            if 'Earnings Date' in cal.index:
                dates = cal.loc['Earnings Date']
                if isinstance(dates, (list, pd.Series)):
                    for d in dates:
                        dt = pd.to_datetime(d).date()
                        if is_future(dt):
                            return dt
                else:
                    dt = pd.to_datetime(dates).date()
                    if is_future(dt):
                        return dt
    except Exception:
        pass
    return None

def get_next_earnings_fmp(ticker):
    """Try Financial Modeling Prep API"""
    try:
        url = f"https://financialmodelingprep.com/api/v3/earning_calendar?symbol={ticker}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data:
            dt = pd.to_datetime(data[0].get('date')).date()
            if is_future(dt):
                return dt
    except Exception:
        pass
    return None

def get_next_earnings(ticker):
    """Try multiple methods and force a future date"""
    methods = [
        get_next_earnings_yf_calendar,
        get_next_earnings_yf_info,
        get_next_earnings_fmp,
        get_next_earnings_yahoo_scrape,
    ]
    
    for method in methods:
        result = method(ticker)
        if result and is_future(result):
            return result
    return "TBD"

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
        return safe_float(fi.get("market_cap") or fi.get("marketCap"))
    except Exception:
        return None

def reaction(price_df, date, trading_days):
    try:
        d = pd.to_datetime(date).normalize()
        pre_data = price_df.loc[:d]
        if pre_data.empty: return None
        pre = pre_data.iloc[-1]["Close"]
        
        post_data = price_df.loc[d + timedelta(days=1):]
        if post_data.empty: return None
        
        idx = min(trading_days - 1, len(post_data) - 1)
        post = post_data.iloc[idx]["Close"]
        return pct(post, pre)
    except Exception:
        return None

# =========================
# MAIN FETCH
# =========================
def fetch_all(tickers, progress):
    rows = []
    prices_1y = yf_prices(tickers, "1y")
    prices_2y = yf_prices(tickers, "2y")

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        mcaps = dict(zip(tickers, ex.map(market_cap, tickers)))

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(get_next_earnings, t): t for t in tickers}
        next_earn = {futures[f]: f.result() for f in as_completed(futures)}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(finnhub_past_earnings, t): t for t in tickers}
        past_earn = {futures[f]: f.result() for f in as_completed(futures)}

    for i, t in enumerate(tickers):
        try:
            p1 = prices_1y[t] if len(tickers) > 1 else prices_1y
            p2 = prices_2y[t] if len(tickers) > 1 else prices_2y

            current = safe_float(p1["Close"].iloc[-1])
            high52 = safe_float(p1["High"].max())
            low52 = safe_float(p1["Low"].min())

            earn_rows = []
            df = past_earn.get(t, pd.DataFrame())
            if not df.empty:
                for _, r in df.iterrows():
                    earn_rows.append({
                        "Date": r["date"].date(),
                        "EPS Actual": r.get("actual"),
                        "EPS Est.": r.get("estimate"),
                        "Surprise": r.get("surprise"),
                        "1D Reaction %": reaction(p2, r["date"], 1),
                        "3D Reaction %": reaction(p2, r["date"], 3),
                    })
            
            if not earn_rows:
                earn_rows.append({
                    "Date": None, "EPS Actual": None, "EPS Est.": None,
                    "Surprise": None, "1D Reaction %": None, "3D Reaction %": None
                })

            for e in earn_rows:
                rows.append({
                    "Ticker": t,
                    "Market Cap": format_market_cap(mcaps.get(t)),
                    "Current Price": current,
                    "52W High": high52,
                    "52W Low": low52,
                    "Δ vs 52W High %": pct(current, high52),
                    "Δ vs 52W Low %": pct(current, low52),
                    "Next Earnings": next_earn.get(t),
                    **e
                })
        except Exception:
            rows.append({"Ticker": t})
        progress.progress((i + 1) / len(tickers))
    return rows

# =========================
# UI
# =========================
st.title("📊 Earnings Radar")

uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
tickers_text = st.text_area("Enter tickers", "AAPL\nMSFT\nNVDA\nGOOGL")

tickers = set()
if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            for col in ["Ticker", "Symbol", "ticker", "symbol"]:
                if col in df.columns:
                    tickers.update(df[col].dropna().astype(str).str.upper())
                    break
        except: st.warning(f"Could not read {f.name}")

tickers.update(t.strip().upper() for t in tickers_text.replace(",", "\n").split() if t.strip())
tickers = sorted(tickers)

if st.button("Fetch Earnings"):
    if not tickers:
        st.warning("No tickers provided")
    else:
        progress = st.progress(0.0)
        final_rows = fetch_all(tickers, progress)
        df_result = pd.DataFrame(final_rows)
        
        # --- APPLY PERCENTAGE FORMATTING ---
        pct_cols = [
            "Δ vs 52W High %", "Δ vs 52W Low %", 
            "1D Reaction %", "3D Reaction %"
        ]
        
        # Use Streamlit's column config for clean number formatting
        st.dataframe(
            df_result, 
            use_container_width=True,
            column_config={
                col: st.column_config.NumberColumn(format="%.2f%%") 
                for col in pct_cols
            }
        )
