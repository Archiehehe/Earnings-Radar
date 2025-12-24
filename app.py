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
                dt = pd.to_datetime(match.group(1)).date()
                if is_future(dt): return dt
    except: pass
    return None

def get_next_earnings_yf_calendar(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is not None and not cal.empty:
            if 'Earnings Date' in cal.index:
                dates = cal.loc['Earnings Date']
                if isinstance(dates, (list, pd.Series, pd.Index)):
                    for d in dates:
                        dt = pd.to_datetime(d).date()
                        if is_future(dt): return dt
                else:
                    dt = pd.to_datetime(dates).date()
                    if is_future(dt): return dt
    except: pass
    return None

def get_next_earnings(ticker):
    methods = [get_next_earnings_yf_calendar, get_next_earnings_yahoo_scrape]
    for method in methods:
        result = method(ticker)
        if result: return result
    return "TBD"

# =========================
# REACTIONS (FIXED FOR TRADING DAYS)
# =========================
def reaction(price_df, earnings_date, days_after):
    """
    Finds the reaction after earnings by strictly using the next 
    available trading days, skipping weekends and holidays.
    """
    try:
        # 1. Normalize dates to compare without time
        price_df.index = pd.to_datetime(price_df.index).normalize()
        e_date = pd.to_datetime(earnings_date).normalize()
        
        # 2. Get the 'Pre-Earnings' price (last close on or before earnings date)
        pre_data = price_df.loc[:e_date]
        if pre_data.empty:
            return None
        pre_close = pre_data.iloc[-1]["Close"]
        
        # 3. Get all trading days strictly AFTER the earnings date
        post_data = price_df.loc[e_date + timedelta(days=1):]
        if post_data.empty:
            return None
            
        # 4. Pick the Nth available trading day (e.g., 1st or 3rd)
        # If we ask for 3 days but only have 2, take the latest available
        idx = min(days_after - 1, len(post_data) - 1)
        post_close = post_data.iloc[idx]["Close"]
        
        return pct(post_close, pre_close)
    except Exception:
        return None

# =========================
# DATA FETCHING
# =========================
def finnhub_past_earnings(ticker, limit=4):
    try:
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r[:limit])
        if not df.empty:
            df["date"] = pd.to_datetime(df["period"])
            return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def yf_prices(tickers, period):
    return yf.download(tickers=tickers, period=period, group_by="ticker", progress=False)

def market_cap(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        return safe_float(fi.get("market_cap") or fi.get("marketCap"))
    except: return None

def fetch_all(tickers, progress):
    rows = []
    prices_2y = yf_prices(tickers, "2y")
    
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        mcaps = dict(zip(tickers, ex.map(market_cap, tickers)))
        next_earn = {t: ex.submit(get_next_earnings, t) for t in tickers}
        past_earn = {t: ex.submit(finnhub_past_earnings, t) for t in tickers}

    for i, t in enumerate(tickers):
        try:
            # Handle single vs multi-ticker dataframe structure
            p2 = prices_2y[t] if len(tickers) > 1 else prices_2y
            
            current = safe_float(p2["Close"].iloc[-1])
            high52 = safe_float(p2["High"].iloc[-252:].max()) # Approx 1yr of trading
            low52 = safe_float(p2["Low"].iloc[-252:].min())

            df_past = past_earn[t].result()
            earn_rows = []
            
            if not df_past.empty:
                for _, r in df_past.iterrows():
                    earn_rows.append({
                        "Date": r["date"].date(),
                        "EPS Actual": r.get("actual"),
                        "EPS Est.": r.get("estimate"),
                        "Surprise": r.get("surprise"),
                        "1D Reaction %": reaction(p2, r["date"], 1),
                        "3D Reaction %": reaction(p2, r["date"], 3),
                    })
            else:
                earn_rows.append({"Date": None, "EPS Actual": None, "EPS Est.": None, "Surprise": None, "1D Reaction %": None, "3D Reaction %": None})

            for e in earn_rows:
                rows.append({
                    "Ticker": t,
                    "Market Cap": format_market_cap(mcaps.get(t)),
                    "Current Price": current,
                    "52W High": high52,
                    "52W Low": low52,
                    "Δ vs 52W High %": pct(current, high52),
                    "Δ vs 52W Low %": pct(current, low52),
                    "Next Earnings": next_earn[t].result(),
                    **e
                })
        except: rows.append({"Ticker": t})
        progress.progress((i + 1) / len(tickers))
    return rows

# =========================
# UI
# =========================
st.title("📊 Earnings Radar")

tickers_text = st.text_area("Enter tickers", "AAPL\nMSFT\nNVDA\nGOOGL")
tickers = sorted(set(t.strip().upper() for t in tickers_text.replace(",", "\n").split() if t.strip()))

if st.button("Fetch Earnings"):
    if not tickers:
        st.warning("No tickers provided")
    else:
        progress = st.progress(0.0)
        final_rows = fetch_all(tickers, progress)
        df_result = pd.DataFrame(final_rows)
        
        pct_cols = ["Δ vs 52W High %", "Δ vs 52W Low %", "1D Reaction %", "3D Reaction %"]
        st.dataframe(
            df_result, 
            use_container_width=True,
            column_config={col: st.column_config.NumberColumn(format="%.2f%%") for col in pct_cols}
        )
