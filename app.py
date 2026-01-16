import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import time

# =========================
# CONFIG
# =========================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
MAX_WORKERS = 4

st.set_page_config(page_title="Earnings Radar", layout="wide", page_icon="📊")

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
        time.sleep(0.5)
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
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        mcaps = dict(zip(tickers, ex.map(market_cap, tickers)))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_next_earnings, t): t for t in tickers}
        next_earn = {}
        for f in as_completed(futures):
            try:
                next_earn[futures[f]] = f.result()
            except Exception:
                next_earn[futures[f]] = "TBD"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(finnhub_past_earnings, t): t for t in tickers}
        past_earn = {futures[f]: f.result() for f in as_completed(futures)}

    for i, t in enumerate(tickers):
        try:
            p1 = prices_1y[t] if len(tickers) > 1 else prices_1y
            p2 = prices_2y[t] if len(tickers) > 1 else prices_2y

            current = safe_float(p1["Close"].iloc[-1]) if not p1.empty else None
            high52 = safe_float(p1["High"].max()) if not p1.empty else None
            low52 = safe_float(p1["Low"].min()) if not p1.empty else None

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
        except Exception as ex:
            st.warning(f"Error processing {t}: {str(ex)}")
            rows.append({"Ticker": t})
        progress.progress((i + 1) / len(tickers))
    return rows

# =========================
# UI
# =========================
# Sidebar for inputs
with st.sidebar:
    st.header("Portfolio Input")
    uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    tickers_text = st.text_area("Or enter tickers (one per line)", "AAPL\nMSFT\nNVDA\nGOOGL")
    
    st.divider()
    st.markdown("**About Earnings Radar**")
    st.markdown("Track upcoming earnings and historical reactions for your stocks. Built with Streamlit and yfinance.")
    st.markdown("[GitHub Repo](https://github.com/Archiehehe/earnings)")

# Main content
st.title("📊 Earnings Radar")
st.markdown("Upload your portfolio or enter tickers to view upcoming earnings calendars and historical market reactions.")

# Collect tickers from both sources
tickers = set()

# Process uploaded files
if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
    for f in uploaded_files:
        try:
            # Read the file
            if f.name.endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            
            # Look for ticker column
            ticker_found = False
            for col in ["Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"]:
                if col in df.columns:
                    file_tickers = df[col].dropna().astype(str).str.strip().str.upper()
                    tickers.update(file_tickers)
                    st.success(f"✅ Found {len(file_tickers)} tickers in {f.name} (column: '{col}')")
                    ticker_found = True
                    break
            
            if not ticker_found:
                st.warning(f"⚠️ No ticker column found in {f.name}. Looking for columns: Ticker, Symbol, ticker, symbol")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
                
        except Exception as e:
            st.error(f"❌ Could not read {f.name}: {str(e)}")

# Process manual text input
manual_tickers = [t.strip().upper() for t in tickers_text.replace(",", "\n").split() if t.strip()]
if manual_tickers:
    tickers.update(manual_tickers)

# Convert to sorted list
tickers = sorted(tickers)

# Show what we found
if tickers:
    st.success(f"🎯 Total tickers loaded: {len(tickers)}")
    with st.expander("View all tickers"):
        st.write(", ".join(tickers))
else:
    st.warning("⚠️ No tickers loaded. Please upload a file or enter tickers manually.")

# Fetch button
if st.button("Fetch Earnings", type="primary", disabled=len(tickers)==0):
    with st.spinner(f"Fetching data for {len(tickers)} ticker(s)... This may take a moment."):
        progress = st.progress(0.0)
        final_rows = fetch_all(tickers, progress)
        df_result = pd.DataFrame(final_rows)
        
        # Sort by Ticker and Date descending
        df_result = df_result.sort_values(["Ticker", "Date"], ascending=[True, False])
        
        # Column configs for better display
        pct_cols = ["Δ vs 52W High %", "Δ vs 52W Low %", "1D Reaction %", "3D Reaction %"]
        column_config = {col: st.column_config.NumberColumn(format="%.2f%%") for col in pct_cols}
        column_config.update({
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "52W High": st.column_config.NumberColumn(format="$%.2f"),
            "52W Low": st.column_config.NumberColumn(format="$%.2f"),
            "EPS Actual": st.column_config.NumberColumn(format="%.4f"),
            "EPS Est.": st.column_config.NumberColumn(format="%.4f"),
            "Surprise": st.column_config.NumberColumn(format="%.4f"),
            "Next Earnings": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
        })
        
        st.subheader("Earnings Report")
        st.dataframe(df_result, hide_index=True, column_config=column_config)

        # Export to Excel
        st.divider()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name='Earnings_Report')
        
        st.download_button(
            label="📥 Download Results as Excel",
            data=buffer,
            file_name=f"earnings_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary"
        )

# Footer
st.divider()
st.markdown("Powered by yfinance, Finnhub, and Streamlit. Data may have delays or inaccuracies – always verify with official sources.")
