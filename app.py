import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# ⚙️ CONFIGURATION & STYLE
# =========================
st.set_page_config(
    page_title="Earnings Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# API Keys
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
MAX_WORKERS = 8

# =========================
# 🛠️ HELPER FUNCTIONS
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
    if date_obj is None:
        return False
    return date_obj >= datetime.now().date()

def format_market_cap(val):
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    elif val >= 1e9:
        return f"${val / 1e9:.2f}B"
    elif val >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:.2f}"

def format_percentage(val):
    if val is None:
        return "-"
    color = "green" if val >= 0 else "red"
    return f":{color}[{val:.2f}%]"

# =========================
# 📅 EARNINGS DATE FETCHERS
# =========================
def get_next_earnings_yahoo_scrape(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        text = response.text
        if 'Earnings Date' in text:
            import re
            match = re.search(r'(\w{3}\s+\d{1,2},\s+\d{4})', text)
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
            dates = cal.loc['Earnings Date'] if 'Earnings Date' in cal.index else None
            if dates is not None:
                # Handle both single date and list of dates
                if isinstance(dates, (list, pd.Series)):
                    dates = dates[0]
                dt = pd.to_datetime(dates).date()
                if is_future(dt): return dt
    except: pass
    return None

def get_next_earnings(ticker):
    # Try methods in order of reliability/speed
    methods = [get_next_earnings_yf_calendar, get_next_earnings_yahoo_scrape]
    for method in methods:
        res = method(ticker)
        if res: return res
    return "TBD"

# =========================
# 📉 HISTORICAL DATA
# =========================
def finnhub_past_earnings(ticker, limit=4):
    try:
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=5).json()
        if isinstance(r, list):
            df = pd.DataFrame(r[:limit])
            if not df.empty:
                df["date"] = pd.to_datetime(df["period"])
                return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    # Batch fetch prices for efficiency
    try:
        data_1y = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
        data_2y = yf.download(tickers, period="2y", group_by='ticker', threads=True, progress=False)
        return data_1y, data_2y
    except Exception as e:
        return None, None

def reaction(price_df, date, trading_days):
    try:
        d = pd.to_datetime(date).normalize()
        # Find closest trading day before
        pre_data = price_df.loc[:d]
        if pre_data.empty: return None
        pre = pre_data.iloc[-1]["Close"]
        
        # Find price N days after
        post_data = price_df.loc[d + timedelta(days=1):]
        if post_data.empty: return None
        idx = min(trading_days - 1, len(post_data) - 1)
        post = post_data.iloc[idx]["Close"]
        
        return pct(post, pre)
    except: return None

# =========================
# 🚀 MAIN LOGIC
# =========================
def analyze_portfolio(tickers):
    status = st.status("🚀 Launching Earnings Radar...", expanded=True)
    
    # 1. Market Data
    status.write("📉 Fetching price history...")
    p1, p2 = fetch_market_data(list(tickers))
    
    # 2. Market Caps
    status.write("💰 Calculating market caps...")
    mcaps = {}
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(lambda t: yf.Ticker(t).fast_info.market_cap, t): t for t in tickers}
        for f in as_completed(futures):
            try: mcaps[futures[f]] = f.result()
            except: mcaps[futures[f]] = None

    # 3. Next Earnings
    status.write("📅 Scanning for future earnings dates...")
    next_earn = {}
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(get_next_earnings, t): t for t in tickers}
        for f in as_completed(futures):
            next_earn[futures[f]] = f.result()

    # 4. Past Earnings & Reactions
    status.write("📊 Analyzing historical reactions...")
    rows = []
    
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(finnhub_past_earnings, t): t for t in tickers}
        past_earn_data = {futures[f]: f.result() for f in as_completed(futures)}

    # 5. Assemble Data
    for t in tickers:
        try:
            # Handle Single vs Multi-Index levels from yfinance
            if len(tickers) > 1:
                t_p1 = p1[t] if t in p1 else p1
                t_p2 = p2[t] if t in p2 else p2
            else:
                t_p1 = p1
                t_p2 = p2

            # Basic Stats
            current = safe_float(t_p1["Close"].iloc[-1]) if not t_p1.empty else 0
            high52 = safe_float(t_p1["High"].max()) if not t_p1.empty else 0
            low52 = safe_float(t_p1["Low"].min()) if not t_p1.empty else 0
            
            # Historical Rows
            hist_df = past_earn_data.get(t, pd.DataFrame())
            earn_rows = []
            
            if not hist_df.empty:
                for _, r in hist_df.iterrows():
                    earn_rows.append({
                        "Date": r["date"].date(),
                        "Surprise %": r.get("surprise"),
                        "1D Reaction %": reaction(t_p2, r["date"], 1),
                        "3D Reaction %": reaction(t_p2, r["date"], 3),
                    })
            else:
                earn_rows.append({"Date": None, "Surprise %": None, "1D Reaction %": None})

            # Create a row for every historical event found
            for e in earn_rows:
                rows.append({
                    "Ticker": t,
                    "Market Cap": format_market_cap(mcaps.get(t)),
                    "Price": current,
                    "Next Earnings": next_earn.get(t, "TBD"),
                    "Near High %": pct(current, high52),
                    "Near Low %": pct(current, low52),
                    **e
                })
        except Exception as e:
            print(f"Error processing {t}: {e}")
            rows.append({"Ticker": t, "Next Earnings": "Error"})

    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
    return pd.DataFrame(rows)

# =========================
# 🖥️ UI LAYOUT
# =========================
with st.sidebar:
    st.header("1. Input Data")
    uploaded_files = st.file_uploader("📂 Upload Portfolio (CSV/Excel)", accept_multiple_files=True)
    
    manual_tickers = st.text_area(
        "📝 Or Enter Tickers (one per line)", 
        "AAPL\nNVDA\nMSFT\nAMD\nTSLA",
        height=150
    )
    
    st.info("💡 Tip: Uploading a CSV with a 'Ticker' column works best!")

    # Parse Tickers
    tickers = set()
    if uploaded_files:
        for f in uploaded_files:
            try:
                df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                cols = [c for c in df.columns if c.lower() in ['ticker', 'symbol']]
                if cols:
                    tickers.update(df[cols[0]].astype(str).str.upper().tolist())
            except: st.error(f"Failed to read {f.name}")
            
    tickers.update([t.strip().upper() for t in manual_tickers.split() if t.strip()])
    
    run_btn = st.button("🚀 Analyze Portfolio", type="primary")

# MAIN PAGE
st.title("📡 Earnings Radar")
st.markdown("Track upcoming earnings dates and analyze historical price volatility.")

if run_btn and tickers:
    df_result = analyze_portfolio(sorted(list(tickers)))
    
    if not df_result.empty:
        # === PRE-PROCESSING FOR DISPLAY ===
        # Fix mixed types for Streamlit/Arrow compatibility
        df_result["Next Earnings"] = df_result["Next Earnings"].astype(str)
        
        # === DASHBOARD STATS ===
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Tickers", len(tickers))
        with c2:
            upcoming = df_result[df_result["Next Earnings"] != "TBD"]["Next Earnings"].nunique()
            st.metric("Confirmed Dates", upcoming)
        with c3:
            avg_move = df_result["1D Reaction %"].abs().mean()
            st.metric("Avg Volatility (1D)", f"{avg_move:.1f}%" if pd.notnull(avg_move) else "-")
        with c4:
            winners = len(df_result[df_result["1D Reaction %"] > 0])
            st.metric("Positive History", f"{winners} events")

        st.divider()

        # === INTERACTIVE TABLE ===
        st.subheader("📊 Detailed Analysis")
        
        # Configure columns for a beautiful table
        column_config = {
            "Ticker": st.column_config.TextColumn("Symbol", width="small"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Next Earnings": st.column_config.TextColumn("Next Event", help="Projected or confirmed date"),
            "1D Reaction %": st.column_config.NumberColumn(
                "1D Move", 
                format="%.2f%%",
                help="Price change 1 day after earnings"
            ),
            "3D Reaction %": st.column_config.NumberColumn(
                "3D Move", 
                format="%.2f%%"
            ),
            "Surprise %": st.column_config.NumberColumn(
                "Surprise", 
                format="%.2f%%"
            ),
            "Near High %": st.column_config.ProgressColumn(
                "vs 52W High", 
                format="%.0f%%", 
                min_value=-100, max_value=0
            ),
        }

        st.dataframe(
            df_result,
            use_container_width=True, # Using standard width param
            column_config=column_config,
            height=600,
            hide_index=True
        )

        # === EXPORT ===
        st.divider()
        buffer = io.BytesIO()
        # FIXED: Using openpyxl explicitly to avoid crashes
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name='Earnings Radar')
            
        st.download_button(
            label="📥 Download Report (Excel)",
            data=buffer.getvalue(),
            file_name=f"Earnings_Radar_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    else:
        st.warning("No data found. Please check your tickers.")
elif run_btn and not tickers:
    st.warning("Please enter at least one ticker symbol.")
else:
    # Empty State
    st.info("👈 Add tickers in the sidebar to get started!")
