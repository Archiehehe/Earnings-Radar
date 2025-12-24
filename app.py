import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from groq import Groq  # Free AI Provider

# =========================
# CONFIG & SECRETS
# =========================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
MAX_WORKERS = 8

st.set_page_config(page_title="Earnings Radar", layout="wide")

# =========================
# RADAR HELPERS
# =========================
def safe_float(x):
    try: return float(x)
    except: return None

def pct(a, b):
    if a is None or b in (None, 0): return None
    return (a - b) / b * 100

def is_future(date_obj):
    if date_obj is None: return False
    return date_obj >= datetime.now().date()

def format_market_cap(val):
    if val is None or not isinstance(val, (int, float)): return "N/A"
    if val >= 1e12: return f"{val / 1e12:.2f}T"
    elif val >= 1e9: return f"{val / 1e9:.2f}B"
    elif val >= 1e6: return f"{val / 1e6:.2f}M"
    return f"{val:.2f}"

# =========================
# NEXT EARNINGS LOGIC
# =========================
def get_next_earnings(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is not None and not cal.empty and 'Earnings Date' in cal.index:
            dates = cal.loc['Earnings Date']
            d_list = dates if isinstance(dates, (list, pd.Series, pd.Index)) else [dates]
            for d in d_list:
                dt = pd.to_datetime(d).date()
                if is_future(dt): return dt
        
        url = f"https://finance.yahoo.com/quote/{ticker}"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        import re
        match = re.search(r'(\w{3}\s+\d{1,2},\s+\d{4})', r.text)
        if match:
            dt = pd.to_datetime(match.group(1)).date()
            if is_future(dt): return dt
    except: pass
    return "TBD"

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
    except: return None

# =========================
# AI HELPERS
# =========================
def fetch_transcript_text(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        return " ".join([p.get_text() for p in soup.find_all('p')])[:12000]
    except: return ""

def analyze_with_groq(ticker, text):
    if not GROQ_API_KEY: return {"summary": "No Key", "sentiment": 0, "notes": "Check Secrets"}
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"Analyze earnings for {ticker}. Return JSON: 'summary' (2 sentences), 'sentiment' (1-5), 'notes' (2 points). Text: {text[:5000]}"
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama3-8b-8192", response_format={"type":"json_object"})
        import json
        return json.loads(res.choices[0].message.content)
    except: return {"summary": "Error", "sentiment": 0, "notes": "N/A"}

# =========================
# UI LAYOUT
# =========================
tab1, tab2 = st.tabs(["📊 Earnings Radar", "🤖 AI Sentiment Analyzer"])

with tab1:
    st.title("📊 Earnings Radar")
    up_radar = st.file_uploader("Upload Files", type=["csv", "xlsx"], key="r_up")
    txt_radar = st.text_area("Tickers", "AAPL\nMSFT\nNVDA", key="r_txt")
    
    tickers = set(t.strip().upper() for t in txt_radar.replace(",", "\n").split() if t.strip())
    if up_radar:
        df_u = pd.read_csv(up_radar) if up_radar.name.endswith(".csv") else pd.read_excel(up_radar)
        for c in df_u.columns:
            if c.lower() in ["ticker", "symbol"]:
                tickers.update(df_u[c].dropna().astype(str).str.upper())
                break

    if st.button("Fetch Radar Data"):
        t_list = sorted(list(tickers))
        prog = st.progress(0.0)
        prices_2y = yf.download(t_list, period="2y", group_by="ticker", progress=False)
        
        raw_results = []
        for i, t in enumerate(t_list):
            try:
                p_df = prices_2y[t] if len(t_list) > 1 else prices_2y
                curr = safe_float(p_df["Close"].iloc[-1])
                nxt = get_next_earnings(t)
                hist = finnhub_past_earnings(t)
                
                earn_rows = []
                if not hist.empty:
                    for _, r in hist.iterrows():
                        earn_rows.append({
                            "Date": r["date"].date(),
                            "EPS Actual": r.get("actual"),
                            "1D Reaction %": reaction(p_df, r["date"], 1),
                            "3D Reaction %": reaction(p_df, r["date"], 3),
                        })
                else:
                    earn_rows.append({"Date": None, "EPS Actual": None, "1D Reaction %": None, "3D Reaction %": None})

                raw_results.append({
                    "Ticker": t,
                    "Current Price": curr,
                    "Δ vs 52W High %": pct(curr, p_df["High"].iloc[-252:].max()),
                    "Next Earnings": nxt,
                    "EarningsData": earn_rows
                })
            except: pass
            prog.progress((i + 1) / len(t_list))

        # --- TRUNCATE / FLATTEN LOGIC RESTORED ---
        flattened = []
        for r in raw_results:
            for e in r["EarningsData"]:
                row = {k: v for k, v in r.items() if k != "EarningsData"}
                row.update(e)
                flattened.append(row)
        
        st.dataframe(pd.DataFrame(flattened), use_container_width=True)

with tab2:
    st.title("🤖 AI Sentiment Analyzer")
    up_ai = st.file_uploader("Upload CSV/Excel (Ticker & URL columns)", type=["csv", "xlsx"], key="ai_up")
    
    if up_ai:
        df_ai = pd.read_csv(up_ai) if up_ai.name.endswith(".csv") else pd.read_excel(up_ai)
        # Robust column detection
        cols = {c.lower().strip(): c for c in df_ai.columns}
        t_col = cols.get("ticker")
        u_col = cols.get("url")

        if st.button("Analyze Transcripts"):
            if not t_col or not u_col:
                st.error(f"Missing columns. Found: {list(df_ai.columns)}. Need 'Ticker' and 'URL'.")
            else:
                ai_final = []
                prog_ai = st.progress(0.0)
                for idx, row in df_ai.iterrows():
                    tick = str(row[t_col])
                    url = str(row[u_col])
                    st.write(f"Processing {tick}...")
                    text = fetch_transcript_text(url)
                    res = analyze_with_groq(tick, text)
                    ai_final.append({"Ticker": tick, "Sentiment": res.get("sentiment"), "Summary": res.get("summary"), "Notes": res.get("notes")})
                    prog_ai.progress((idx + 1) / len(df_ai))
                
                res_df = pd.DataFrame(ai_final)
                st.subheader("Results")
                st.dataframe(res_df, use_container_width=True) # Display on website
                st.download_button("Download Report", res_df.to_csv(index=False), "ai_report.csv")
