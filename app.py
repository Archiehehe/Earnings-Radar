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

st.set_page_config(page_title="Earnings Radar & AI Sentiment", layout="wide")

# =========================
# GENERAL HELPERS
# =========================
def safe_float(x):
    try: return float(x)
    except: return None

def pct(a, b):
    if a is None or b in (None, 0): return None
    return (a - b) / b * 100

def is_future(date_obj):
    """Checks if a date is today or in the future"""
    if date_obj is None: return False
    return date_obj >= datetime.now().date()

def format_market_cap(val):
    """Formats large numbers to M, B, or T"""
    if val is None or not isinstance(val, (int, float)): return "N/A"
    if val >= 1e12: return f"{val / 1e12:.2f}T"
    elif val >= 1e9: return f"{val / 1e9:.2f}B"
    elif val >= 1e6: return f"{val / 1e6:.2f}M"
    return f"{val:.2f}"

# =========================
# RADAR LOGIC (TAB 1)
# =========================
def get_next_earnings(ticker):
    """Attempts to find the next valid FUTURE earnings date"""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is not None and not cal.empty and 'Earnings Date' in cal.index:
            dates = cal.loc['Earnings Date']
            d_list = dates if isinstance(dates, (list, pd.Series, pd.Index)) else [dates]
            for d in d_list:
                dt = pd.to_datetime(d).date()
                if is_future(dt): return dt
        
        # Fallback to Scrape
        url = f"https://finance.yahoo.com/quote/{ticker}"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        import re
        match = re.search(r'(\w{3}\s+\d{1,2},\s+\d{4})', r.text)
        if match:
            dt = pd.to_datetime(match.group(1)).date()
            if is_future(dt): return dt
    except: pass
    return "TBD"

def reaction(price_df, earnings_date, days_after):
    """Finds reaction using the next available trading days"""
    try:
        price_df.index = pd.to_datetime(price_df.index).normalize()
        e_date = pd.to_datetime(earnings_date).normalize()
        
        pre_data = price_df.loc[:e_date]
        if pre_data.empty: return None
        pre_close = pre_data.iloc[-1]["Close"]
        
        post_data = price_df.loc[e_date + timedelta(days=1):]
        if post_data.empty: return None
        
        idx = min(days_after - 1, len(post_data) - 1)
        post_close = post_data.iloc[idx]["Close"]
        return pct(post_close, pre_close)
    except: return None

# =========================
# AI SENTIMENT LOGIC (TAB 2)
# =========================
def fetch_transcript_text(url):
    """Scrapes paragraph text from a URL"""
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        text = " ".join([p.get_text() for p in soup.find_all('p')])
        return text[:12000] # Limit characters for the AI context window
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_with_groq(ticker, text):
    """Calls Groq AI for free sentiment analysis"""
    if not GROQ_API_KEY:
        return {"summary": "Missing API Key", "sentiment": 3, "notes": "Add GROQ_API_KEY to Secrets"}
    
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    Analyze this earnings transcript for {ticker}. Provide:
    1. A 2-sentence summary of the results.
    2. A sentiment score (1 to 5) where 1 is bearish and 5 is bullish.
    3. Two bullet points of key risks or catalysts.
    Return ONLY a JSON object with keys: 'summary', 'sentiment', 'notes'.
    
    Transcript: {text[:5000]}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(chat_completion.choices[0].message.content)
    except:
        return {"summary": "AI Analysis Failed", "sentiment": 0, "notes": "N/A"}

# =========================
# STREAMLIT UI
# =========================
tab1, tab2 = st.tabs(["📊 Earnings Radar", "🤖 AI Sentiment Analyzer"])

# ---- TAB 1: RADAR ----
with tab1:
    st.subheader("Stock Performance & Upcoming Dates")
    
    up_radar = st.file_uploader("Upload Tickers (CSV/XLSX)", type=["csv", "xlsx"], key="radar_up")
    txt_radar = st.text_area("Or enter Tickers (one per line)", "AAPL\nMSFT\nNVDA", key="radar_txt")
    
    tickers = set(t.strip().upper() for t in txt_radar.replace(",", "\n").split() if t.strip())
    if up_radar:
        df_up = pd.read_csv(up_radar) if up_radar.name.endswith(".csv") else pd.read_excel(up_radar)
        for c in ["Ticker", "Symbol", "ticker"]:
            if c in df_up.columns:
                tickers.update(df_up[c].dropna().astype(str).str.upper())
                break

    if st.button("Run Radar Scan"):
        t_list = sorted(list(tickers))
        if not t_list:
            st.warning("Please provide tickers.")
        else:
            prog = st.progress(0.0)
            prices = yf.download(t_list, period="2y", group_by="ticker", progress=False)
            
            final_data = []
            for i, t in enumerate(t_list):
                try:
                    p_df = prices[t] if len(t_list) > 1 else prices
                    curr = safe_float(p_df["Close"].iloc[-1])
                    mcap_raw = safe_float(yf.Ticker(t).fast_info.get("market_cap"))
                    nxt = get_next_earnings(t)
                    
                    # Fetch Historicals from Finnhub
                    fh_url = f"https://finnhub.io/api/v1/stock/earnings?symbol={t}&token={FINNHUB_API_KEY}"
                    hist = requests.get(fh_url).json()[:3] # Get last 3 reports
                    
                    for r in hist:
                        final_data.append({
                            "Ticker": t,
                            "Market Cap": format_market_cap(mcap_raw),
                            "Current Price": curr,
                            "Next Earnings": nxt,
                            "Report Date": r.get("period"),
                            "EPS Actual": r.get("actual"),
                            "1D Reaction %": reaction(p_df, r.get("period"), 1),
                            "3D Reaction %": reaction(p_df, r.get("period"), 3),
                            "Δ vs 52W High %": pct(curr, p_df["High"].iloc[-252:].max())
                        })
                except: pass
                prog.progress((i + 1) / len(t_list))
            
            res_df = pd.DataFrame(final_data)
            pct_cols = ["1D Reaction %", "3D Reaction %", "Δ vs 52W High %"]
            st.dataframe(res_df, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.2f%%") for c in pct_cols})

# ---- TAB 2: AI SENTIMENT ----
with tab2:
    st.subheader("AI Earnings Sentiment")
    st.write("Upload a CSV with columns: **Ticker** and **URL**.")
    
    up_sent = st.file_uploader("Upload CSV with Transcript URLs", type=["csv"], key="sent_up")
    
    if up_sent:
        df_sent = pd.read_csv(up_sent)
        if st.button("Start AI Analysis"):
            if "Ticker" not in df_sent.columns or "URL" not in df_sent.columns:
                st.error("Error: CSV must contain 'Ticker' and 'URL' columns.")
            else:
                ai_results = []
                sent_prog = st.progress(0.0)
                for idx, row in df_sent.iterrows():
                    ticker = row['Ticker']
                    st.write(f"Processing {ticker}...")
                    transcript = fetch_transcript_text(row['URL'])
                    analysis = analyze_with_groq(ticker, transcript)
                    
                    ai_results.append({
                        "Ticker": ticker,
                        "Sentiment Score": analysis.get("sentiment"),
                        "AI Summary": analysis.get("summary"),
                        "Key Risks/Notes": analysis.get("notes")
                    })
                    sent_prog.progress((idx + 1) / len(df_sent))
                
                final_ai_df = pd.DataFrame(ai_results)
                st.dataframe(final_ai_df, use_container_width=True)
                
                # Visual Chart
                if not final_ai_df.empty:
                    st.bar_chart(final_ai_df.set_index("Ticker")["Sentiment Score"])
                
                st.download_button("Download AI Report", final_ai_df.to_csv(index=False), "ai_sentiment.csv")
