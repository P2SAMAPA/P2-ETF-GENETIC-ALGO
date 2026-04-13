import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import json
from huggingface_hub import hf_hub_download

# UI Config
st.set_page_config(layout="wide", page_title="SAMBA — Graph-Mamba ETF Engine")

# Correct SAMBA Styles
st.markdown("""
    <style>
    .main { background-color: #faf8f5; }
    .hero { background: white; border-radius: 12px; padding: 35px; border: 1px solid #E6E6F2; margin-bottom: 25px; }
    .ticker { font-size: 72px; font-weight: 800; color: #0E1117; line-height: 1; }
    .conviction-text { color: #7D49F8; font-size: 24px; font-weight: 600; margin-top: 10px; }
    .source-badge { background: #F0E6FF; color: #7D49F8; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; display: inline-block; margin-top: 15px; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #F0F0F5; text-align: center; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        token = st.secrets.get("HF_TOKEN") # or os.getenv
        path = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-genetic-algo-results", 
            filename="strategy_results.json", 
            repo_type="dataset", 
            token=token
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def get_next_nyse_date():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now()+timedelta(days=7))
    return schedule.index[0].strftime('%Y-%m-%d')

# Header
st.markdown('<h1 style="margin-bottom:0;">SAMBA — Graph-Mamba ETF Engine</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5E6271;">Genetic Algorithm Optimization · 60/20/20 Fitness · 1d/3d/5d Horizons</p>', unsafe_allow_html=True)

data = load_latest_results()
next_date = get_next_nyse_date()

tab_a, tab_b = st.tabs(["🌊 Option A — FI / Alts", "🌊 Option B — Equity Sectors"])

def render_module(module_key):
    if data is None or module_key not in data or not data[module_key]:
        st.warning(f"Training for {module_key} module is currently in progress or results are missing.")
        return

    # Extract top pick from the latest shrinking window (the one with the highest start_year)
    sorted_windows = sorted(data[module_key], key=lambda x: x['start_year'], reverse=True)
    latest_run = sorted_windows[0]
    
    # logic format: [Macro, Op, Thresh, ETF, Horizon]
    ticker = latest_run['logic'][3] 
    # Conviction logic: % of windows agreeing on this ETF
    total_windows = len(data[module_key])
    agreement = sum(1 for w in data[module_key] if w['logic'][3] == ticker)
    conviction = round((agreement / total_windows) * 100, 1)

    st.markdown(f"""
        <div class="hero">
            <div class="ticker">{ticker}</div>
            <div class="conviction-text">{conviction}% conviction</div>
            <div style="color:#8C91A1; font-size:14px; margin-top:15px;">
                Signal for {next_date} · Generated {datetime.now().strftime('%H:%M')} UTC
            </div>
            <div class="source-badge">Source: Shrinking Window</div>
        </div>
    """, unsafe_allow_html=True)

    # Metrics (These would ideally be calculated during training and saved in the JSON)
    m1, m2, m3, m4, m5 = st.columns(5)
    for i, col in enumerate([m1, m2, m3, m4, m5]):
        with col:
            st.markdown('<div class="metric-card"><p style="color:gray; font-size:12px;">ANN RETURN</p><h3>--</h3></div>', unsafe_allow_html=True)

with tab_a:
    render_module("FI")

with tab_b:
    render_module("EQ")
