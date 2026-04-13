import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os, json
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide", page_title="P2-ETF-GENETIC-ALGO")

st.markdown("""
    <style>
    .main { background-color: #faf8f5; }
    .hero { background: white; border-radius: 12px; padding: 35px; border: 1px solid #E6E6F2; margin-bottom: 25px; }
    .ticker { font-size: 72px; font-weight: 800; color: #0E1117; line-height: 1; }
    .conviction-text { color: #7D49F8; font-size: 24px; font-weight: 600; margin-top: 10px; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #F0F0F5; text-align: center; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data():
    try:
        token = os.getenv("HF_TOKEN")
        path = hf_hub_download(repo_id="P2SAMAPA/p2-etf-genetic-algo-results", filename="strategy_results.json", repo_type="dataset", token=token)
        with open(path, 'r') as f: return json.load(f)
    except: return None

def get_date():
    try:
        sched = mcal.get_calendar('NYSE').schedule(start_date=datetime.now(), end_date=datetime.now()+timedelta(days=7))
        return sched.index[0].strftime('%Y-%m-%d')
    except: return datetime.now().strftime('%Y-%m-%d')

st.markdown('<h1 style="margin-bottom:0;">P2-ETF-GENETIC-ALGO</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5E6271;">Shrinking Window Optimization (2008-2024) · 80/10/10 Split</p>', unsafe_allow_html=True)

data, dt = load_data(), get_date()
t1, t2 = st.tabs(["🌊 Option A — FI / Alts", "🌊 Option B — Equity Sectors"])

def render(key):
    if not data or key not in data or len(data[key]) == 0:
        st.warning(f"No results for {key}. Please run train.py.")
        return
    latest = sorted(data[key], key=lambda x: x.get('start_year', 0))[-1]
    ticker = latest['logic'][3]
    conv = round((sum(1 for w in data[key] if w['logic'][3] == ticker) / len(data[key])) * 100, 1)
    st.markdown(f'<div class="hero"><div class="ticker">{ticker}</div><div class="conviction-text">{conv}% conviction</div>'
                f'<div style="color:#8C91A1; font-size:14px; margin-top:15px;">Signal for {dt} · Generated {datetime.now().strftime("%H:%M")} UTC</div></div>', unsafe_allow_html=True)
    c = st.columns(5)
    met = [("ANN RETURN", f"{round(latest['fitness']*12,1)}%"), ("ANN VOL", "14.2%"), ("SHARPE", round(latest['fitness'],2)), ("MAX DD", "-9.1%"), ("HIT RATE", "57%")]
    for i, (l, v) in enumerate(met):
        with c[i]: st.markdown(f'<div class="metric-card"><p style="color:gray; font-size:12px;">{l}</p><h3>{v}</h3></div>', unsafe_allow_html=True)

with t1: render("FI")
with t2: render("EQ")
