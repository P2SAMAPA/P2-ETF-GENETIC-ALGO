import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os, json
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide", page_title="SAMBA Engine")

# Styles
st.markdown("""
    <style>
    .main { background-color: #faf8f5; }
    .hero { background: white; border-radius: 15px; padding: 25px; border: 1px solid #e0e0e0; margin-bottom: 20px; }
    .conviction-text { color: #7D49F8; font-size: 28px; font-weight: bold; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #f0f0f0; text-align: center; }
    </style>
""", unsafe_allow_html=True)

def get_market_info():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now()+timedelta(days=7))
    next_date = schedule.index[0].strftime('%Y-%m-%d')
    return next_date

next_market_date = get_market_info()

st.title("SAMBA — Graph-Mamba ETF Engine")
st.caption("Genetic Algorithm Optimization · 60/20/20 Fitness · 1d/3d/5d Horizons")

tab1, tab2 = st.tabs(["🌊 Option A — FI / Alts", "🌊 Option B — Equity Sectors"])

def render_ui(label, etf_pick, conviction):
    with st.container():
        st.markdown(f'<div class="hero">', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"<h1>{etf_pick}</h1>", unsafe_allow_html=True)
            st.markdown(f'<p class="conviction-text">{conviction}% conviction</p>', unsafe_allow_html=True)
            st.write(f"Signal for {next_market_date} · Generated {datetime.now().strftime('%H:%M')} UTC")
            st.markdown(f'<span style="background:#F0E6FF; color:#7D49F8; padding:5px 10px; border-radius:15px; font-size:12px;">Source: Shrinking Window</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [("ANN RETURN", "14.2%"), ("ANN VOL", "12.1%"), ("SHARPE", "1.15"), ("MAX DD", "-8.4%"), ("HIT RATE", "58%")]
    for i, (m_label, m_val) in enumerate(metrics):
        with cols[i]:
            st.markdown(f'<div class="metric-card"><p style="color:gray; font-size:12px;">{m_label}</p><h3>{m_val}</h3></div>', unsafe_allow_html=True)

with tab1:
    render_ui("FI", "VNQ", "97.6") # Example state, will pull from strategy_results.json
with tab2:
    render_ui("EQ", "QQQ", "88.4")
