import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os
import json
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide", page_title="P2-ETF-GENETIC-ALGO")

st.markdown("""
    <style>
    .main { background-color: #faf8f5; }
    .hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 35px; color: white; margin-bottom: 25px; }
    .ticker { font-size: 72px; font-weight: 800; line-height: 1; }
    .conviction-text { font-size: 24px; font-weight: 600; margin-top: 10px; opacity: 0.95; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E6E6F2; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-label { color: #8C91A1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #0E1117; margin-top: 8px; }
    .subheader { font-size: 20px; font-weight: 600; margin: 20px 0 15px 0; color: #2C3E50; }
    .divider { border-top: 1px solid #E6E6F2; margin: 20px 0; }
    .fitness-bad { color: #e74c3c; font-weight: 600; }
    .fitness-good { color: #27ae60; font-weight: 600; }
    .cash-hero { background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%); }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_results():
    try:
        token = os.getenv("HF_TOKEN")
        path = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-genetic-algo-results", 
            filename="strategy_results.json", 
            repo_type="dataset", 
            token=token
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load results: {str(e)}")
        return None

def get_next_trading_day():
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date=datetime.now(), 
            end_date=datetime.now() + timedelta(days=10)
        )
        if len(schedule) > 0:
            return schedule.index[0].strftime('%Y-%m-%d')
    except:
        pass
    return datetime.now().strftime('%Y-%m-%d')

def format_fitness(fitness):
    """Format Sortino fitness value with color coding."""
    if fitness < -10:
        return "Poor"
    elif fitness < 0:
        return f"{fitness:.2f}"
    else:
        return f"{fitness:.2f}"

def render_hero_card(ticker, mode_name, signal_date, generated_time, is_cash=False):
    """Display hero card with different styling for cash selection."""
    hero_class = "hero" if not is_cash else "hero cash-hero"
    ticker_display = "CASH" if is_cash else ticker
    conviction_text = "Holding Cash" if is_cash else f"GA‑Selected Rule"
    
    st.markdown(f'''
        <div class="{hero_class}">
            <div class="ticker">{ticker_display}</div>
            <div class="conviction-text">{conviction_text}</div>
            <div style="font-size: 14px; margin-top: 15px; opacity: 0.8;">
                Mode: {mode_name} · Signal for {signal_date} · Generated {generated_time}
            </div>
        </div>
    ''', unsafe_allow_html=True)

def render_metrics(metrics):
    """Display five key performance metrics."""
    cols = st.columns(5)
    metric_items = [
        ("Annual Return", f"{metrics.get('annual_return', 0):.1f}%"),
        ("Annual Vol", f"{metrics.get('annual_volatility', 0):.1f}%"),
        ("Sharpe", f"{metrics.get('sharpe', 0):.2f}"),
        ("Max DD", f"{metrics.get('max_drawdown', 0):.1f}%"),
        ("Hit Rate", f"{metrics.get('hit_rate', 0):.1f}%")
    ]
    for col, (label, value) in zip(cols, metric_items):
        with col:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
            ''', unsafe_allow_html=True)

def render_logic_card(logic):
    """Show the trading rule in a readable card."""
    if not logic or len(logic) < 5:
        return
    macro, operator, threshold, etf, horizon = logic
    action = "Buy" if etf != 'CASH' else "Hold"
    target = etf if etf != 'CASH' else "Cash"
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Strategy Logic</div>
            <div style="font-size: 14px; margin-top: 10px; text-align: left;">
                If {macro} {operator} {threshold:.2f}<br>
                Then {action.lower()} {target} for {horizon} days
            </div>
        </div>
    ''', unsafe_allow_html=True)

def render_mode_tab(mode_data, mode_name):
    """Display a single mode (daily or global) for a universe."""
    if not mode_data:
        st.warning(f"No {mode_name} data available.")
        return
    
    logic = mode_data['logic']
    metrics = mode_data['metrics']
    ticker = logic[3] if len(logic) > 3 else 'CASH'
    is_cash = (ticker == 'CASH')
    signal_date = get_next_trading_day()
    generated_time = datetime.now().strftime("%H:%M UTC")
    
    # Hero card
    render_hero_card(ticker, mode_name, signal_date, generated_time, is_cash)
    
    # Metrics
    st.markdown('<div class="subheader">📈 Performance Metrics</div>', unsafe_allow_html=True)
    render_metrics(metrics)
    
    # Logic and training info
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        render_logic_card(logic)
    with col2:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Training Data</div>
                <div style="font-size: 14px; margin-top: 10px; text-align: left;">
                    Period: {mode_data.get('training_start', 'N/A')} to {mode_data.get('training_end', 'N/A')}<br>
                    Observations: {mode_data.get('training_data_points', 0)}<br>
                    Sortino Fitness: {format_fitness(mode_data.get('fitness', 0))}
                </div>
            </div>
        ''', unsafe_allow_html=True)

def render_universe(data, universe_name):
    """Render a single universe with two sub-tabs: Daily and Global."""
    if not data:
        st.warning("No data loaded.")
        return
    
    universe_data = data.get(universe_name)
    if not universe_data:
        st.warning(f"No data available for {universe_name}.")
        return
    
    tab_daily, tab_global = st.tabs(["📅 Daily Trading (504d)", "🌍 Global Training (2008‑YTD)"])
    
    with tab_daily:
        render_mode_tab(universe_data.get('daily'), "Daily Trading")
    
    with tab_global:
        render_mode_tab(universe_data.get('global'), "Global Training")

# -----------------------------------------------------------------------
# Main App
st.markdown('<h1 style="margin-bottom:0;">P2-ETF-GENETIC-ALGO</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5E6271; margin-bottom: 30px;">Evolutionary ETF Predictor · Walk‑Forward Sortino Fitness · Daily & Global Modes</p>', unsafe_allow_html=True)

data = load_results()

if data:
    tab_fi, tab_eq = st.tabs(["🌊 Fixed Income / Alternatives", "⚡ Equity Sectors"])
    
    with tab_fi:
        render_universe(data, "FI")
    
    with tab_eq:
        render_universe(data, "EQ")
else:
    st.error("Unable to load strategy results. Please ensure you have run train.py and the Hugging Face dataset exists.")
