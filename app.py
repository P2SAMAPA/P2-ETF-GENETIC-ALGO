import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import os
import json
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide", page_title="P2-ETF-GENETIC-ALGO")

# Custom CSS
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
    .info-text { color: #5E6271; font-size: 14px; margin-bottom: 20px; }
    .divider { border-top: 1px solid #E6E6F2; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_results():
    """Load results from Hugging Face"""
    try:
        token = os.getenv("HF_TOKEN")
        if not token:
            token = None
        
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
    """Get next NYSE trading day"""
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

def format_metric(value, is_percentage=False):
    """Format metric values for display"""
    if is_percentage:
        return f"{value:.1f}%"
    return f"{value:.2f}"

def render_hero_card(ticker, conviction, signal_date, generated_time):
    """Render the hero section"""
    st.markdown(f'''
        <div class="hero">
            <div class="ticker">{ticker}</div>
            <div class="conviction-text">{conviction:.1f}% conviction</div>
            <div style="font-size: 14px; margin-top: 15px; opacity: 0.8;">
                Signal for {signal_date} · Generated {generated_time}
            </div>
        </div>
    ''', unsafe_allow_html=True)

def render_metrics(metrics):
    """Render metrics cards"""
    cols = st.columns(5)
    
    metric_items = [
        ("Annual Return", f"{metrics['annual_return']:.1f}%", True),
        ("Annual Vol", f"{metrics['annual_volatility']:.1f}%", True),
        ("Sharpe", f"{metrics['sharpe']:.2f}", False),
        ("Max DD", f"{metrics['max_drawdown']:.1f}%", True),
        ("Hit Rate", f"{metrics['hit_rate']:.1f}%", True)
    ]
    
    for col, (label, value, is_pct) in zip(cols, metric_items):
        with col:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
            ''', unsafe_allow_html=True)

def render_universe(data, universe_name, display_name):
    """Render a complete universe view (FI or EQ)"""
    if not data or universe_name not in data:
        st.warning(f"No data available for {display_name}. Please run train.py first.")
        return
    
    universe_data = data[universe_name]
    signal_date = get_next_trading_day()
    generated_time = datetime.now().strftime("%H:%M UTC")
    
    # Create tabs for Fixed Dataset and Shrinking Consensus
    tab_fixed, tab_shrinking = st.tabs([
        "📊 Fixed Dataset (2008-2026YTD)",
        "🔄 Shrinking Consensus (2008-2024)"
    ])
    
    # Tab 1: Fixed Dataset
    with tab_fixed:
        fixed = universe_data.get('fixed_dataset')
        if fixed:
            ticker = fixed['logic'][3]
            metrics = fixed['metrics']
            
            render_hero_card(ticker, 100.0, signal_date, generated_time)
            st.markdown('<div class="subheader">📈 Performance Metrics</div>', unsafe_allow_html=True)
            render_metrics(metrics)
            
            # Additional info
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Strategy Logic</div>
                        <div style="font-size: 14px; margin-top: 10px; text-align: left;">
                            If {fixed['logic'][0]} {fixed['logic'][1]} {fixed['logic'][2]:.2f}<br>
                            Then buy {fixed['logic'][3]} for {fixed['logic'][4]} days
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Backtest Period</div>
                        <div style="font-size: 24px; margin-top: 10px;">
                            {fixed['start_year']} - {fixed['end_year']}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("Fixed dataset results not available yet. Run train.py to generate.")
    
    # Tab 2: Shrinking Windows Consensus
    with tab_shrinking:
        consensus = universe_data.get('consensus')
        shrinking_windows = universe_data.get('shrinking_windows', [])
        
        if consensus and shrinking_windows:
            ticker = consensus['etf']
            conviction = consensus['conviction']
            metrics = consensus['metrics']
            
            render_hero_card(ticker, conviction, signal_date, generated_time)
            
            st.markdown('<div class="subheader">📊 Consensus Performance Metrics</div>', unsafe_allow_html=True)
            render_metrics(metrics)
            
            # Consensus details
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Windows Analyzed</div>
                        <div style="font-size: 32px;">{consensus['num_windows']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Windows Picking {ticker}</div>
                        <div style="font-size: 32px;">{consensus['consensus_windows']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Conviction Score</div>
                        <div style="font-size: 32px;">{conviction:.0f}%</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            # Show individual window picks (expandable)
            with st.expander("📋 View All Shrinking Window Results"):
                window_data = []
                for window in shrinking_windows:
                    window_data.append({
                        'Window': f"{window['start_year']}-{window['end_year']}",
                        'ETF': window['logic'][3],
                        'Fitness': f"{window['fitness']:.2f}",
                        'Ann Return': f"{window['metrics']['annual_return']:.1f}%",
                        'Sharpe': f"{window['metrics']['sharpe']:.2f}"
                    })
                st.dataframe(pd.DataFrame(window_data), use_container_width=True)
        else:
            st.info("Shrinking windows consensus not available yet. Run train.py to generate.")

# Main App
st.markdown('<h1 style="margin-bottom:0;">P2-ETF-GENETIC-ALGO</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5E6271; margin-bottom: 30px;">Evolutionary ETF Predictor · Genetic Algorithm Optimization · Multi-Horizon Macro Signals</p>', unsafe_allow_html=True)

# Load data
data = load_results()

if data:
    # Create tabs for FI and EQ universes
    tab_fi, tab_eq = st.tabs(["🌊 Option A — Fixed Income / Alternatives", "⚡ Option B — Equity Sectors"])
    
    with tab_fi:
        render_universe(data, "FI", "Fixed Income & Alternatives")
    
    with tab_eq:
        render_universe(data, "EQ", "Equity Sectors")
else:
    st.error("Unable to load strategy results. Please ensure you have run train.py and the Hugging Face dataset exists.")
