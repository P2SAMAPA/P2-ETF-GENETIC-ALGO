# P2-ETF-GENETIC-ALGO

An evolutionary engine that predicts the highest‑reward ETF for the next US market trading date by optimising macro‑signal logic across multiple time horizons.

## 🚀 Engine Objectives

- **Target:** Predict the top‑performing ETF for the next NYSE trading date.  
- **Universes:**  
  - **Option A – Fixed Income / Alternatives:** TLT, LQD, HYG, VNQ, GLD, SLV, VCIT, CASH vs. AGG.  
  - **Option B – Equity Sectors:** QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLRE, XLB, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, CASH vs. SPY.  
- **Optimisation:** Evaluates 1‑day, 3‑day and 5‑day holding periods within a single rule.

## 🧠 Methodology (improved v2)

- **Walk‑Forward Sortino Fitness:** The Genetic Algorithm evaluates each candidate rule using **walk‑forward cross‑validation** (5 folds). The fitness function is the **annualised Sortino ratio** (downside‑only volatility), not raw returns or Sharpe. This directly penalises strategies with large drawdowns.  
- **Population & Diversity:** Population size = 200, tournament selection (size 7), elitism of 20 individuals, and adaptive mutation rates. Immigration is applied implicitly via mutation strength.  
- **Cash Gene:** The chromosome includes a **CASH** asset. The GA can choose to hold cash when no ETF offers a favourable risk‑adjusted outlook.  
- **Three Training Modes:**  
  - **Daily (504 days):** Trained on the most recent 2 years to capture the current regime.  
  - **Fixed (2008‑YTD):** Trained on the entire available history for long‑term robustness.  
  - **Shrinking Windows Consensus:** Runs the GA on 17 rolling windows (2008‑2024), then picks the most frequently selected ETF across windows, with conviction scoring.  
- **Macro Features:** VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD.  
- **Transaction Cost:** 13.5 bps per trade.  

## 📊 Dashboard

The Streamlit app (`app.py`) displays **three sub‑tabs per universe**:

- **📅 Daily (504d):** The best rule from the recent 2‑year walk‑forward training.  
- **📆 Fixed (2008‑YTD):** The best rule from the full historical dataset.  
- **🔄 Shrinking Consensus:** Consensus ETF across all shrinking windows, with conviction and performance metrics.  

Each tab shows a hero card with the selected ETF (or CASH), full backtest metrics, and the decoded trading rule.

## 🛠 Project Structure
P2-ETF-GENETIC-ALGO/
├── app.py # Streamlit dashboard
├── engine.py # Core Genetic Algorithm with walk‑forward Sortino fitness
├── train.py # Headless training (daily + fixed + shrinking)
├── requirements.txt
├── README.md
├── .github/workflows/
│ └── daily_run.yml # Automated trigger at 22:15 UTC

text

## 📦 Data Pipeline

- **Source:** `P2SAMAPA/fi‑etf‑macro‑signal‑master‑data` (master_data.parquet)
- **Destination:** `P2SAMAPA/p2‑etf‑genetic‑algo‑results` (strategy_results.json)
