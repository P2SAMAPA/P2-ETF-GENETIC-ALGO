# P2-ETF-GENETIC-ALGO

An evolutionary engine designed to predict the highest return ETF for the next US market trading date by optimizing macro-signal logic across multiple time horizons.

## 🚀 Engine Objectives
- **Target:** Predict the top-performing ETF for the next NYSE date.
- **Universes:**
    - **Option A (Fixed Income/Alts):** TLT, LQD, HYG, VNQ, GLD, SLV, VCIT vs. AGG.
    - **Option B (Equity):** QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLRE, XLB, XLP, XLU, GDX, XME, IWM vs. SPY.
- **Optimization:** Evaluates 1-day, 3-day, and 5-day holding periods to select the winner.

## 🧠 Methodology
1. **Shrinking Windows:** The engine runs 17 distinct iterations starting from 2008 through 2024.
2. **80/10/10 Split:** Every window is strictly partitioned into Training (80%), Validation (10%), and Test (10%).
3. **Genetic Algorithm:** Evolves logic gates based on 6 macro features: `VIX`, `DXY`, `T10Y2Y`, `TBILL_3M`, `IG_SPREAD`, and `HY_SPREAD`.
4. **Weighted Fitness (60/20/20):**
    - 60% Total Returns
    - 20% Sharpe Ratio
    - 20% Max Drawdown (Lower is better)
5. **Hard Filter:** Any year or window resulting in a negative total return is automatically excluded from the weighting process.
6. **Cost Modeling:** Includes a transaction slippage/cost of 12–15 bps per trade.

## 🛠 Project Structure
- `app.py`: Streamlit dashboard (SAMBA Style) featuring the NYSE market calendar.
- `engine.py`: The core Genetic Algorithm and backtesting logic.
- `train.py`: Headless training script for automated GitHub Actions execution.
- `.github/workflows/daily_run.yml`: Automated trigger at 22:15 UTC.

## 📊 Data Pipeline
- **Source:** `fi-etf-macro-signal-master-data` (Hugging Face).
- **Destination:** `p2-etf-genetic-algo-results` (Hugging Face).
