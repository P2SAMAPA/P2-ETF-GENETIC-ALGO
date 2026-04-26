import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from engine import GeneticEngine
from huggingface_hub import hf_hub_download, HfApi

FI_ASSETS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
EQ_ASSETS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLRE", "XLB", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM"]
MACROS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

def clean_numpy(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [clean_numpy(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    return obj

def calculate_metrics(returns, risk_free_rate=0.02):
    if len(returns) < 10:
        return {'annual_return': 0.0, 'annual_volatility': 0.0, 'sharpe': 0.0,
                'max_drawdown': 0.0, 'hit_rate': 0.0, 'sortino': 0.0}
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 10:
        return {'annual_return': 0.0, 'annual_volatility': 0.0, 'sharpe': 0.0,
                'max_drawdown': 0.0, 'hit_rate': 0.0, 'sortino': 0.0}

    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    annual_vol = returns.std() * np.sqrt(252)

    excess = returns - (risk_free_rate / 252)
    sharpe = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)

    downside = excess[excess < 0].std()
    sortino = (excess.mean() * 252) / (downside * np.sqrt(252) + 1e-9) if downside > 0 else 0.0

    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret - peak) / peak
    max_dd = abs(drawdown.min()) if not pd.isna(drawdown.min()) else 0.0
    hit_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

    return {
        'annual_return': float(annual_return * 100),
        'annual_volatility': float(annual_vol * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd * 100),
        'hit_rate': float(hit_rate * 100)
    }

def train_ga_engine(df, assets, benchmark, mode_name, data_slice_func):
    data = data_slice_func(df)
    if len(data) < 252:
        return None

    engine = GeneticEngine(assets, benchmark, MACROS)
    logic, fitness = engine.evolve(data)

    if logic is None:
        return None

    returns = engine.run_backtest(data, logic)
    metrics = calculate_metrics(returns)

    return {
        'mode': mode_name,
        'logic': clean_numpy(logic),
        'fitness': float(fitness),
        'metrics': metrics,
        'training_data_points': len(data),
        'training_start': str(data.index[0].date()),
        'training_end': str(data.index[-1].date())
    }

def daily_slice(df):
    return df.iloc[-504:]

def fixed_slice(df):
    return df

def shrinking_windows_slice(df, assets, benchmark):
    """Compute shrinking windows consensus with the new GA engine."""
    results = []
    for start_year in range(2008, 2025):
        window_df = df[(df.index >= f"{start_year}-01-01") & (df.index <= "2024-12-31")]
        if len(window_df) < 252:
            continue
        engine = GeneticEngine(assets, benchmark, MACROS)
        logic, fitness = engine.evolve(window_df)
        if logic:
            returns = engine.run_backtest(window_df, logic)
            metrics = calculate_metrics(returns)
            results.append({
                'window_start': start_year,
                'window_end': 2024,
                'logic': clean_numpy(logic),
                'fitness': float(fitness),
                'metrics': metrics,
                'ticker': logic[3]
            })
    return results

def consensus_from_shrinking(shrinking_results):
    """Pick the most frequently chosen ETF across windows."""
    if not shrinking_results:
        return None
    valid = [r for r in shrinking_results if r.get('fitness', -999) > -10]
    if not valid:
        valid = shrinking_results
    vote = {}
    for r in valid:
        t = r['ticker']
        vote[t] = vote.get(t, 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(valid) * 100
    metrics_avg = {}
    pick_windows = [r for r in valid if r['ticker'] == pick]
    if pick_windows:
        for key in pick_windows[0]['metrics']:
            metrics_avg[key] = np.mean([r['metrics'][key] for r in pick_windows])
    return {
        'ticker': pick,
        'conviction': conviction,
        'num_windows': len(valid),
        'num_pick_windows': vote[pick],
        'metrics': metrics_avg,
        'windows': shrinking_results
    }

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        return

    repo_data = "P2SAMAPA/fi-etf-macro-signal-master-data"
    repo_results = "P2SAMAPA/p2-etf-genetic-algo-results"

    print("Loading master data...")
    try:
        path = hf_hub_download(repo_id=repo_data, filename="master_data.parquet",
                               repo_type="dataset", token=token)
        df = pd.read_parquet(path)
        print(f"Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    final_results = {"FI": {}, "EQ": {}}

    for universe_name, assets, benchmark in [("FI", FI_ASSETS, "AGG"), ("EQ", EQ_ASSETS, "SPY")]:
        print(f"\n=== {universe_name} Universe ===")

        # Daily mode
        daily = train_ga_engine(df, assets, benchmark, "Daily (504d)", daily_slice)
        if daily:
            final_results[universe_name]["daily"] = daily

        # Fixed mode
        fixed = train_ga_engine(df, assets, benchmark, "Fixed (2008‑YTD)", fixed_slice)
        if fixed:
            final_results[universe_name]["fixed"] = fixed

        # Shrinking windows consensus
        shrinking = shrinking_windows_slice(df, assets, benchmark)
        consensus = consensus_from_shrinking(shrinking)
        if consensus:
            final_results[universe_name]["shrinking"] = consensus

    output_file = "strategy_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    try:
        api = HfApi()
        api.upload_file(path_or_fileobj=output_file, path_in_repo=output_file,
                        repo_id=repo_results, repo_type="dataset", token=token)
        print(f"Uploaded to {repo_results}")
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == "__main__":
    main()
