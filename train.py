import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from engine import GeneticEngine
from huggingface_hub import hf_hub_download, HfApi

FI_ASSETS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
EQ_ASSETS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLRE", "XLB", "XLP", "XLU", "GDX", "XME", "IWM"]
MACROS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

def clean_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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

def calculate_metrics(returns):
    """Calculate actual performance metrics from returns series"""
    if len(returns) < 10:
        return {
            'annual_return': 0,
            'annual_volatility': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'hit_rate': 0
        }
    
    # Remove NaN/inf
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Annualized volatility
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
    
    # Maximum drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    # Hit rate (percentage of positive days)
    hit_rate = (returns > 0).sum() / len(returns)
    
    return {
        'annual_return': float(annual_return * 100),
        'annual_volatility': float(annual_volatility * 100),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown * 100),
        'hit_rate': float(hit_rate * 100)
    }

def run_shrinking_window(df, assets, benchmark, window_name, start_year, end_year=2024):
    """Run genetic algorithm on a specific time window"""
    window_df = df[(df.index >= f"{start_year}-01-01") & (df.index <= f"{end_year}-12-31")]
    
    if len(window_df) < 252:  # Need at least 1 year of data
        return None
    
    engine = GeneticEngine(assets, benchmark, MACROS)
    logic, fitness = engine.evolve(window_df)
    
    if logic:
        # Calculate actual metrics from backtest
        returns = engine.run_backtest(window_df, logic)
        metrics = calculate_metrics(returns)
        
        return {
            'window_name': window_name,
            'start_year': start_year,
            'end_year': end_year,
            'logic': clean_numpy(logic),
            'fitness': float(fitness),
            'metrics': metrics
        }
    
    return None

def run_fixed_dataset(df, assets, benchmark, start_year=2008, end_year=2026):
    """Run on fixed dataset 2008-2026YTD"""
    # Get data through current date
    current_date = datetime.now()
    end_date = current_date if current_date.year <= end_year else f"{end_year}-12-31"
    
    fixed_df = df[(df.index >= f"{start_year}-01-01") & (df.index <= end_date)]
    
    if len(fixed_df) < 252:
        return None
    
    engine = GeneticEngine(assets, benchmark, MACROS)
    logic, fitness = engine.evolve(fixed_df)
    
    if logic:
        returns = engine.run_backtest(fixed_df, logic)
        metrics = calculate_metrics(returns)
        
        return {
            'window_name': f'Fixed {start_year}-{end_date.year}YTD',
            'start_year': start_year,
            'end_year': end_date.year,
            'logic': clean_numpy(logic),
            'fitness': float(fitness),
            'metrics': metrics
        }
    
    return None

def get_consensus_pick(all_windows):
    """Calculate consensus pick across all shrinking windows"""
    if not all_windows:
        return None
    
    picks = {}
    for window in all_windows:
        etf = window['logic'][3]
        picks[etf] = picks.get(etf, 0) + 1
    
    # Find most common ETF
    consensus_etf = max(picks, key=picks.get)
    conviction = (picks[consensus_etf] / len(all_windows)) * 100
    
    # Calculate average metrics for consensus picks
    consensus_metrics = {
        'annual_return': 0,
        'annual_volatility': 0,
        'sharpe': 0,
        'max_drawdown': 0,
        'hit_rate': 0
    }
    
    consensus_windows = [w for w in all_windows if w['logic'][3] == consensus_etf]
    for window in consensus_windows:
        for key in consensus_metrics:
            consensus_metrics[key] += window['metrics'][key]
    
    for key in consensus_metrics:
        consensus_metrics[key] /= len(consensus_windows)
    
    return {
        'etf': consensus_etf,
        'conviction': conviction,
        'num_windows': len(all_windows),
        'consensus_windows': len(consensus_windows),
        'metrics': consensus_metrics
    }

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        return
    
    repo_data = "P2SAMAPA/fi-etf-macro-signal-master-data"
    repo_results = "P2SAMAPA/p2-etf-genetic-algo-results"
    
    print("Loading master data...")
    try:
        path = hf_hub_download(
            repo_id=repo_data, 
            filename="master_data.parquet", 
            repo_type="dataset", 
            token=token
        )
        df = pd.read_parquet(path)
        print(f"Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    final_results = {
        "FI": {
            "shrinking_windows": [],
            "fixed_dataset": None,
            "consensus": None
        },
        "EQ": {
            "shrinking_windows": [],
            "fixed_dataset": None,
            "consensus": None
        }
    }
    
    # Run for both universes
    for universe_name, assets, benchmark in [
        ("FI", FI_ASSETS, "AGG"), 
        ("EQ", EQ_ASSETS, "SPY")
    ]:
        print(f"\nProcessing {universe_name} universe...")
        
        # 1. Run shrinking windows (2008-2024)
        shrinking_results = []
        for year in range(2008, 2025):
            print(f"  Running shrinking window starting {year}...")
            result = run_shrinking_window(df, assets, benchmark, f"Shrinking_{year}", year, 2024)
            if result:
                shrinking_results.append(result)
                print(f"    ✓ Fitness: {result['fitness']:.4f}")
        
        final_results[universe_name]["shrinking_windows"] = shrinking_results
        
        # 2. Calculate consensus from shrinking windows
        if shrinking_results:
            consensus = get_consensus_pick(shrinking_results)
            final_results[universe_name]["consensus"] = consensus
            print(f"  Consensus pick: {consensus['etf']} ({consensus['conviction']:.1f}% conviction)")
        
        # 3. Run fixed dataset (2008-2026YTD)
        print(f"  Running fixed dataset 2008-2026YTD...")
        fixed_result = run_fixed_dataset(df, assets, benchmark, 2008, 2026)
        if fixed_result:
            final_results[universe_name]["fixed_dataset"] = fixed_result
            print(f"    ✓ Fixed dataset pick: {fixed_result['logic'][3]}")
    
    # Save results
    output_file = "strategy_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Upload to Hugging Face
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=output_file,
            path_in_file=output_file,
            repo_id=repo_results,
            repo_type="dataset",
            token=token
        )
        print(f"Uploaded to {repo_results}")
    except Exception as e:
        print(f"ERROR uploading: {e}")

if __name__ == "__main__":
    main()
