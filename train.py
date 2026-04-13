import pandas as pd
import numpy as np
import os
import json
from engine import GeneticEngine
from huggingface_hub import hf_hub_download, HfApi

FI_ASSETS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
EQ_ASSETS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLRE", "XLB", "XLP", "XLU", "GDX", "XME", "IWM"]
MACROS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

def convert_types(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def main():
    token = os.getenv("HF_TOKEN")
    repo_data = "P2SAMAPA/fi-etf-macro-signal-master-data"
    repo_results = "P2SAMAPA/p2-etf-genetic-algo-results"
    file_path = hf_hub_download(repo_id=repo_data, filename="master_data.parquet", repo_type="dataset", token=token)
    df = pd.read_parquet(file_path)
    final_results = {"FI": [], "EQ": []}
    for universe, assets, bench in [("FI", FI_ASSETS, "AGG"), ("EQ", EQ_ASSETS, "SPY")]:
        engine = GeneticEngine(assets, bench, MACROS)
        for start_year in range(2008, 2025):
            window_df = df[df.index >= f"{start_year}-01-01"]
            if len(window_df) < 50: continue
            best_logic, fitness = engine.evolve(window_df)
            if best_logic:
                clean_logic = [convert_types(x) for x in best_logic]
                final_results[universe].append({
                    "start_year": int(start_year),
                    "logic": clean_logic,
                    "fitness": float(fitness)
                })
    with open("strategy_results.json", "w") as f:
        json.dump(final_results, f, default=convert_types)
    api = HfApi()
    api.upload_file(path_or_fileobj="strategy_results.json", path_in_repo="strategy_results.json", 
                    repo_id=repo_results, repo_type="dataset", token=token)

if __name__ == "__main__":
    main()
