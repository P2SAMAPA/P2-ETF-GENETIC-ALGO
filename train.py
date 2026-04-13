import pandas as pd
import numpy as np
import os, json
from engine import GeneticEngine
from huggingface_hub import hf_hub_download, HfApi

FI_ASSETS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
EQ_ASSETS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLRE", "XLB", "XLP", "XLU", "GDX", "XME", "IWM"]
MACROS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

def clean(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, list): return [clean(x) for x in obj]
    return obj

def main():
    token = os.getenv("HF_TOKEN")
    repo_data = "P2SAMAPA/fi-etf-macro-signal-master-data"
    repo_results = "P2SAMAPA/p2-etf-genetic-algo-results"
    path = hf_hub_download(repo_id=repo_data, filename="master_data.parquet", repo_type="dataset", token=token)
    df = pd.read_parquet(path)
    final_results = {"FI": [], "EQ": []}
    for uni, assets, bench in [("FI", FI_ASSETS, "AGG"), ("EQ", EQ_ASSETS, "SPY")]:
        engine = GeneticEngine(assets, bench, MACROS)
        for year in range(2008, 2025):
            win_df = df[df.index >= f"{year}-01-01"]
            if len(win_df) < 50: continue
            logic, fit = engine.evolve(win_df)
            if logic:
                final_results[uni].append({"start_year": int(year), "logic": clean(logic), "fitness": clean(fit)})
    with open("strategy_results.json", "w") as f:
        json.dump(final_results, f)
    HfApi().upload_file(path_or_fileobj="strategy_results.json", path_in_repo="strategy_results.json", repo_id=repo_results, repo_type="dataset", token=token)

if __name__ == "__main__":
    main()
