import numpy as np
import pandas as pd

class GeneticEngine:
    def __init__(self, assets, benchmark, macros, cost_bps=13.5):
        self.assets = assets
        self.benchmark = benchmark
        self.macros = macros
        self.cost_bps = cost_bps / 10000
        self.pop_size = 50
        self.generations = 20

    def calculate_fitness(self, rets):
        if len(rets) < 10: return -1.0
        total_ret = (1 + rets).prod() - 1
        if total_ret <= 0: return -1.0
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
        cum_ret = (1 + rets).cumprod()
        peak = cum_ret.cummax()
        max_dd = abs(((cum_ret - peak) / peak).min())
        return (0.60 * float(total_ret)) + (0.20 * float(sharpe)) + (0.20 * (1 - float(max_dd)))

    def run_backtest(self, df, chromo):
        col, op, thresh, etf, horizon = chromo
        condition = (df[col] > thresh) if op == '>' else (df[col] < thresh)
        raw_signal = condition.astype(int).shift(1).fillna(0)
        signal = raw_signal.rolling(window=int(horizon), min_periods=1).max()
        asset_rets = df[etf].pct_change().fillna(0)
        trades = signal.diff().abs().fillna(0)
        return (signal * asset_rets) - (trades * self.cost_bps)

    def evolve(self, data):
        n = len(data)
        train_df = data.iloc[:int(n*0.8)]
        val_df = data.iloc[int(n*0.8):int(n*0.9)]
        horizons = [1, 3, 5]
        population = [[np.random.choice(self.macros), np.random.choice(['>', '<']), np.random.uniform(-2, 2), np.random.choice(self.assets), np.random.choice(horizons)] for _ in range(self.pop_size)]
        best_chromo, max_fit = None, -np.inf
        for gen in range(self.generations):
            fits = [self.calculate_fitness(self.run_backtest(train_df, c)) for c in population]
            indices = np.argsort(fits)[-10:]
            for idx in indices:
                v_fit = self.calculate_fitness(self.run_backtest(val_df, population[idx]))
                if v_fit > max_fit:
                    max_fit, best_chromo = v_fit, population[idx]
            new_pop = [population[i] for i in indices]
            while len(new_pop) < self.pop_size:
                child = list(population[np.random.choice(indices)])
                child[2] += np.random.normal(0, 0.1)
                if np.random.rand() < 0.1: child[4] = np.random.choice(horizons)
                new_pop.append(child)
            population = new_pop
        return best_chromo, max_fit
