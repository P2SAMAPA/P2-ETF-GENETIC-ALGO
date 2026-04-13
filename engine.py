import numpy as np
import pandas as pd
from scipy.stats import norm

class GeneticEngine:
    def __init__(self, assets, benchmark, macros, cost_bps=13.5):
        self.assets = assets
        self.benchmark = benchmark
        self.macros = macros
        self.cost_bps = cost_bps / 10000
        self.pop_size = 50
        self.generations = 20

    def calculate_fitness(self, rets):
        """Applies 60/20/20 weighting and filters negative returns."""
        if len(rets) < 10: return -1.0
        
        total_ret = (1 + rets).prod() - 1
        if total_ret <= 0: return -1.0  # Exclude negative return years
        
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
        cum_ret = (1 + rets).cumprod()
        peak = cum_ret.cummax()
        max_dd = abs(((cum_ret - peak) / peak).min())
        
        # Fitness = 60% Return + 20% Sharpe + 20% (1 - MaxDD)
        fitness = (0.60 * total_ret) + (0.20 * sharpe) + (0.20 * (1 - max_dd))
        return fitness

    def run_backtest(self, df, chromo):
        """Simulates logic: If Macro [Op] Thresh, Long ETF for [H] days."""
        col, op, thresh, etf, horizon = chromo
        condition = (df[col] > thresh) if op == '>' else (df[col] < thresh)
        
        raw_signal = condition.astype(int).shift(1).fillna(0)
        # Apply holding period (horizon)
        signal = raw_signal.rolling(window=horizon, min_periods=1).max()
        
        asset_rets = df[etf].pct_change().fillna(0)
        trades = signal.diff().abs().fillna(0)
        net_rets = (signal * asset_rets) - (trades * self.cost_bps)
        return net_rets

    def evolve(self, data):
        """80/10/10 Split and GA evolution."""
        n = len(data)
        train_df = data.iloc[:int(n*0.8)]
        val_df = data.iloc[int(n*0.8):int(n*0.9)]
        # test_df = data.iloc[int(n*0.9):] # Reserved for OOS reporting

        # Population: [Macro, Op, Thresh, ETF, Horizon]
        horizons = [1, 3, 5]
        population = []
        for _ in range(self.pop_size):
            population.append([
                np.random.choice(self.macros),
                np.random.choice(['>', '<']),
                np.random.uniform(-2, 2),
                np.random.choice(self.assets),
                np.random.choice(horizons)
            ])

        best_chromo = None
        max_fit = -np.inf

        for gen in range(self.generations):
            fits = []
            for chromo in population:
                t_rets = self.run_backtest(train_df, chromo)
                fit = self.calculate_fitness(t_rets)
                fits.append(fit)

            # Selection & Validation
            indices = np.argsort(fits)[-10:]
            for idx in indices:
                v_rets = self.run_backtest(val_df, population[idx])
                v_fit = self.calculate_fitness(v_rets)
                if v_fit > max_fit:
                    max_fit = v_fit
                    best_chromo = population[idx]

            # Crossover/Mutation
            new_pop = [population[i] for i in indices]
            while len(new_pop) < self.pop_size:
                parent = population[np.random.choice(indices)]
                child = parent.copy()
                child[2] += np.random.normal(0, 0.1) # Mutate threshold
                if np.random.rand() < 0.1: child[4] = np.random.choice(horizons)
                new_pop.append(child)
            population = new_pop

        return best_chromo, max_fit
