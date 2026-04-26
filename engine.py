import numpy as np
import pandas as pd
from copy import deepcopy

class GeneticEngine:
    def __init__(self, assets, benchmark, macros, cost_bps=13.5):
        self.assets = list(assets) + ['CASH']   # allow explicit cash holding
        self.benchmark = benchmark
        self.macros = macros
        self.cost_bps = cost_bps / 10000
        self.pop_size = 200                      # increased
        self.generations = 100                   # increased
        self.tournament_size = 7                 # increased
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        self.elite_size = 20                     # increased

    # ------------------------------------------------------------------
    def run_backtest(self, df, chromosome):
        """Execute a rule‑based trading strategy on historical data."""
        macro_col, operator, threshold, etf, horizon = chromosome

        if macro_col not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        if etf not in df.columns and etf != 'CASH':
            return pd.Series([0] * len(df), index=df.index)

        if operator == '>':
            condition = df[macro_col] > threshold
        else:
            condition = df[macro_col] < threshold

        raw_signal = condition.astype(int).shift(1).fillna(0)
        signal = raw_signal.rolling(window=int(horizon), min_periods=1).max()

        if etf == 'CASH':
            asset_rets = pd.Series(0.0, index=df.index)
        else:
            asset_rets = df[etf].pct_change().fillna(0)

        trades = signal.diff().abs().fillna(0)
        returns = (signal * asset_rets) - (trades * self.cost_bps)

        return returns

    # ------------------------------------------------------------------
    def sortino_ratio(self, returns, risk_free_rate=0.02):
        """Annualised Sortino ratio (downside deviation only)."""
        if len(returns) < 10:
            return -10.0
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 10:
            return -10.0

        excess = returns - (risk_free_rate / 252)
        downside = excess[excess < 0].std()
        if downside == 0 or np.isnan(downside):
            return 0.0
        annual_excess = excess.mean() * 252
        annual_downside = downside * np.sqrt(252)
        return annual_excess / annual_downside

    # ------------------------------------------------------------------
    def walk_forward_fitness(self, chromosome, data, n_folds=5,
                             risk_free_rate=0.02):
        """Compute average Sortino over multiple out‑of‑sample periods."""
        n = len(data)
        fold_len = n // (n_folds + 1)
        sortinos = []

        for i in range(n_folds):
            train_end = (i + 1) * fold_len
            val_end   = train_end + fold_len
            if val_end > n:
                break
            val_df = data.iloc[train_end:val_end]
            rets = self.run_backtest(val_df, chromosome)
            sr = self.sortino_ratio(rets, risk_free_rate)
            sortinos.append(sr)

        if not sortinos:
            return -10.0
        return np.mean(sortinos)

    # ------------------------------------------------------------------
    def create_random_chromosome(self):
        """Generate a random rule."""
        macro = np.random.choice(self.macros)
        operator = np.random.choice(['>', '<'])

        if macro in ['VIX', 'DXY', 'TBILL_3M']:
            threshold = np.random.uniform(0, 100)
        elif macro in ['T10Y2Y', 'IG_SPREAD', 'HY_SPREAD']:
            threshold = np.random.uniform(-5, 5)
        else:
            threshold = np.random.uniform(-2, 2)

        etf = np.random.choice(self.assets)   # includes CASH
        horizon = np.random.choice([1, 3, 5])

        return [macro, operator, threshold, etf, horizon]

    # ------------------------------------------------------------------
    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    # ------------------------------------------------------------------
    def mutate(self, chromosome, generation):
        mutated = deepcopy(chromosome)
        current_rate = self.mutation_rate * (1 - generation / self.generations)

        for i in range(len(mutated)):
            if np.random.rand() < current_rate:
                if i == 0:
                    mutated[i] = np.random.choice(self.macros)
                elif i == 1:
                    mutated[i] = np.random.choice(['>', '<'])
                elif i == 2:
                    if mutated[0] in ['VIX', 'DXY', 'TBILL_3M']:
                        mutated[i] = np.random.uniform(0, 100)
                    elif mutated[0] in ['T10Y2Y', 'IG_SPREAD', 'HY_SPREAD']:
                        mutated[i] = np.random.uniform(-5, 5)
                    else:
                        mutated[i] = np.random.uniform(-2, 2)
                elif i == 3:
                    mutated[i] = np.random.choice(self.assets)
                elif i == 4:
                    mutated[i] = np.random.choice([1, 3, 5])
        return mutated

    # ------------------------------------------------------------------
    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(2):
            indices = np.random.choice(len(population), self.tournament_size, replace=False)
            best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
            selected.append(population[best_idx])
        return selected[0], selected[1]

    # ------------------------------------------------------------------
    def evolve(self, data):
        """Main GA loop with walk‑forward Sortino fitness."""
        population = [self.create_random_chromosome() for _ in range(self.pop_size)]
        best_chromosome = None
        best_fitness = -np.inf
        patience = 0

        for gen in range(self.generations):
            fitnesses = [self.walk_forward_fitness(c, data) for c in population]

            current_best_idx = np.argmax(fitnesses)
            current_best_fit = fitnesses[current_best_idx]

            if current_best_fit > best_fitness:
                best_fitness = current_best_fit
                best_chromosome = deepcopy(population[current_best_idx])
                patience = 0
            else:
                patience += 1

            if patience > 15:   # early stopping after 15 generations without improvement
                break

            # Elitism
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [deepcopy(population[i]) for i in elite_indices]

            # Breed new individuals
            while len(new_population) < self.pop_size:
                p1, p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1, gen)
                c2 = self.mutate(c2, gen)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population

        if best_chromosome is None:
            return None, -10.0

        # Out‑of‑sample test on the last fold (hold‑out)
        n = len(data)
        test_start = int(n * 0.85)
        test_df = data.iloc[test_start:]
        test_rets = self.run_backtest(test_df, best_chromosome)
        test_fitness = self.sortino_ratio(test_rets)

        return best_chromosome, test_fitness
