import numpy as np
import pandas as pd
from copy import deepcopy

class GeneticEngine:
    def __init__(self, assets, benchmark, macros, cost_bps=13.5):
        self.assets = assets
        self.benchmark = benchmark
        self.macros = macros
        self.cost_bps = cost_bps / 10000
        self.pop_size = 100
        self.generations = 50
        self.tournament_size = 5
        self.crossover_rate = 0.7
        self.mutation_rate = 0.2
        self.elite_size = 10

    def calculate_fitness(self, returns, risk_free_rate=0.02):
        """Calculate fitness with proper metrics"""
        if len(returns) < 10:
            return -999.0
        
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 10:
            return -999.0
        
        total_ret = (1 + returns).prod() - 1
        
        # Penalize negative returns heavily instead of hard rejection
        if total_ret <= 0:
            return -100.0 * abs(total_ret)
        
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
        
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.expanding().max()
        drawdown = (cum_ret - peak) / peak
        max_dd = abs(drawdown.min()) if not pd.isna(drawdown.min()) else 1.0
        
        calmar = total_ret / (max_dd + 1e-9)
        
        # Composite fitness: 50% return, 25% Sharpe, 25% Calmar (capped)
        fitness = (0.50 * total_ret * 100) + (0.25 * sharpe) + (0.25 * min(calmar, 5.0))
        
        return float(fitness)

    def run_backtest(self, df, chromosome):
        """Run backtest for a single chromosome"""
        macro_col, operator, threshold, etf, horizon = chromosome
        
        if macro_col not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        
        if operator == '>':
            condition = df[macro_col] > threshold
        else:
            condition = df[macro_col] < threshold
        
        raw_signal = condition.astype(int).shift(1).fillna(0)
        signal = raw_signal.rolling(window=int(horizon), min_periods=1).max()
        
        if etf not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        
        asset_rets = df[etf].pct_change().fillna(0)
        trades = signal.diff().abs().fillna(0)
        returns = (signal * asset_rets) - (trades * self.cost_bps)
        
        return returns

    def create_random_chromosome(self):
        """Create a random chromosome with valid bounds"""
        macro = np.random.choice(self.macros)
        operator = np.random.choice(['>', '<'])
        
        if macro in ['VIX', 'DXY', 'TBILL_3M']:
            threshold = np.random.uniform(0, 100)
        elif macro in ['T10Y2Y', 'IG_SPREAD', 'HY_SPREAD']:
            threshold = np.random.uniform(-5, 5)
        else:
            threshold = np.random.uniform(-2, 2)
        
        etf = np.random.choice(self.assets)
        horizon = np.random.choice([1, 3, 5])
        
        return [macro, operator, threshold, etf, horizon]

    def crossover(self, parent1, parent2):
        """Single-point crossover between two parents"""
        if np.random.rand() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2

    def mutate(self, chromosome, generation):
        """Mutate a chromosome with adaptive mutation rate"""
        mutated = deepcopy(chromosome)
        current_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
        
        for i in range(len(mutated)):
            if np.random.rand() < current_mutation_rate:
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

    def tournament_selection(self, population, fitnesses):
        """Tournament selection for parent choice"""
        selected = []
        for _ in range(2):
            tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
            best_idx = tournament_indices[np.argmax([fitnesses[i] for i in tournament_indices])]
            selected.append(population[best_idx])
        return selected[0], selected[1]

    def evolve(self, data):
        """Main evolution loop with proper GA operators"""
        n = len(data)
        train_df = data.iloc[:int(n * 0.7)]
        val_df = data.iloc[int(n * 0.7):int(n * 0.85)]
        test_df = data.iloc[int(n * 0.85):]
        
        population = [self.create_random_chromosome() for _ in range(self.pop_size)]
        best_chromosome = None
        best_validation_fitness = -np.inf
        patience_counter = 0
        
        for generation in range(self.generations):
            train_fitnesses = []
            for chromo in population:
                returns = self.run_backtest(train_df, chromo)
                fitness = self.calculate_fitness(returns)
                train_fitnesses.append(fitness)
            
            validation_fitnesses = []
            for chromo in population:
                returns = self.run_backtest(val_df, chromo)
                fitness = self.calculate_fitness(returns)
                validation_fitnesses.append(fitness)
            
            current_best_idx = np.argmax(validation_fitnesses)
            current_best_fitness = validation_fitnesses[current_best_idx]
            
            if current_best_fitness > best_validation_fitness:
                best_validation_fitness = current_best_fitness
                best_chromosome = deepcopy(population[current_best_idx])
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > 10:
                break
            
            elite_indices = np.argsort(validation_fitnesses)[-self.elite_size:]
            new_population = [deepcopy(population[i]) for i in elite_indices]
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.tournament_selection(population, validation_fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, generation)
                child2 = self.mutate(child2, generation)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            population = new_population
        
        if best_chromosome:
            test_returns = self.run_backtest(test_df, best_chromosome)
            test_fitness = self.calculate_fitness(test_returns)
            return best_chromosome, test_fitness
        
        return None, -999.0
