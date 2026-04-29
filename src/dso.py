import numpy as np


class DSO:
    def __init__(self, dim=30, population_size=30, max_eval=1000, lower_bound=-10, upper_bound=10, objective_function=None, seed=None):
        self.dim = dim
        self.pop_size = population_size
        self.max_eval = max_eval
        self.lb = lower_bound
        self.ub = upper_bound
        self.objective_function = objective_function

        if seed is not None:
            np.random.seed(seed)

        self.sleep = 0.9
        self.wake = 1.1
    
    def init_population(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def evaluate(self, position):
        return self.objective_function(position)
        
    def evaluate_population(self, population):
        fitness_values = []
        for agent_position in population:
            fitness_values.append(self.evaluate(agent_position))
        return np.array(fitness_values, dtype=float)
    
    def mean_position(self, population):
        return np.mean(population, axis=0)

    def sleep_phase(self, value):
        return value * np.exp(-1 / self.sleep)
    
    def wake_phase(self, value, threshold):
        return threshold + (value - threshold) * np.exp(-1 / self.wake)
    

    def homeostatic_init(self, population, X_best):
        X_mean = self.mean_position(population)
        mu = np.random.uniform(0, 1)        
        r = np.random.uniform(0, 1, self.dim)
        H_o = population + r * (X_best - mu * X_mean)
        return H_o, mu


    def homeostatic_limits(self, iteration): 
        pass
        

    def update_population(self):
        pass
    def run(self):
        pass
