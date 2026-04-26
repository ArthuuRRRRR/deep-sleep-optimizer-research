import numpy as np


class DSO:
    def __init__(self, dim=30, population_size=30, max_eval=1000, lower_bound=-10, upper_bound=10):
        self.dim = dim
        self.pop_size = population_size
        self.max_eval = max_eval
        self.lb = lower_bound
        self.ub = upper_bound

        self.sleep = 0.9
        self.wake = 1.1
    
    def init_population(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def evaluate():
        pass

    def mean_position(self, population):
        return np.mean(population, axis=0)

    def sleep_phase(self, value):
        return value * np.exp(-1 / self.sleep_power)
    
    def wake_phase(self, value, threshold):
        return threshold + (value - threshold) * np.exp(-1 / self.wake_power)
    
    def update_population():
        pass
    def run():
        pass
