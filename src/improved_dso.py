import numpy as np
from dso import DSO
from fitness_function import fitness_function


class DSO_Improved(DSO):
    def __init__(self, *args, penalty_weight=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_weight = penalty_weight

    def evaluate_candidate(self, position):
        fitness_value, repaired_position = fitness_function(self.objective_function,position,self.lb,self.ub,self.penalty_weight)
        return fitness_value, repaired_position

    def apply_sleep_or_wake(self, H_0, mu, iteration):
        if mu < 0.5:
            candidate = self.sleep_phase(H_0, iteration)
        else:
            candidate = self.wake_phase(H_0, mu, iteration)

        progress = min(1.0, self.evaluations_used / self.max_eval)

        if progress < 0.6:
            sigma = 0.001 * (self.ub - self.lb) * (1.0 - progress)
            noise = self.random_gen.normal(0.0, sigma, self.dim)
            candidate = candidate + mu * noise

        #return np.clip(candidate, self.lb, self.ub)
        return candidate