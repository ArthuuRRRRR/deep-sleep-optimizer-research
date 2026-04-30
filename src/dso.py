import numpy as np


class DSO:
    def __init__(self, dim=30, population_size=30, max_eval=1000, lower_bound=-10, upper_bound=10, objective_function=None, seed=None):
        self.dim = dim
        self.pop_size = population_size
        self.max_eval = max_eval
        self.lb = lower_bound
        self.ub = upper_bound
        self.objective_function = objective_function

        self.random_gen = np.random.default_rng(seed)

        self.sleep = 0.9
        self.wake = 1.1
        self.T = 24
        self.a = 0.5
        self.H_plus = 1.0
        self.H_minus = 0.0

    def init_population(self):
        return self.random_gen.uniform(self.lb, self.ub, (self.pop_size, self.dim))

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
    
    def homeostatic_init(self, population, agent, X_best):
        X_mean = self.mean_position(population)
        mu = self.random_gen.uniform(0, 1)
        r = self.random_gen.uniform(0, 1, self.dim)

        H_0 = agent + r * (X_best - mu * X_mean)

        return H_0, mu

    def homeostatic_limits(self, iteration):
        alpha = self.random_gen.uniform(0, 1)
        C_t = np.sin((2 * np.pi / self.T) * (iteration - alpha))

        Hmstc_max = self.H_plus + self.a * C_t
        Hmstc_min = self.H_minus + self.a * C_t

        return Hmstc_min, Hmstc_max

    def generate_threshold(self, iteration):
        H_min, H_max = self.homeostatic_limits(iteration)
        return self.random_gen.uniform(H_min, H_max)

    def apply_sleep_or_wake(self, H_0, mu, threshold):
        if mu < 0.5:
            return self.sleep_phase(H_0)
        else:
            return self.wake_phase(H_0, threshold)

    def evaluate_candidate(self, position):
        repaired_position = np.clip(position, self.lb, self.ub)
        fitness_value = self.evaluate(repaired_position)
        return fitness_value, repaired_position

    def update_best_solution(self):
        best_index = int(np.argmin(self.fitness_values))
        self.best_position = self.population[best_index].copy()
        self.best_fitness = float(self.fitness_values[best_index])

    def update_population(self, iteration):
        for agent_index in range(self.pop_size):
            if self.evaluations_used >= self.max_eval:
                break

            agent = self.population[agent_index]

            H_0, mu = self.homeostatic_init(self.population,agent,self.best_position)

            threshold = self.generate_threshold(iteration)
            candidate_position = self.apply_sleep_or_wake(H_0,mu,threshold)
            candidate_fitness, repaired_candidate = self.evaluate_candidate(candidate_position)

            self.evaluations_used += 1

            if candidate_fitness <= self.fitness_values[agent_index]:
                self.population[agent_index] = repaired_candidate
                self.fitness_values[agent_index] = candidate_fitness

                if candidate_fitness < self.best_fitness:
                    self.best_position = repaired_candidate.copy()
                    self.best_fitness = float(candidate_fitness)

    def run(self):
        self.population = self.init_population()
        self.fitness_values = self.evaluate_population(self.population)
        self.evaluations_used = self.pop_size

        self.update_best_solution()

        self.fitness_history = [self.best_fitness]
        self.evaluation_history = [self.evaluations_used]

        iteration = 0

        while self.evaluations_used < self.max_eval:
            iteration += 1
            self.update_population(iteration)

            self.fitness_history.append(self.best_fitness)
            self.evaluation_history.append(self.evaluations_used)

        return (self.best_position,self.best_fitness,self.fitness_history,self.evaluation_history,self.evaluations_used)