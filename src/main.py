
import numpy as np
from dso import DSO
from benchmarks import function1_sphere, function2_schwefel_222, function3_rosenbrock, function4_rastrigin, function5_ackley, function6_griewank
from benchmarks import Parameters_Benchmarks
from fitness_function import fitness_function

benchmark = Parameters_Benchmarks["function1_sphere"]

objective_function = benchmark["function"]
lb = benchmark["lower_bound"]
ub = benchmark["upper_bound"]

dso = DSO(dim=30,population_size=30,max_eval=1000,lower_bound=lb,upper_bound=ub,objective_function=objective_function,seed=42)

best_position, best_fitness, history, eval_history, evals = dso.run()

print("Best Position:", best_position)
print("Best Fitness:", best_fitness)