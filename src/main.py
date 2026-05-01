
import numpy as np
from dso import DSO
from benchmarks import function1_sphere, function2_schwefel_222, function3_rosenbrock, function4_rastrigin, function5_ackley, function6_griewank
from benchmarks import Parameters_Benchmarks
from fitness_function import fitness_function
from improved_dso import DSO_Improved
from monte_carlo import monte_carlo_DSO, monte_carlo_DSO_improved, save_results
from display_result import analyze_results

benchmark = Parameters_Benchmarks["function1_sphere"]

objective_function = benchmark["function"]
lb = benchmark["lower_bound"]
ub = benchmark["upper_bound"]

#dso = DSO(dim=30,population_size=30,max_eval=1000,lower_bound=lb,upper_bound=ub,objective_function=objective_function,seed=42)

#dso_improved = DSO_Improved(dim=30,population_size=30,max_eval=1000,lower_bound=lb,upper_bound=ub,objective_function=objective_function,seed=42, penalty_weight=10.0)

monte_carlo_DSO_results = monte_carlo_DSO(dim=30, population_size=30, max_eval=1000, lower_bound=lb, upper_bound=ub, objective_function=objective_function, n_runs=20, seed_depart=42)
monte_carlo_DSO_improved_results = monte_carlo_DSO_improved(dim=30, population_size=30, max_eval=1000, lower_bound=lb, upper_bound=ub, objective_function=objective_function, n_runs=20, seed_depart=42, penalty_weight=10.0)

analyze_results(monte_carlo_DSO_results, monte_carlo_DSO_improved_results, label_1="DSO", label_2="DSO Improved")


#save_results(monte_carlo_DSO_results, "dso_results.csv")
#save_results(monte_carlo_DSO_improved_results, "dso_improved_results.csv")