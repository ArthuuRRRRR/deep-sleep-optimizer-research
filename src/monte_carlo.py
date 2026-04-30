from dso import DSO
import numpy as np
import pandas as pd


def simple_DSO(dim, population_size, max_eval, lower_bound, upper_bound,objective_function, seed):
    
    dso = DSO(dim=dim,population_size=population_size,max_eval=max_eval,lower_bound=lower_bound,upper_bound=upper_bound,objective_function=objective_function,seed=seed)

    best_position, best_fitness, history, eval_history, evals = dso.run()

    return best_position, best_fitness, history

def monte_carlo_DSO(dim=30, population_size=30, max_eval=1000,lower_bound=None, upper_bound=None,objective_function=None, n_runs=20, seed_depart=42):

    resultats = []

    for run in range(n_runs):
        seed = seed_depart + run

        best_position, best_fitness, history = simple_DSO(dim,population_size,max_eval,lower_bound,upper_bound,objective_function,seed)

        resultats.append({"run_id": run + 1,"seed": seed,"best_final": best_fitness,"history": history})

        print(f"Run {run + 1} | best={best_fitness}")

    return resultats