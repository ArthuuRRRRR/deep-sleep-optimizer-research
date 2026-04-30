import numpy as np

def compute_boundary_violation(position, lower_bound, upper_bound):
    lower_violation = np.maximum(0.0, lower_bound - position)
    upper_violation = np.maximum(0.0, position - upper_bound)
    return float(np.sum(lower_violation + upper_violation))


def fitness_function(objective_function, position, lower_bound, upper_bound, penalty_weight=10.0):
    repaired_position = np.clip(position, lower_bound, upper_bound)
    objective_value = objective_function(repaired_position)

    violation_amount = compute_boundary_violation(position, lower_bound, upper_bound)
    penalty = penalty_weight * violation_amount

    value_final = float(objective_value + penalty)

    return value_final, repaired_position