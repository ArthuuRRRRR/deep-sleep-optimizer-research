import numpy as np

################### Unimodal functions
def function1_sphere(position):
    return float(np.sum(position**2))

def function2_schwefel_222(position):
    absolute_position = np.abs(position)
    sum_part = np.sum(absolute_position)
    product_part = np.prod(absolute_position)

    return float(sum_part + product_part)

def function3_rosenbrock(position):
    total = 0.0
    for index in range(len(position) - 1):
        current_value = position[index]
        next_value = position[index + 1]
        total += 100.0 * (next_value - current_value**2) ** 2 + (current_value - 1.0) ** 2

    return float(total)

#################### Multimodal functions
def function4_rastrigin(position):
    dimension = len(position)
    return float(10.0 * dimension + np.sum(position**2 - 10.0 * np.cos(2.0 * np.pi * position)))

def function5_ackley(position):
    dimension = len(position)
    square_average = np.sum(position**2) / dimension
    cosine_average = np.sum(np.cos(2.0 * np.pi * position)) / dimension
    first_part = -20.0 * np.exp(-0.2 * np.sqrt(square_average))
    second_part = -np.exp(cosine_average)

    return float(first_part + second_part + 20.0 + np.e)

def function6_griewank(position):
    sum_part = np.sum(position**2) / 4000.0
    product_part = 1.0
    for index, value in enumerate(position):
        product_part *= np.cos(value / np.sqrt(index + 1))

    return float(1.0 + sum_part - product_part)

Parameters_Benchmarks = {
    "function1_sphere": {
        "name": "Sphere",
        "lower_bound": -100.0,
        "upper_bound": 100.0,
        "known_optimum": 0.0,
        "function": function1_sphere
    },
    "function2_schwefel_222": {
        "name": "Schwefel 222",
        "lower_bound": -10.0,
        "upper_bound": 10.0,
        "known_optimum": 0.0,
        "function": function2_schwefel_222
    },
    "function3_rosenbrock": {
        "name": "Rosenbrock",
        "lower_bound": -30,
        "upper_bound": 30,
        "known_optimum": 0.0,
        "function": function3_rosenbrock
    },
    "function4_rastrigin": {
        "name": "Rastrigin",
        "lower_bound": -5.12,
        "upper_bound": 5.12,
        "known_optimum": 0.0,
        "function": function4_rastrigin
    },
    "function5_ackley": {
        "name": "Ackley",
        "lower_bound": -32.0,
        "upper_bound": 32.0,
        "known_optimum": 0.0,
        "function": function5_ackley
    },
    "function6_griewank": {
        "name": "Griewank",
        "lower_bound": -600.0,
        "upper_bound": 600.0,
        "known_optimum": 0.0,
        "function": function6_griewank
    }
}