from __future__ import annotations

import numpy as np
from dso import DSO
from benchmarks import function1_sphere, function2_schwefel_222, function3_rosenbrock, function4_rastrigin, function5_ackley, function6_griewank
from benchmarks import Parameters_Benchmarks
from fitness_function import fitness_function
from improved_dso import DSO_Improved
from monte_carlo import monte_carlo_DSO, monte_carlo_DSO_improved, save_results
from display_result import analyze_results, print_summary, statistical_test
import argparse
from pathlib import Path


DEFAULT_BENCHMARK = "function2_schwefel_222"


def build_parser() -> argparse.ArgumentParser:
    benchmark_choices = list(Parameters_Benchmarks.keys()) + ["all"]

    parser = argparse.ArgumentParser(description="Compare DSO et DSO Improved sur une ou plusieurs fonctions benchmark.")

    parser.add_argument(
        "--benchmark",
        choices=benchmark_choices,
        default=DEFAULT_BENCHMARK,
        help=("Benchmark à exécuter. Utiliser 'all' pour lancer toutes les fonctions. "f"Valeur par défaut : {DEFAULT_BENCHMARK}."),)
    parser.add_argument("--dim", type=int, default=30, help="Dimension du problème.")
    parser.add_argument("--population-size", type=int, default=30, help="Taille de la population.")
    parser.add_argument("--max-eval", type=int, default=1000, help="Budget maximal d'évaluations.")
    parser.add_argument("--n-runs", type=int, default=20, help="Nombre de runs Monte-Carlo.")
    parser.add_argument("--seed", type=int, default=42, help="Seed de départ.")
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=10.0,
        help="Poids de pénalité utilisé par DSO Improved.",)
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Sauvegarde les résultats DSO et DSO Improved en CSV.",)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Dossier où enregistrer les CSV si --save-csv est activé.",)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Désactive les figures matplotlib et affiche seulement les statistiques.",)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dim < 1:
        raise ValueError("La dimension doit être >= 1.")
    if args.population_size < 1:
        raise ValueError("La taille de population doit être >= 1.")
    if args.max_eval < args.population_size:
        raise ValueError("max_eval doit être supérieur ou égal à population_size.")
    if args.n_runs < 1:
        raise ValueError("Le nombre de runs doit être >= 1.")
    if args.penalty_weight < 0:
        raise ValueError("penalty_weight doit être positif ou nul.")


def selected_benchmarks(benchmark_arg: str) -> list[str]:
    if benchmark_arg == "all":
        return list(Parameters_Benchmarks.keys())
    return [benchmark_arg]


def run_one_benchmark(benchmark_key: str, args: argparse.Namespace) -> dict[str, Any]:
    benchmark = Parameters_Benchmarks[benchmark_key]
    objective_function = benchmark["function"]
    lower_bound = benchmark["lower_bound"]
    upper_bound = benchmark["upper_bound"]
    benchmark_name = benchmark["name"]

    print("\n" + "=" * 80)
    print(f"Benchmark : {benchmark_name} ({benchmark_key})")
    print(f"Bornes    : [{lower_bound}, {upper_bound}]")
    print(f"Optimum connu : {benchmark.get('known_optimum', 'non renseigné')}")
    print(
        "Paramètres : "
        f"dim={args.dim}, population_size={args.population_size}, "
        f"max_eval={args.max_eval}, n_runs={args.n_runs}, seed={args.seed}"
    )

    dso_results = monte_carlo_DSO(
        dim=args.dim,
        population_size=args.population_size,
        max_eval=args.max_eval,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        objective_function=objective_function,
        n_runs=args.n_runs,
        seed_depart=args.seed,
    )

    dso_improved_results = monte_carlo_DSO_improved(
        dim=args.dim,
        population_size=args.population_size,
        max_eval=args.max_eval,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        objective_function=objective_function,
        n_runs=args.n_runs,
        seed_depart=args.seed,
        penalty_weight=args.penalty_weight,
    )

    if args.no_plots:
        print_summary(dso_results, "DSO")
        print_summary(dso_improved_results, "DSO Improved")
        statistical_test(dso_results, dso_improved_results, "DSO", "DSO Improved")
    else:
        analyze_results(
            dso_results,
            dso_improved_results,
            label_1="DSO",
            label_2="DSO Improved",
            objective_function=objective_function,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=args.dim,
        )

    if args.save_csv:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dso_filename = output_dir / f"{benchmark_key}_dso_results.csv"
        dso_improved_filename = output_dir / f"{benchmark_key}_dso_improved_results.csv"

        save_results(dso_results, str(dso_filename))
        save_results(dso_improved_results, str(dso_improved_filename))

        print("\nRésultats sauvegardés :")
        print(f"- {dso_filename}")
        print(f"- {dso_improved_filename}")

    return {
        "benchmark_key": benchmark_key,
        "benchmark_name": benchmark_name,
        "dso_results": dso_results,
        "dso_improved_results": dso_improved_results,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    for benchmark_key in selected_benchmarks(args.benchmark):
        run_one_benchmark(benchmark_key, args)


if __name__ == "__main__":
    main()
