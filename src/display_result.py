import numpy as np
import matplotlib.pyplot as plt


def compute_median_q1_q3(results):
    histories = [r["history"] for r in results]
    min_len = min(len(h) for h in histories)

    histories = np.array([h[:min_len] for h in histories], dtype=float)

    median = np.median(histories, axis=0)
    q1 = np.percentile(histories, 25, axis=0)
    q3 = np.percentile(histories, 75, axis=0)

    return median, q1, q3


def plot_convergence(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    median_1, q1_1, q3_1 = compute_median_q1_q3(results_1)
    median_2, q1_2, q3_2 = compute_median_q1_q3(results_2)

    min_len = min(len(median_1), len(median_2))
    generations = range(min_len)

    median_1, q1_1, q3_1 = median_1[:min_len], q1_1[:min_len], q3_1[:min_len]
    median_2, q1_2, q3_2 = median_2[:min_len], q1_2[:min_len], q3_2[:min_len]

    plt.figure(figsize=(12, 6))

    plt.plot(generations, median_1, label=label_1, linewidth=2)
    plt.fill_between(generations, q1_1, q3_1, alpha=0.2)

    plt.plot(generations, median_2, label=label_2, linewidth=2)
    plt.fill_between(generations, q1_2, q3_2, alpha=0.2)

    plt.yscale("log")
    plt.xlabel("Itérations")
    plt.ylabel("Meilleure fitness")
    plt.title("Courbe de convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_boxplot(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    values_1 = [r["best_final"] for r in results_1]
    values_2 = [r["best_final"] for r in results_2]

    plt.figure(figsize=(7, 5))
    plt.boxplot([values_1, values_2], labels=[label_1, label_2])

    plt.ylabel("Fitness finale")
    plt.title("Comparaison des performances finales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_violin(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    values_1 = [r["best_final"] for r in results_1]
    values_2 = [r["best_final"] for r in results_2]

    plt.figure(figsize=(7, 5))

    parts = plt.violinplot(
        [values_1, values_2],
        showmeans=True
    )

    colors = ["lightblue", "lightgreen"]

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    plt.xticks([1, 2], [label_1, label_2])
    plt.ylabel("Fitness finale")
    plt.title("Distribution des performances finales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sorted_final_scores(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    values_1 = sorted([r["best_final"] for r in results_1])
    values_2 = sorted([r["best_final"] for r in results_2])

    plt.figure(figsize=(9, 5))
    plt.plot(values_1, marker="o", linewidth=1.5, label=label_1)
    plt.plot(values_2, marker="o", linewidth=1.5, label=label_2)

    plt.xlabel("Run trié")
    plt.ylabel("Fitness finale")
    plt.title("Scores finaux triés")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(results, algo_name="DSO"):
    values = np.array([r["best_final"] for r in results], dtype=float)

    print(f"\n--- Résumé {algo_name} ---")
    print("Nombre de runs :", len(values))
    print("Moyenne :", np.mean(values))
    print("Écart-type :", np.std(values))
    print("Minimum :", np.min(values))
    print("Q1 :", np.percentile(values, 25))
    print("Médiane :", np.median(values))
    print("Q3 :", np.percentile(values, 75))
    print("Maximum :", np.max(values))


def analyze_results(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    print_summary(results_1, label_1)
    print_summary(results_2, label_2)

    plot_convergence(results_1, results_2, label_1, label_2)
    plot_boxplot(results_1, results_2, label_1, label_2)
    plot_violin(results_1, results_2, label_1, label_2)
    plot_sorted_final_scores(results_1, results_2, label_1, label_2)