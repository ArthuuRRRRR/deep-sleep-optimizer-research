import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def compute_median_q1_q3_from_histories(results):
    histories = [np.array(r["history"], dtype=float) for r in results]
    min_len = min(len(h) for h in histories)

    trimmed = np.array([h[:min_len] for h in histories], dtype=float)

    median = np.median(trimmed, axis=0)
    q1 = np.percentile(trimmed, 25, axis=0)
    q3 = np.percentile(trimmed, 75, axis=0)

    return median, q1, q3, min_len


def plot_convergence_vs_budget(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    median_1, q1_1, q3_1, min_len_1 = compute_median_q1_q3_from_histories(results_1)
    median_2, q1_2, q3_2, min_len_2 = compute_median_q1_q3_from_histories(results_2)

    min_len = min(min_len_1, min_len_2)

    x1 = np.array(results_1[0]["eval_history"][:min_len], dtype=float)
    x2 = np.array(results_2[0]["eval_history"][:min_len], dtype=float)

    plt.figure(figsize=(10, 5))

    plt.plot(x1, median_1[:min_len], label=label_1, linewidth=2)
    plt.fill_between(x1, q1_1[:min_len], q3_1[:min_len], alpha=0.2)

    plt.plot(x2, median_2[:min_len], label=label_2, linewidth=2)
    plt.fill_between(x2, q1_2[:min_len], q3_2[:min_len], alpha=0.2)

    all_vals = np.concatenate([median_1[:min_len], median_2[:min_len]])
    if np.all(all_vals > 0):
        plt.yscale("log")

    plt.xlabel("Budget utilisé (nombre d'évaluations)")
    plt.ylabel("Meilleure fitness")
    plt.title("Convergence en fonction du budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_final_positions_2d(results_1, results_2, label_1="DSO", label_2="DSO Improved"):
    positions_1 = np.array([r["best_position"] for r in results_1], dtype=float)
    positions_2 = np.array([r["best_position"] for r in results_2], dtype=float)

    if positions_1.shape[1] < 2 or positions_2.shape[1] < 2:
        print("Impossible de tracer les positions 2D : dimension < 2.")
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(positions_1[:, 0], positions_1[:, 1], label=label_1, alpha=0.7)
    plt.scatter(positions_2[:, 0], positions_2[:, 1], label=label_2, alpha=0.7)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Positions finales des meilleures solutions")
    plt.legend()
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

def plot_function_contour_with_positions(objective_function, lower_bound, upper_bound, results_1, results_2,dim=30, label_1="DSO", label_2="DSO Improved"):
    if dim < 2:
        print("Impossible de tracer un contour 2D : dimension < 2.")
        return

    x = np.linspace(lower_bound, upper_bound, 100)
    y = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    point = np.zeros(dim)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point[:] = 0.0
            point[0] = X[i, j]
            point[1] = Y[i, j]
            Z[i, j] = objective_function(point)

    positions_1 = np.array([r["best_position"] for r in results_1], dtype=float)
    positions_2 = np.array([r["best_position"] for r in results_2], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30)
    #plt.scatter(positions_1[:, 0], positions_1[:, 1], label=label_1, alpha=0.7)
    #plt.scatter(positions_2[:, 0], positions_2[:, 1], label=label_2, alpha=0.7)
    plt.scatter(positions_1[:, 0],positions_1[:, 1],label=label_1,alpha=0.7,s=80,marker="o",facecolors="none",edgecolors="blue")

    plt.scatter(positions_2[:, 0],positions_2[:, 1],label=label_2,alpha=0.7,s=50,marker="x",color="orange")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Contour de la fonction et solutions finales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_some_run_histories(results, label="DSO", max_runs=5):
    plt.figure(figsize=(10, 5))

    for i, r in enumerate(results[:max_runs]):
        x = np.array(r["eval_history"], dtype=float)
        y = np.array(r["history"], dtype=float)
        plt.plot(x, y, linewidth=1.2, alpha=0.8, label=f"{label} run {i+1}")

    all_vals = []
    for r in results[:max_runs]:
        all_vals.extend(r["history"])

    all_vals = np.array(all_vals, dtype=float)
    if np.all(all_vals > 0):
        plt.yscale("log")

    plt.xlabel("Budget utilisé (FEs)")
    plt.ylabel("Meilleure fitness")
    plt.title(f"Quelques trajectoires de convergence - {label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def print_summary(results, algo_name="DSO",):
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

def statistical_test(results_1, results_2, label_1="DSO", label_2="DSO Improved", alpha=0.05):
    values_1 = np.array([r["best_final"] for r in results_1], dtype=float)
    values_2 = np.array([r["best_final"] for r in results_2], dtype=float)

    differences = values_2 - values_1  # positif = Improved est pire, car minimisation

    wins = np.sum(values_2 < values_1)
    losses = np.sum(values_2 > values_1)
    ties = np.sum(np.isclose(values_2, values_1))

    print("\n--- Test statistique Wilcoxon apparié ---")
    print(f"Comparaison : {label_1} vs {label_2}")
    print("Hypothèse nulle H0 : pas de différence significative entre les deux versions.")
    print("Hypothèse alternative H1 : différence significative entre les deux versions.")
    print("Nombre de victoires Improved :", wins)
    print("Nombre de défaites Improved :", losses)
    print("Nombre d'égalités :", ties)
    print("Différence médiane Improved - Original :", np.median(differences))

    if np.allclose(differences, 0.0):
        print("Test Wilcoxon non applicable : toutes les différences sont nulles ou quasi nulles.")
        print("Conclusion : aucune différence observable entre les deux versions.")
        return None

    stat, p_value = wilcoxon(values_2, values_1, alternative="two-sided", zero_method="zsplit")

    print("Statistique W :", stat)
    print("p-value :", p_value)

    if p_value < alpha:
        print(f"Conclusion : différence statistiquement significative au seuil alpha={alpha}.")
        if np.median(differences) < 0:
            print(f"Interprétation : {label_2} est significativement meilleur en médiane.")
        else:
            print(f"Interprétation : {label_2} est significativement moins bon en médiane.")
    else:
        print(f"Conclusion : différence non significative au seuil alpha={alpha}.")

    return stat, p_value


def analyze_results(results_1,results_2,label_1="DSO",label_2="DSO Improved",objective_function=None,lower_bound=None,upper_bound=None,dim=30):
    print_summary(results_1, label_1)
    print_summary(results_2, label_2)

    #plot_violin(results_1, results_2, label_1, label_2)
    plot_convergence_vs_budget(results_1, results_2, label_1, label_2)
    plot_some_run_histories(results_1, label=label_1, max_runs=5)
    plot_some_run_histories(results_2, label=label_2, max_runs=5)
    plot_final_positions_2d(results_1, results_2, label_1, label_2)

    if objective_function is not None and lower_bound is not None and upper_bound is not None:
        plot_function_contour_with_positions(objective_function,lower_bound,upper_bound,results_1,results_2,dim=dim,label_1=label_1,label_2=label_2)
    statistical_test(results_1, results_2, label_1, label_2, alpha=0.05)