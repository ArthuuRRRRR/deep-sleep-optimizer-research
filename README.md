# Projet Meta-Optimisation - DSO et DSO Improved

Ce projet reconstitue  l'algorithme **DSO** (provenent de “The Deep Sleep Optimizer :A Human-Based Metaheuristic Approach” 
)avec une version améliorée, appelée **DSO Improved**, sur plusieurs fonctions benchmark classiques d'optimisation continue.

Le programme permet d'exécuter des expériences Monte-Carlo, d'afficher des statistiques, de tracer les courbes de convergence et, si souhaité, de sauvegarder les résultats au format CSV.

---

## 1. Objectif du projet

L'objectif est d'évaluer les performances de deux versions de l'algorithme DSO :

- **DSO original** : version de base de l'algorithme.
- **DSO Improved** : version améliorée avec une gestion différente du paramètre `mu`, une pénalisation des violations de bornes et l'ajout d'un bruit contrôlé en début d'optimisation.

Les deux algorithmes sont comparés sur des fonctions de test unimodales et multimodales.

---

## 2. Structure du projet

```text
.
├── src/
│   ├── main.py
│   ├── benchmarks.py
│   ├── dso.py
│   ├── improved_dso.py
│   ├── fitness_function.py
│   ├── monte_carlo.py
│   └── display_result.py
└── README.md
```

### Description des fichiers

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée principal du programme. Permet de choisir le benchmark, les paramètres et le mode d'exécution. |
| `benchmarks.py` | Contient les fonctions benchmark : Sphere, Schwefel 2.22, Rosenbrock, Rastrigin, Ackley et Griewank. |
| `dso.py` | Implémentation de l'algorithme DSO original. |
| `improved_dso.py` | Implémentation de DSO Improved, héritant de la classe DSO. |
| `fitness_function.py` | Fonction de fitness avec pénalité en cas de dépassement des bornes. |
| `monte_carlo.py` | Fonctions permettant de lancer plusieurs runs Monte-Carlo et de sauvegarder les résultats. |
| `display_result.py` | Fonctions d'affichage, de résumé statistique, de visualisation et de test de Wilcoxon. |

---

## 3. Installation

### Prérequis

Le projet nécessite Python 3 et les bibliothèques suivantes :

```bash
pip install numpy pandas matplotlib scipy
```


---

## 4. Exécution du projet

Le fichier `main.py` se trouve dans le dossier `src`.

Depuis la racine du projet, exécuter :

```bash
python src/main.py
```

Ou entrer d'abord dans le dossier `src` :

```bash
cd src
python main.py
```

Par défaut, le programme exécute le benchmark :

```text
function1_sphere
```

---

## 5. Exemples d'utilisation

### Exécuter le benchmark par défaut

```bash
python src/main.py
```

### Exécuter Rastrigin en dimension 30

```bash
python src/main.py --benchmark function4_rastrigin --dim 30 --max-eval 1000 --n-runs 20
```

### Exécuter tous les benchmarks

```bash
python src/main.py --benchmark all
```

### Exécuter sans afficher les graphiques

Utile pour tester rapidement le programme ou éviter l'ouverture de fenêtres Matplotlib.

```bash
python src/main.py --benchmark function4_rastrigin --no-plots
```

### Sauvegarder les résultats en CSV

```bash
python src/main.py --benchmark function4_rastrigin --save-csv
```

Les fichiers seront enregistrés par défaut dans le dossier :

```text
results/
```

### Choisir un dossier de sortie personnalisé

```bash
python src/main.py --benchmark function4_rastrigin --save-csv --output-dir mes_resultats
```

---

## 6. Arguments disponibles

| Argument | Description | Valeur par défaut |
|---|---|---|
| `--benchmark` | Benchmark à exécuter. Utiliser `all` pour lancer tous les benchmarks. | `function2_schwefel_222` |
| `--dim` | Dimension du problème. | `30` |
| `--population-size` | Taille de la population. | `30` |
| `--max-eval` | Budget maximal d'évaluations de la fonction objectif. | `1000` |
| `--n-runs` | Nombre de runs Monte-Carlo. | `20` |
| `--seed` | Seed de départ utilisée pour les runs. | `42` |
| `--penalty-weight` | Poids de pénalité utilisé par DSO Improved. | `10.0` |
| `--save-csv` | Sauvegarde les résultats en fichiers CSV. | Désactivé |
| `--output-dir` | Dossier de sauvegarde des CSV. | `results` |
| `--no-plots` | Désactive les graphiques et affiche seulement les statistiques. | Désactivé |

---

## 7. Benchmarks disponibles

Les fonctions disponibles sont :

| Clé à utiliser | Nom de la fonction | Type |
|---|---|---|
| `function1_sphere` | Sphere | Unimodale |
| `function2_schwefel_222` | Schwefel 2.22 | Unimodale |
| `function3_rosenbrock` | Rosenbrock | Unimodale |
| `function4_rastrigin` | Rastrigin | Multimodale |
| `function5_ackley` | Ackley | Multimodale |
| `function6_griewank` | Griewank | Multimodale |

Exemple :

```bash
python src/main.py --benchmark function5_ackley
```

---

## 8. Résultats affichés

Pour chaque benchmark, le programme affiche :

- le nom de la fonction benchmark ;
- les bornes inférieure et supérieure ;
- l'optimum connu ;
- les paramètres utilisés ;
- un résumé statistique des performances de DSO ;
- un résumé statistique des performances de DSO Improved ;
- un test statistique de Wilcoxon apparié ;
- des graphiques de convergence et de distribution, sauf si `--no-plots` est utilisé.

Les indicateurs statistiques affichés sont notamment :

- moyenne ;
- écart-type ;
- minimum ;
- quartiles Q1 et Q3 ;
- médiane ;
- maximum.

---

## 9. Sauvegarde des résultats

Lorsque l'option `--save-csv` est activée, deux fichiers sont générés pour chaque benchmark :

```text
<benchmark>_dso_results.csv
<benchmark>_dso_improved_results.csv
```

Exemple avec Rastrigin :

```text
function4_rastrigin_dso_results.csv
function4_rastrigin_dso_improved_results.csv
```

Chaque ligne correspond à un run Monte-Carlo et contient notamment :

- l'identifiant du run ;
- la seed utilisée ;
- la meilleure fitness finale ;
- la meilleure position trouvée ;
- l'historique de convergence ;
- le budget d'évaluations utilisé.

---

## 10. Exemple complet recommandé

Pour lancer une comparaison propre entre DSO et DSO Improved sur Rastrigin :

```bash
python src/main.py --benchmark function4_rastrigin --dim 30 --population-size 30 --max-eval 1000 --n-runs 20 --seed 42 --save-csv
```

Pour exécuter tous les benchmarks sans graphiques et sauvegarder les résultats :

```bash
python src/main.py --benchmark all --no-plots --save-csv
```

---

## 11. Notes importantes

- Le projet est conçu pour un problème de **minimisation**.
- L'optimum connu des fonctions benchmark utilisées est généralement `0.0`.
- Une valeur de fitness plus faible indique une meilleure solution.
- Le test de Wilcoxon permet de vérifier si la différence entre DSO et DSO Improved est statistiquement significative.

---


## 13. Auteur

Arthur Delhaye, Projet réalisé dans le cadre universitaire à l'UQAR

