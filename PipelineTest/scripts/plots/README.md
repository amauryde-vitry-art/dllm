# scripts/plots — Modular Plotting Library

Bibliothèque de visualisations modulaires pour l'analyse de détection d'hallucinations dans les modèles de diffusion discrets.

## Utilisation

```bash
# Tous les modes
python -m PipelineTest.scripts.run_analysis --config llada --mode all

# Un seul mode
python -m PipelineTest.scripts.run_analysis --config llada --mode heatmaps

# Plusieurs modes
python -m PipelineTest.scripts.run_analysis --config dream --mode ar1_per_sample,ar1_per_trajectory

# Standalone
python -m PipelineTest.scripts.plots.heatmaps --config llada
```

### Options

| Argument | Description | Défaut |
|----------|-------------|--------|
| `--config` | Configuration (`llada` ou `dream`) | `llada` |
| `--mode` | Mode(s) séparés par virgule, ou `all` | `all` |
| `--n-samples` | Nombre d'échantillons pour les modes per-sample | `20` |
| `--no-padding` | Exclure les tokens de padding | `True` |

---

## Modes disponibles

| Mode | Fichier | Description |
|------|---------|-------------|
| `heatmaps` | `heatmaps.py` | Heatmaps des paramètres AR(1) (φ, intercept, σ) + matrice de corrélation des features |
| `ar1_per_sample` | `ar1_per_sample.py` | Mean/Var masked entropy par sample : données vs reconstruction AR(1) correct/halluc |
| `ar1_per_trajectory` | `ar1_per_trajectory.py` | Trajectoires de 10 tokens (espacés uniformément dans l'ordre d'unmasking), données vs AR(1) |
| `benchmark_global` | `benchmark_global.py` | Courbes population (mean/var trajectories), scatter plots, top features par AUC |
| `benchmark_per_sample` | `benchmark_per_sample.py` | Histogrammes des features (correct vs halluc), évolution per-step par sample |
| `plotly_samples` | `plotly_samples.py` | Visualisations interactives Plotly (entropy heatmap + tokens proposés au hover) |

---

## Organisation des dossiers de sauvegarde

Les plots sont sauvegardés **à côté des fichiers de résultats**, dans un sous-dossier `plots/<mode>/` :

```
PipelineTest/res/
├── LLADA_64steps_64tokens_lowconf/
│   ├── outputs_LLADA_64steps_64tokens_lowconf.pt
│   └── plots/
│       ├── heatmaps/
│       │   ├── ar1_heatmap_correct_nopad.png
│       │   ├── ar1_heatmap_halluc_nopad.png
│       │   ├── ar1_heatmap_correct.png
│       │   ├── ar1_heatmap_halluc.png
│       │   └── correlation_heatmap.png
│       ├── ar1_per_sample/
│       │   ├── ar1_per_sample_mean_var_nopad.png
│       │   └── ar1_per_sample_mean_var.png
│       ├── ar1_per_trajectory/
│       │   ├── ar1_per_trajectory_nopad.png
│       │   └── ar1_per_trajectory.png
│       ├── benchmark_global/
│       │   ├── trajectories_mean_var.png
│       │   ├── scatter_mean_vs_var.png
│       │   └── scatter_top_features.png
│       ├── benchmark_per_sample/
│       │   ├── baseline_feature_distributions.png
│       │   ├── markovian_feature_distributions.png
│       │   └── per_step_features.png
│       └── plotly_samples/
│           ├── PlotlyEntropy_*.html
│           └── PlotlyLogProbs_*.html
│
└── DREAM_64steps_64tokens_maskgit/
    ├── outputs_DREAM_64steps_64tokens_maskgit.pt
    └── plots/
        └── (même structure que ci-dessus)
```

### Logique de nommage

- **`_nopad`** : version excluant les tokens de padding (détectés via `histories_x0`)
- **Sans suffixe** : version incluant tous les tokens (même padding)
- Les modes `ar1_per_sample` et `ar1_per_trajectory` génèrent les deux variantes quand `--no-padding` est activé (par défaut)

---

## Architecture du code

```
scripts/plots/
├── __init__.py            # Registre MODES
├── utils.py               # PlotContext (chargement données, split AR1, cache modèles)
├── heatmaps.py            # Mode: heatmaps
├── ar1_per_sample.py      # Mode: ar1_per_sample
├── ar1_per_trajectory.py  # Mode: ar1_per_trajectory
├── benchmark_global.py    # Mode: benchmark_global
├── benchmark_per_sample.py# Mode: benchmark_per_sample
└── plotly_samples.py      # Mode: plotly_samples
```

### `PlotContext` (utils.py)

Classe centrale qui charge les données une seule fois et les met en cache :
- `outputs` : le fichier `.pt` fusionné
- `positions`, `labels` : échantillons matchés avec le JSON d'évaluation
- `entropies`, `masks` : tenseurs pré-calculés
- `padding_2d` : masque de padding (N, D)
- `ar1_idx` / `eval_idx` : split 50/50 stratifié (seed=42)
- `get_ar1_models(no_padding)` : fit AR(1) correct + halluc (caché)
- `get_save_dir(mode_name)` : retourne le chemin de sauvegarde
- `select_samples(n)` : sélection équilibrée de samples depuis la moitié eval
