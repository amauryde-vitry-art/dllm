# Notes — Feature Selection pour la détection d'hallucinations dans les dLLMs

## Contexte

On dispose de ~37 features extraites des trajectoires de démasquage des modèles de diffusion (LLaDa, DREAM), calculées sur 24 configurations (2 modèles × 4 résolutions steps/tokens × 3 datasets QA). L'objectif est de sélectionner un sous-ensemble compact et robuste de features pour prédire si une réponse est hallucinée (classification binaire).

## Pourquoi Stability Selection ?

Le problème classique de la sélection de features par L1 (Lasso / Logistic Regression pénalisée) est la **sensibilité au choix de la régularisation** : un λ trop petit sélectionne tout, un λ trop grand ne sélectionne rien, et le résultat change radicalement avec de petites perturbations des données. C'est un problème critique quand on veut justifier dans un papier *pourquoi* ces features et pas d'autres.

**Stability Selection** (Meinshausen & Bühlmann, 2010, JRSS-B, ~3700 citations) résout ce problème en combinant sub-sampling et randomisation de la pénalité, avec une **garantie théorique sur le nombre de faux positifs**.

## Contenu de l'article (Meinshausen & Bühlmann, 2010)

### Idée centrale

Au lieu de faire une seule régression L1, on répète B fois (B = 500 ici) la procédure suivante :

1. **Sub-sampling** : tirer 50% des données sans remise.
2. **Randomisation** : multiplier chaque feature par un poids aléatoire $w_j \sim \text{Uniform}(0.5, 1)$. Cela « handicape » aléatoirement les features : une feature avec $w_j$ petit a moins de chance d'être sélectionnée par le Lasso. Seules les features véritablement informatives survivent malgré ce handicap.
3. **Régression L1** : ajuster une Logistic Regression pénalisée L1 avec un C tiré aléatoirement dans une grille log-espacée $[10^{-2}, 10^{0}]$.
4. **Enregistrer** quelles features ont un coefficient $\beta_j \neq 0$.

On obtient pour chaque feature $j$ une **probabilité de sélection** :

$$\hat{\Pi}_j = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}[\beta_j^{(b)} \neq 0]$$

Les features avec $\hat{\Pi}_j > \pi_{\text{thr}}$ (seuil typique : 0.6 à 0.9) sont retenues.

### Garantie théorique (Théorème 1)

Le résultat clé est une borne sur l'espérance du nombre de faux positifs $V$ (features sélectionnées à tort) :

$$\mathbb{E}[V] \leq \frac{q^2}{(2\pi_{\text{thr}} - 1) \cdot p}$$

où :
- $q$ = nombre moyen de features sélectionnées par sous-échantillon
- $\pi_{\text{thr}}$ = seuil de probabilité de sélection (> 0.5)
- $p$ = nombre total de features

Cette borne est **non-asymptotique**, ne dépend pas de la distribution des données, et ne nécessite aucune hypothèse de sparsité. Pour $\pi_{\text{thr}} = 0.9$, on a $\mathbb{E}[V] \leq q^2 / (0.8p)$.

### Pourquoi la randomisation des poids ?

La condition KKT pour que $\beta_j = 0$ dans le Lasso est :

$$\left| \frac{1}{n} X_j^\top (y - \hat{p}) \right| \leq \frac{w_j}{C}$$

Avec un poids $w_j$ petit, la borne droite est plus stricte → il faut une corrélation plus forte avec le résidu pour que la feature soit sélectionnée. Seules les features avec un vrai signal survivent systématiquement à cette perturbation.

### Avantages par rapport aux alternatives

| Méthode | Problème |
|---|---|
| Lasso simple | Résultat dépend fortement de λ, instable |
| Forward/backward stepwise | Pas de garantie théorique, greedy |
| Boruta | Test statistique vs. features shadow, mais pas de borne E[V] |
| mRMR | Greedy, sensible à l'estimation de MI sur petits échantillons |

Stability Selection combine le meilleur des deux mondes : une méthode pratique (sub-sampling + L1) avec des garanties formelles (borne E[V]).

## Pipeline implémenté

### Étape 1 — Pré-filtrage par corrélation (`prefilter.py`)
Suppression des features quasi-dupliquées ($|r| > 0.95$). Parmi chaque paire corrélée, on garde celle qui a la plus forte corrélation point-bisériale avec le label.

### Étape 2 — Sélection (3 méthodes)

1. **Stability Selection** (`stability_selection.py`) — méthode principale, décrite ci-dessus.
2. **mRMR** (`mrmr_selection.py`) — Maximum Relevance Minimum Redundancy (Peng et al., 2005). Sélection gloutonne maximisant $I(f;y) - \frac{1}{|S|} \sum_{s \in S} I(f;s)$. Utilisé en ablation.
3. **Boruta** (`boruta_selection.py`) — Kursa & Rudnicki (2010). Test contre des features shadow permutées via Random Forest. Utilisé en ablation.

### Étape 3 — Évaluation (`evaluate.py`)

Deux protocoles :

- **`benchmark_evaluate_with_selection`** : split séquentiel 80/20 identique au benchmark (`Baseline_and_markovian_features.py`). Feature selection sur le train uniquement → **pas de leakage**. Résultats directement comparables aux baselines.
- **`nested_cv_evaluate_with_selection`** : nested CV 5×5 avec feature selection refaite dans chaque fold outer. Plus rigoureux mais pas comparable au benchmark existant.

### Étape 4 — Analyse (`FeatureFrequency.py`)
Agrégation des features sélectionnées sur les 24 configurations pour identifier les features les plus robustes (fréquence de sélection cross-config).

## Références

- Meinshausen, N. & Bühlmann, P. (2010). *Stability Selection*. Journal of the Royal Statistical Society, Series B, 72(4), 417–473.
- Peng, H., Long, F. & Ding, C. (2005). *Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy*. IEEE TPAMI, 27(8), 1226–1238.
- Kursa, M.B. & Rudnicki, W.R. (2010). *Feature Selection with the Boruta Package*. Journal of Statistical Software, 36(11).
