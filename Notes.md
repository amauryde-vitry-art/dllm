# Notes

## DLLM

### GenerateBaseSamplerOutputs

- **Prise en main du package `dllm`.**

- **Étude de cas simple** sur plusieurs modèles de diffusion :
  - Génération de code (Fibonacci, DFS)
  - Réponse factuelle (capitale de l'Australie, révolution industrielle)
  - Génération narrative libre (histoire d'astronaute)
  
  Génération par blocs → comportement autorégressif bloc par bloc. Première stratégie de remasking testée : **remasking uniforme**.

- **Premières idées sur l'incertitude** : observer le nombre de changements de tokens à chaque pas de diffusion, les log-probabilités, et l'entropie.

- **Nouvelle stratégie de remasking** : on remasque chaque position avec probabilité $\max(p_{\max} - p_{\text{actuelle}}, 0)$. Les tokens remasqués sont redistribués sur les pas de diffusion restants (uniformément sur l'ensemble des pas, pas uniquement au sein du bloc courant).
  - Résultat mitigé : beaucoup d'hésitation du modèle sur des tokens simples.

- **Mise en place de Plotly** pour des visualisations interactives (hover sur les tokens).

- **Mise en place de la pipeline** `GenerateBaseSamplerOutputsAndExtractInfo` (génération → extraction → visualisation).

---

### PipelineTest

**Objectif** : trouver des grandeurs signal pour détecter les hallucinations.

**Setup expérimental** :
- Base de test : TriviaQA (pour avoir un ground truth factuel)
- Évaluation : LLM-as-a-judge, même protocole que le papier *TraceDet*

#### Évolution des features

**Premières features testées :**
- $\text{Mean}(\text{Var}(\text{Entropie sur les pas de diffusion}) \text{ sur les tokens})$
- $\text{Mean}(\text{Var}(\text{LogProbs sur les pas de diffusion}) \text{ sur les tokens})$

→ Ce sont les deux grandeurs avec le plus de signal. Sur un scatter plot à 32 tokens : signal fort. À 64 tokens : plus bruité.

**Ajouts suivants :**
- $\text{Mean}(\text{Entropie})$ et $\text{Mean}(\text{LogProbs})$ → ajoutent du bruit, assez peu corrélés avec l'output (mais l'output étant binaire 0/1, la corrélation est de toute façon limitée).

**Séquences entières :**
- Tentative de garder les séquences entières de features et de mesurer l'AUC → trop peu d'échantillons, montée rapide en grande dimension, le modèle overfitte et donne des AUC artificiellement élevées sans vrai signal.

**Features supplémentaires :**
- Skewness et kurtosis (sur les pas de diffusion) de l'entropie et des log-probs, en gardant la séquence entière → capturer l'asymétrie et la forme de la distribution au cours de la diffusion.
- Quantization sur les pas de diffusion + calcul d'entropie par position de token.

**Analyse PCA :**
- Avec autant de features et si peu d'échantillons, le problème est mal posé. Analyse PCA : ~8 composantes avant que la variance expliquée devienne du bruit. En projetant sur ces composantes et en relançant la classification → l'AUC chute, résultat nul.
- **Questions ouvertes** : PCA sur séquences entières vs summary statistics ? Quelle feature est la plus corrélée avec la 1ère composante principale ?

**Idée prometteuse — Semantic Entropy :**
- Clusteriser les embeddings (token IDs → espace sémantique) et calculer une entropie inter-clusters.
- Intuition : si le modèle hésite entre `[SPACE]` et `[NEWLINE]`, c'est bénin. S'il hésite entre deux mots de sens très différents, c'est un signal d'hallucination.

#### Modèles testés

**Random Forest** avec grid search CV :
```python
param_grid = {
    "n_estimators": [1500, 2500, 3500],
    "max_depth": [1, 2, 3],
    "max_features": [1, 2, "sqrt", None],
    "min_samples_split": [2, 5, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "random_state": [42],
}
```

**Régression logistique** (ElasticNet) avec grid search CV :
```python
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    "l1_ratio": [0, 0.5, 1],
    "solver": ["saga"],
}
```

---

## To do

- [ ] Clustering des embeddings (semantic entropy)
- [ ] Analyse de corrélation PCA : comparer summary statistics vs PCA sur séquences entières, identifier les features les plus corrélées avec PC1
- [ ] Tester d'autres métriques que ROC AUC


## Clusterisation des embeddings pour l'entropie sémantique

### Approche globale (abandonnée)

Le clustering statique de tout le vocabulaire (HDBSCAN, KMeans, RQ sur les input embeddings) est trop rigide :
- Le nombre de clusters est arbitraire et la qualité dépend fortement de cet hyperparamètre.
- Deux tokens sémantiquement proches peuvent tomber dans des clusters différents (effet de frontière).
- Les embeddings d'entrée capturent la similarité morphologique/syntaxique, pas sémantique : la cosine similarity entre tokens ne reflète pas toujours la proximité de sens (ex. "avocat" fruit vs "avocat" métier sont dans le même cluster, alors qu'on voudrait les distinguer selon le contexte).

**Piste intéressante mais coûteuse** : la cosine similarity sur embeddings statiques ne capture pas la vraie proximité sémantique (ex. "avocat" fruit vs métier). On pourrait utiliser un **cross-encoder pré-entraîné** (type SBERT cross-encoder) qui prend deux mots en entrée et produit directement un score de similarité appris, bien plus fin qu'une simple distance cosinus. Problème : le coût est prohibitif (un forward pass du cross-encoder par paire de tokens candidats, à chaque position, à chaque step de diffusion).

### Approche locale (retenue) : clustering dynamique au sein du top-K

**Idée** : ne pas clusteriser tout le vocabulaire a priori, mais clusteriser *à la volée* les tokens candidats du modèle à chaque position/step.

1. Prendre les top-K tokens (par probabilité softmax), typiquement K=20.
2. Calculer la cosine similarity pairwise entre leurs embeddings → matrice K×K.
3. Agglomerative clustering avec seuil de distance (pas de K fixe → le nombre de clusters émerge).
4. Sommer les probabilités par cluster → Shannon entropy sur les clusters.

**Avantages** :
- Pas d'hyperparamètre global (nombre de clusters) ; seuls le top-K et le seuil cosine comptent.
- Adaptatif : le clustering dépend de la distribution du modèle à chaque position.
- Si le modèle hésite entre `\n`, `\t`, espace → 1 cluster → entropie ~0 (hésitation bénigne).
- Si le modèle hésite entre "dog" et "equation" → 2 clusters → entropie élevée (signal d'hallucination).

**Variante** : ne garder que les tokens avec logit > 0 (i.e. prob > 1/V, le modèle leur accorde plus que le hasard) au lieu d'un top-K fixe.

### Approche continue (retenue) : dispersion pondérée sans clustering

Le clustering local reste fragile à cause du seuil de distance (deux tokens à 0.69 vs 0.71 de cosine sim basculent brutalement). On peut supprimer le clustering entièrement et utiliser une **dispersion pondérée par les probabilités** :

$$D(\text{pos}) = \sum_{i,j} p_i \cdot p_j \cdot (1 - \cos(\hat{e}_i, \hat{e}_j))$$

**Simplification algébrique** : en développant, on obtient une formule fermée qui ne nécessite ni top-K ni calcul pairwise :

$$D = 1 - \left\|\sum_i p_i \hat{e}_i\right\|^2 = 1 - \|\bar{e}\|^2$$

où $\bar{e} = \sum_i p_i \hat{e}_i$ est le barycentre pondéré des embeddings L2-normalisés. La somme porte sur **tout le vocabulaire**, sans top-K.

**Propriétés :**
- **Aucun hyperparamètre** : ni top-K, ni seuil, ni nombre de clusters.
- **Continue** : pas d'effet de seuil, la contribution de chaque token est lisse.
- **Pondérée par les probas** : un token à 1% contribue 100× moins qu'un token à 10%.
- **Implicitement contextuelle** : les $p_i$ viennent du softmax des logits conditionnés au contexte ; la pondération rend donc la mesure dépendante du contexte sans recourir à un modèle externe (SBERT, etc.).

**Interprétation :**
- $\|\bar{e}\|^2 \approx 1$ → distribution concentrée sur des tokens proches → $D \approx 0$ (bénin).
- $\|\bar{e}\|^2 \ll 1$ → distribution dispersée sur des tokens éloignés → $D$ élevé (signal d'hallucination).
- Modèle sûr ($p_1 \approx 1$) → $\bar{e} \approx \hat{e}_1$ → $\|\bar{e}\|^2 \approx 1$ → $D \approx 0$.

**Coût** : un seul matmul `p @ E_normed` soit `(B, T, V) @ (V, D) → (B, T, D)` puis norme. Complexité $O(V \cdot D)$ par position, linéaire en V. Overhead < 1% vs le forward pass du modèle. Vectorisable sur toutes les positions d'un step.

**Implémenté** dans `dllm/core/samplers/mdlm.py` (`compute_semantic_dispersion`) et stocké dans `histories_semantic_dispersion`.


### Nouvelle approche pour le clustering d'embedding (STC)

Basé sur l'article *"Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models"* (EACL 2026) — https://arxiv.org/abs/2603.20161v1

**Méthode implémentée** (cf. `PipelineTest/clusteringTokens.py`) :

1. **Extraction des embeddings** : on récupère les **input embeddings** (couche d'embedding du modèle) et les **output embeddings** (poids du `lm_head`) de LLaDA-8B-Instruct. Chaque token a donc deux vecteurs de dimension $D$.

2. **Concaténation** : les deux vecteurs sont concaténés pour former une représentation unifiée de dimension $2D$ par token. L'idée est que les input embeddings capturent la représentation syntaxique/morphologique tandis que les output embeddings capturent la sémantique prédictive — leur combinaison donne une représentation plus riche.

3. **Filtrage** :
   - **Stopwords** : exclus via la liste NLTK (mots grammaticaux sans contenu sémantique : "the", "is", "a", etc.). Chacun reçoit son propre cluster singleton.
   - **Chiffres arabes** : exclus car des chiffres avec des embeddings proches ne sont pas mathématiquement équivalents (ex. "42" et "43" ont des embeddings voisins mais des sens différents). Chacun reçoit aussi un cluster singleton.

4. **Normalisation L2** : les vecteurs concaténés sont L2-normalisés pour que la distance euclidienne corresponde à la distance cosinus.

5. **Agglomerative Clustering** : scikit-learn `AgglomerativeClustering` avec distance cosinus, linkage average, et $n = 16\,000$ clusters (valeur empirique issue de l'article LENS).

   **Principe de l'Agglomerative Clustering** : c'est un algorithme de clustering hiérarchique **ascendant** (bottom-up). Il part de $N$ clusters singletons (un par point) et fusionne itérativement les deux clusters les plus proches, jusqu'à atteindre le nombre de clusters souhaité.

   - **Étape 0** : chaque point est son propre cluster → $N$ clusters.
   - **Étape $k$** : on cherche la paire de clusters $(C_i, C_j)$ qui minimise la distance inter-clusters, on les fusionne → $N - k$ clusters.
   - On s'arrête quand on atteint $n_\text{clusters} = 16\,000$.

   La **distance inter-clusters** dépend du **linkage** choisi :
   - **Single** : $d(C_i, C_j) = \min_{a \in C_i, b \in C_j} d(a, b)$ → sensible au bruit (effet de chaîne).
   - **Complete** : $d(C_i, C_j) = \max_{a \in C_i, b \in C_j} d(a, b)$ → clusters compacts mais petits.
   - **Average** (utilisé ici) : $d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{a \in C_i} \sum_{b \in C_j} d(a, b)$ → compromis entre single et complete, robuste.
   - **Ward** : minimise l'augmentation de variance intra-cluster (uniquement distance euclidienne).

   **Complexité** : $O(N^2)$ en mémoire (matrice de distances) et $O(N^2 \log N)$ en temps. C'est pour ça que ça coûte cher avec $N = 125\,733$ tokens — la matrice de distances pèse $\sim 126$ Go. En comparaison, KMeans est $O(N)$ en mémoire.

   **Avantage par rapport à KMeans** : l'agglomerative clustering ne suppose pas de forme de cluster (pas de biais sphérique) et fonctionne nativement avec la distance cosinus. KMeans optimise la variance euclidienne, ce qui est moins adapté aux embeddings de haute dimension.

6. **Évaluation** : silhouette score (cosinus) calculé sur un sous-échantillon de 50k tokens pour vérifier la qualité du clustering.

7. **Sauvegarde** :
   - `token_to_cluster_stc.json` : mapping `token_id → cluster_id` pour utilisation à l'inférence.
   - `token_to_cluster_stc_contents.json` : contenu lisible de chaque cluster (tokens décodés, triés par taille de cluster décroissante) pour inspection visuelle.

**Différence avec l'approche locale (top-K)** : le STC est un clustering **global et offline** (pré-calculé une seule fois), alors que l'approche top-K cluster entropy fait un clustering **local et dynamique** à chaque position/step. Les deux sont complémentaires : le STC permet de calculer une entropie sémantique efficacement à l'inférence (simple lookup dans le mapping), tandis que la dispersion continue et le top-K cluster entropy ne nécessitent pas de pré-calcul mais sont plus coûteux.