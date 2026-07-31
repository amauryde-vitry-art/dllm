# Bizarrerie

je ne comprends pas pq la variance accross voc puis moyennée sur les pas de diff a un plus grand pouvoir prédictif ?


# Recherche de features 

- on a d'abord trouvé du signal sur la moyenne de la variance de l'entropy faite sur toute la distribution(variance faite sur les pas de diffusion puis moyenne faite sur les tokens)

- Il s'est aussi posé la quesiton de calculer une entropy sémantiquement plus intéressante: clusteriser les embeddings (cf agglomerative clustering faite sur la concaténation des embeddings en sortie et en entrée du modele LLaDA, /home/adevitry/dllm/Notes.md). Idée: calculer une entropy à partir de ces clusters. Ce n'est pas grave d'hésiter entre un espace et un saut de ligne. 

- Au cours de cette exploration on a trouvé une autre métrique qui pouvait être intéressante, qui s'appelle la semantic dispersion (detail dans /home/adevitry/dllm/Notes.md)

- Nouvelle idée: c'est de calculer le temps de relaxation. On cherche le beta et le alpha de la régression linaire entre (ln(mean(entropy across token))) VS pas de diffusion. Si le tau est grand, le modele mais plus longtemps à arriver à son état stable donc c'est plus propice à une hallucination

- Au cours de cette idée: on a voulu calculer le taux de diff  tous les 10 pas de temps, puis de faire la moyenne de la mean entropy

- ce temps de relaxation on a cherché aussi à le calculer pour la moyenne des log probs

- Pour raffiner, on a tenté de calculer ce temps de relaxation en faisant la moyenne de l entropy uniquement sur les tokens où on a une grosse hésitation.

# Features correlation

## Notations

- Soit $E_{t,j}$ l'entropy au pas de diffusion $t$ pour le token $j$
- Soit $L_{t,j}$ la log-probabilité au pas $t$ pour le token $j$
- Soit $M_{t,j}$ le masque ($1$ = token encore masqué)
- Soit $SE_{t,j}$ la semantic entropy
- Soit $SD_{t,j}$ la semantic dispersion
- $T$ = nombre de pas de diffusion, $K$ = nombre de tokens
- top-$k$ = tokens ayant la plus grande entropy moyenne au cours de la diffusion

## Tableau des features calculées (CreateMetrics.py)

### Variances principales

- VarEntropy: $\mathrm{VarEntropy}=\frac{1}{K}\sum_j \mathrm{Var}_t(E_{t,j})$
- VarEntropyAcrossTokens: $\mathrm{VarEntropyAcrossTokens}=\frac{1}{T}\sum_t \mathrm{Var}_j(E_{t,j})$
- VarMaskedEntropy: $\mathrm{VarMaskedEntropy}=\frac{1}{K}\sum_j \mathrm{Var}_{t:\,M_{t,j}=1}(E_{t,j})$
- VarMaskedEntropyAcrossTokens: $\mathrm{VarMaskedEntropyAcrossTokens}=\frac{1}{T}\sum_t \mathrm{Var}_{j:\,M_{t,j}=1}(E_{t,j})$

### Variances sémantiques

- VarSemanticEntropy: $\mathrm{VarSemanticEntropy}=\frac{1}{K}\sum_j \mathrm{Var}_t(SE_{t,j})$
- VarSemanticEntropyAcrossToken: $\mathrm{VarSemanticEntropyAcrossToken}=\frac{1}{T}\sum_t \mathrm{Var}_j(SE_{t,j})$
- VarSemanticDispersion: $\mathrm{VarSemanticDispersion}=\frac{1}{K}\sum_j \mathrm{Var}_t(SD_{t,j})$
- VarSemanticDispersionAcrossToken: $\mathrm{VarSemanticDispersionAcrossToken}=\frac{1}{T}\sum_t \mathrm{Var}_j(SD_{t,j})$
- VarSemanticEntropyMasked: $\mathrm{VarSemanticEntropyMasked}=\frac{1}{K}\sum_j \mathrm{Var}_{t:\,M_{t,j}=1}(SE_{t,j})$
- VarSemanticEntropyMaskedAcrossTokens: $\mathrm{VarSemanticEntropyMaskedAcrossTokens}=\frac{1}{T}\sum_t \mathrm{Var}_{j:\,M_{t,j}=1}(SE_{t,j})$

### Moyennes

- MeanLogProbs: $\mathrm{MeanLogProbs}=\mathrm{mean}_{t,j}(L_{t,j})$
- MeanEntropy: $\mathrm{MeanEntropy}=\mathrm{mean}_{t,j}(E_{t,j})$
- MeanMaskedEntropy: $\mathrm{MeanMaskedEntropy}=\mathrm{mean}_{t,j}(E_{t,j}\,M_{t,j})$
- MeanEntropyJustUnmasked = moyenne de l'entropy au pas exact de unmask des tokens

### Dérivées discrètes (tau)

- Tau_s: $\tau_s=\frac{\mathrm{mean}_j(E_{s+w,j})-\mathrm{mean}_j(E_{s,j})}{w},\quad w=10$
- MeanTau: $\mathrm{MeanTau}=\mathrm{mean}_s(\tau_s)$
- VarTau: $\mathrm{VarTau}=\mathrm{Var}_s(\tau_s)$

### Fits linéaires log

- AlphaEntropy, BetaEntropy: moyenne sur j des coefficients du fit
  $\log(E_{t,j}+\varepsilon)=\alpha_j t+\beta_j$
- AlphaLogProbs, BetaLogProbs: moyenne sur j des coefficients du fit
  $\log(L_{t,j}+\varepsilon)=\alpha_j t+\beta_j$
- AlphaEntropyAvg, BetaEntropyAvg: fit sur la moyenne top-k des entropies
  $\log\!\left(\mathrm{mean}_{j\in\mathrm{top-}k}(E_{t,j})+\varepsilon\right)=\alpha t+\beta$
- AlphaLogProbsAvg, BetaLogProbsAvg: fit sur la moyenne des logprobs
  $\log\!\left(\mathrm{mean}_j(L_{t,j})+\varepsilon\right)=\alpha t+\beta$

### Fit non linéaire de relaxation

- C_ct, Tau_ct, M_ct via
  $E(t)=C\,t\,\exp\!\left(-\frac{t-m}{\tau}\right)$
  fit sur la courbe d'entropy moyenne top-20

### Feature spécifique unmask

- VarEntropyJustUnmasked = variance de l'entropy au pas exact de unmask des tokens


## Correlation entre features

Analyse faite sur 2048 samples et 29 features.

### Batchs de features très corrélées entre elles (|corr| >= 0.95)

1. Bloc dynamique globale (forte redondance):
	VarEntropy, VarEntropyAcrossTokens, VarSemanticEntropyAcrossToken, VarSemanticDispersion,
	VarLogProbs, MeanLogProbs, MeanEntropy, AlphaLogProbs, BetaLogProbs,
	AlphaLogProbsAvg, BetaLogProbsAvg, MeanMaskedEntropy

2. Bloc masqué across tokens:
	VarMaskedEntropyAcrossTokens, VarSemanticEntropyMaskedAcrossTokens

3. Bloc paramètres du fit C*t*exp(-(t-m)/tau):
	Tau_ct, M_ct

4. Bloc masqué token-wise:
	VarMaskedEntropy, VarSemanticEntropyMasked

### Paires les plus corrélées (exemples)

- VarEntropy VS VarSemanticEntropy: 0.99
- VarMaskedEntropy vs VarSemanticEntropyMasked: 0.993
- BetaLogProbs vs BetaLogProbsAvg: 0.990
- MeanEntropy vs MeanMaskedEntropy: 0.988
- VarEntropyAcrossTokens vs VarSemanticEntropyAcrossToken: 0.988
- AlphaLogProbs vs AlphaLogProbsAvg: 0.988
- VarMaskedEntropyAcrossTokens vs VarSemanticEntropyMaskedAcrossTokens: 0.984
- Tau_ct vs M_ct: -0.957



## Correlation avec les labels

Corrélation de Pearson feature -> label (hallucination=1).

### Plus fortes corrélations positives

1. VarMaskedEntropyAcrossTokens: 0.476
2. VarSemanticEntropyMaskedAcrossTokens: 0.458
3. AlphaEntropyAvg: 0.360
4. VarEntropyJustUnmasked: 0.355
5. VarEntropyAcrossTokens: 0.307
6. VarSemanticEntropyAcrossToken: 0.266
7. MeanMaskedEntropy: 0.210
8. Tau_ct: 0.204
9. VarSemanticDispersionAcrossToken: 0.197
10. MeanEntropyJustUnmasked: 0.193

### Corrélations négatives notables

- BetaEntropyAvg: -0.174
- M_ct: -0.174
- MeanTau: -0.155
- MeanLogProbs: -0.092
- BetaLogProbs: -0.071



## Features selection

On utilise une régression logistique avec pénalité L1 (solver saga) sur les features standardisées.

### Principe

- Le L1 pousse beaucoup de coefficients exactement à 0
- En présence de features redondantes (très corrélées), le modèle garde en général quelques représentants et annule le reste
- On obtient donc un modèle plus sparse et interprétable

### Résultat observé (L1 Logistic)

Features non nulles retenues (11):

- VarMaskedEntropyAcrossTokens
- VarSemanticEntropyAcrossToken
- MeanTau
- AlphaLogProbs
- AlphaEntropy
- BetaEntropy
- AlphaEntropyAvg
- BetaEntropyAvg
- M_ct
- VarEntropyJustUnmasked
- VarSemanticEntropyMasked

Performances (sur les mêmes données d'entraînement):

- PR AUC: 0.861
- ROC AUC: 0.839

Note méthodo:

- Ces scores sont optimistes car calculés sur le train.
- Pour une estimation plus robuste: nested CV ou split test strict après sélection de features.

## Interprétation de la "bizarrerie"

Pourquoi VarEntropyAcrossTokens peut être plus prédictive que d'autres stats:

1. Elle mesure l'hétérogénéité instantanée entre positions (au même pas de diffusion), donc directement l'incertitude structurelle de la séquence.
2. Cette hétérogénéité capte souvent mieux les cas où certaines positions restent très ambiguës alors que d'autres sont déjà stabilisées.
3. Les features de même famille (semantic entropy, logprobs, masked) sont très corrélées, donc VarEntropyAcrossTokens agit comme proxy d'un signal latent partagé.
4. Le fait de moyenner sur les pas réduit le bruit temporel et garde un indicateur global stable par sample.




# Model utilisé

- on a fait 3 modeles différents: RF, reg logistic et lgbm en essayant d'over fitt le moins possible.



# To do:

- strategy pour feature selection
- concentration du signal ?









