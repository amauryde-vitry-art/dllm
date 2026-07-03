# GenerateBaseSamplerOutputsAndExtractInfo

Pipeline d'analyse du processus de diffusion des modèles de langage masqués (MDLM). Ce module permet de **générer des séquences en enregistrant l'historique complet** de chaque étape de diffusion, **d'extraire des métriques** à partir de cet historique, puis de **visualiser** l'évolution du processus.

---

## Architecture

```
GenerateWithMDLMSampler.py    → Génération avec historique complet
GetInfoFromBaseSamplerOutput.py → Extraction des métriques depuis l'historique
PlotResults.py                 → Visualisations (matplotlib + plotly)
run_experiments.py             → Orchestration des deux types d'expériences
main.py                        → Point d'entrée (configuration des modèles et prompts)
```

---

## 1. `GenerateWithMDLMSampler.py` — Génération avec historique complet

La fonction `CreateBaseSampleWithHistory` exécute le sampling d'un modèle de diffusion en enregistrant **l'état complet de chaque étape de débruitage**. Elle retourne un objet `BaseSamplerOutputCompleteHistory` qui contient :

| Champ | Description |
|-------|-------------|
| `histories_x` | Séquence acceptée de token ids (canvas) à chaque step — l'état réel du texte après sélection des tokens |
| `histories_x0` | Séquence proposée de token ids par le modèle à chaque step (avant filtrage par confiance) |
| `histories_logprobs` | Confiance du modèle (probabilité du token prédit) pour chaque position à chaque step |
| `histories_unmask_logprobs` | Probabilité assignée par le modèle aux tokens **déjà démasqués** (mesure de cohérence) |
| `histories_mask` | Masque binaire  indiquant les positions encore masquées à chaque step |
| `histories_remasking` | Masque binaire des positions **remasquées** à chaque step (uniquement avec remasking) |
| `histories_H` | Score $H = p_{\text{max}} - p_{\text{token\_courant}}$ — indicateur d'incertitude utilisé pour décider du remasking |
| `histories_entropy` | Entropie de la distribution de probabilité à chaque position |
| `histories_num_transfer_tokens` | Nombre de tokens à démasquer par step (budget dynamique, ajusté par le remasking) |
| `start_idx_history` | Index de début de la zone de génération pour chaque exemple du batch |
| `max_new_tokens` | Nombre de tokens à générer |
| `block_size` | Taille d'un bloc de diffusion |

---

## 2. `GetInfoFromBaseSamplerOutput.py` — Extraction d'information

Ce module fournit des fonctions utilitaires pour extraire et reformater les données de `BaseSamplerOutputCompleteHistory` en listes indexées par exemple du batch :

| Fonction | Retour |
|----------|--------|
| `getLogProbs(outputs)` | Probabilité des tokens proposés par le model par position et par step |
| `getUnmaskLogProbs(outputs)` | Probabilité des tokens déjà révélés (stabilité) |
| `getEachStepGeneratedSequence(outputs, tokenizer)` | Texte décodé du canvas à chaque step |
| `getEachStepProposedSequence(outputs, tokenizer)` | Texte décodé des prédictions $x_0$ à chaque step |
| `getEachStepProposedTokenIdSequence(outputs, tokenizer)` | IDs des tokens proposés (numpy) |
| `getEachStepMask(outputs, remask=False)` | Masques de diffusion (ou de remasking si `remask=True`) |
| `getEachStepChange(outputs)` | Matrice binaire de changements de tokens id entre steps consécutifs par position |
| `getLevenshtein(Proposed_sequences)` | Distance de Levenshtein token-à-token entre steps |
| `getH(outputs)` | Score H (différence confiance max vs confiance token courant) |
| `getNumTransferTokens(outputs)` | Budget de tokens transférés par step |
| `getEntropy(outputs)` | Entropie par position à chaque step |

---

## 3. `PlotResults.py` — Visualisations

Génère des heatmaps et graphiques interactifs pour analyser le processus de diffusion :

- **LogProbs** — Heatmap de la confiance du modèle (axes : step × position)
- **UnmaskLogProbs** — Stabilité des tokens déjà révélés au fil des steps
- **Masks** — Évolution du masque de diffusion (quelles positions sont encore masquées)
- **Changes** — Positions où le modèle change d'avis entre deux steps
- **Levenshtein** — Distance d'édition entre propositions successives
- **H** — Heatmap du score d'incertitude $H$
- **NumTransferredTokens** — Évolution du budget de démasquage
- **AttentionMask** — Visualisation du masque d'attention

Les positions **remasquées** sont superposées en noir sur les heatmaps lorsque le remasking est actif.

Des versions interactives (Plotly) avec hover sur les tokens sont générées en parallèle.

---

## 4. `run_experiments.py` — Deux types d'expériences

### Expérience 1 : Sans remasking (`run_experiment_MDML`)

Utilise `MDLMSamplerWithCompleteHistory`. Le processus de diffusion standard :
1. Le canvas commence entièrement masqué (zone de génération)
2. À chaque step, le modèle propose des tokens pour toutes les positions masquées
3. Les $k$ positions avec la plus haute confiance sont démasquées définitivement
4. Le budget $k$ suit un scheduler linéaire (plus de tokens démasqués vers la fin)
5. Un token une fois démasqué ne peut **jamais** redevenir masqué

### Expérience 2 : Avec remasking (`run_experiment_MDML_remasking`)

Utilise `MDLMSamplerRemaskingWithCompleteHistory`. Le processus ajoute une étape de **remasking** :

#### Mécanisme du remasking

Après chaque step de démasquage (sauf le dernier), le sampler peut **remasquer** des tokens déjà révélés si le modèle estime qu'ils sont incohérents avec le contexte actuel :

1. **Calcul de H** : Pour chaque position déjà démasquée, on calcule $H_i = p_{\max}(i) - p_{\text{token\_courant}}(i)$, où $p_{\max}$ est la confiance maximale du modèle et $p_{\text{token\_courant}}$ est la probabilité que le modèle assigne au token actuellement présent à cette position.

2. **Décision de remasking** : Les positions avec $H > 0$ (le modèle préférerait un autre token) sont candidates au remasking. Le remasking est **stochastique** : chaque position candidate est remasquée selon une loi de Bernoulli de paramètre $\text{clamp}(H_i, 0, 1)$.

3. **Redistribution du budget** : Les tokens remasqués doivent être re-démasqués plus tard. Le budget de démasquage (`num_transfer_tokens`) est redistribué uniformément sur les steps restants, avec le surplus distribué en priorité aux steps ayant le budget le plus faible.

4. **Restriction à la zone de génération** : Seuls les tokens dans la zone de génération (hors prompt) peuvent être remasqués.

Ce mécanisme permet au modèle de **corriger ses erreurs** au cours du processus de diffusion, plutôt que de s'engager définitivement sur des tokens de faible qualité.

---

## Utilisation

```bash
python main.py
```

La configuration se fait dans `main.py` :
- **Modèles** : liste `MODELS_TO_TEST` avec chemin, nom, dossier de sortie, et flag `remasking`
- **Prompts** : liste `messages` (format chat)
- **Paramètres du sampler** : `SamplerConfig` (nombre de steps, taille de bloc, température, stratégie de remasking)

```python
@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
```

Les résultats (plots + fichiers texte) sont sauvegardés dans le dossier spécifié par `model_cfg['dir']`.

---

## Résultats disponibles

```
Results/
├── LLaDa_8B_no_remasking/        # LLaDa 8B, sampling standard
├── LLaDa_8B_no_remasking_1Block/ # LLaDa 8B, standard, 1 seul bloc
├── LLaDa_8B_remasking/           # LLaDa 8B, avec remasking
└── LLaDa_8B_remasking_1Block/    # LLaDa 8B, avec remasking, 1 seul bloc
```