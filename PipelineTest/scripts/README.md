# PipelineTest/scripts

## Centralized Scripts

Three entry-point scripts that orchestrate the full pipeline.

### Scripts

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `run_evaluation.py` | Train classifiers, report metrics | `evaluation_results.json` |
| `run_feature_selection.py` | Feature ranking & selection | Heatmaps, PCA, univariate AUC |
| `run_analysis.py` | Generate all plots | Trajectories, scatter, AR1 recon, heatmaps |

---

### `run_evaluation.py`

Computes all features (baseline + markovian + AR1), trains Logistic Regression with GridSearchCV, and compares feature sets.

```bash
python -m PipelineTest.scripts.run_evaluation --config llada
python -m PipelineTest.scripts.run_evaluation --config dream
```

**Output**: ROC-AUC, PR-AUC, accuracy for each feature group (Baseline, Markovian, AR1, Combined).

---

### `run_feature_selection.py`

Analyzes feature redundancy and discriminative power:
- Correlation heatmap (features + label)
- PCA scree plot (cumulative variance)
- Univariate ROC-AUC ranking (per-feature)
- L1-regularized coefficient ranking

```bash
python -m PipelineTest.scripts.run_feature_selection --config llada
```

---

### `run_analysis.py`

Generates all visualizations:
- Population-level mean/var entropy trajectories (correct vs halluc)
- Scatter: MeanMaskedEntropy vs VarMaskedEntropyAcrossTokens
- AR(1) per-sample reconstruction (data vs correct model vs halluc model)
- AR(1) parameter heatmaps (φ, intercept, σ)
- Feature distribution histograms (baseline + markovian)

```bash
python -m PipelineTest.scripts.run_analysis --config llada --n-samples 20
```

---

### Configuration

Both LLaDA and Dream configs are built-in. Add new configs by editing the `CONFIGS` dict in each script:

```python
CONFIGS = {
    "llada": {
        "outputs_path": "PipelineTest/res/LLADA_.../outputs_*.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_*.json",
        "name": "LLaDA_64steps_64tokens",
    },
    ...
}
```
