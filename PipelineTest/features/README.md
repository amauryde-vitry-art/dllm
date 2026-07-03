# PipelineTest/features

## Feature Extraction Modules

Modular feature extraction for hallucination detection, organized by theoretical motivation.

### Architecture

```
features/
├── utils.py       # Shared: output loading, labels, padding detection
├── baseline.py    # Simple mean/variance aggregations
├── markovian.py   # Dynamic features from Markovian model (Section 2.3 of report)
├── ar1.py         # AR(1) model fitting & features (Section 2.4 of report)
└── semantic.py    # Semantic entropy & dispersion (Section 2.2 of report)
```

### Module Descriptions

#### `baseline.py` — Baseline Features
Simple statistical aggregations over the entropy/logprob trajectory:
- **Mean features**: `mean_entropy`, `mean_masked_entropy`, `mean_entropy_just_unmasked`, `mean_logprob`
- **Variance features**: `var_entropy`, `var_masked_entropy_across_tokens`, `var_logprob`, etc.
- **No-padding variants**: same features excluding padding tokens

#### `markovian.py` — Markovian Dynamic Features
Features motivated by the exponential decay model $S(t) \approx C \cdot t \cdot e^{-t/\tau}$:
- **Exponential decay fit** ($\alpha$, $\beta$): slope of $\ln H(t)$ → estimates $-1/\tau$
- **Top-k averaged decay**: focuses on hardest tokens
- **Parametric fit** ($C$, $\tau$, $m$): direct fit of rise-then-decay shape
- **Finite differences** (MeanTau): discrete rate of change
- **Shape features on V(t)**: AUC, max, argmax, skewness, kurtosis, curvature

#### `ar1.py` — AR(1) Model
Discretisation of the Ornstein-Uhlenbeck process:
- `fit_ar1_model()` / `fit_ar1_model_no_padding()`: fit $\varphi_{t,d}$, $c_{t,d}$, $\sigma_{t,d}$ per (step, token)
- `get_ar1_features()`: mean/var of predictions from correct vs hallucination models

#### `semantic.py` — Semantic Features
Based on semantic token clustering (STC):
- Variance of semantic entropy (across steps, across tokens, masked variants)
- Variance of semantic dispersion

### Usage

```python
from PipelineTest.features.utils import load_outputs, match_samples, get_pad_token_id
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features
from PipelineTest.features.ar1 import fit_ar1_models, get_ar1_features

outputs = load_outputs("PipelineTest/res/.../outputs_*.pt")
positions, labels, data = match_samples("PipelineTest/res/eval/results_*.json", outputs)

# Baseline
feat, names = get_baseline_features(outputs, pad_token_id=get_pad_token_id(path))

# Markovian
feat_m, names_m = get_markovian_features(outputs, k_tokens=20)

# AR(1)
ar1_models = fit_ar1_models(outputs, train_positions, train_labels)
feat_ar1, names_ar1 = get_ar1_features(outputs, test_positions, effective_mask, ar1_models)
```
