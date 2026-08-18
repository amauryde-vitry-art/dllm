# Feature Selection Pipeline

Feature selection pipeline for hallucination detection in discrete diffusion LLMs (LLaDa, DREAM).

## Quick start

```bash
# Run full pipeline on all 24 configs
python PipelineTest/FeatureSelection/run_all.py

# Analyze feature frequencies across configs
python PipelineTest/FeatureSelection/FeatureFrequency.py
```

## Pipeline

```
prefilter (|r| > 0.95)  →  selection (Stability / mRMR / Boruta)  →  evaluation (80/20 benchmark split)
```

1. **`prefilter.py`** — Remove near-duplicate features by Pearson correlation (threshold 0.95).
2. **`stability_selection.py`** — Primary method. 500 bootstrap rounds with randomized L1 penalty (Meinshausen & Bühlmann, 2010). Theoretical bound on E[false positives].
3. **`mrmr_selection.py`** — Ablation. Greedy max-relevance min-redundancy (Peng et al., 2005).
4. **`boruta_selection.py`** — Ablation. Shadow feature testing via Random Forest (Kursa & Rudnicki, 2010).
5. **`evaluate.py`** — Two evaluation protocols:
   - `benchmark_evaluate_with_selection`: 80/20 sequential split matching the baseline protocol. Feature selection on train only.
   - `nested_cv_evaluate_with_selection`: 5×5 nested CV with selection inside each fold.
6. **`FeatureFrequency.py`** — Aggregate selected features across all 24 configs, rank by frequency.
7. **`configs.py`** — Experiment configurations (24 configs: 2 models × 4 step/token combos × 3 QA datasets).
8. **`run_all.py`** — Orchestrates the full pipeline.

## Outputs

Results are saved in `PipelineTest/res/FeatureSelection/`:
- `results/` — JSON files with selection results and evaluation metrics per config.
- `figures/` — Stability selection probability bar charts and mRMR score plots.

## Notes

See [notes.md](notes.md) for theoretical details on Stability Selection and the pipeline design choices.
