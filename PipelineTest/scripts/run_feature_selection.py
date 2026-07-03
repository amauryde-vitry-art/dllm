"""
scripts/run_feature_selection.py
================================
Centralized feature selection script.

Computes all features, performs:
- Correlation heatmap with labels
- PCA analysis (scree plot, cumulative variance)
- Univariate ROC-AUC per feature
- L1-regularized feature ranking
- Optional: LightGBM/XGBoost feature importance

Usage:
    python -m PipelineTest.scripts.run_feature_selection [--config CONFIG]
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from PipelineTest.features.utils import load_outputs, match_samples, get_pad_token_id
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features


# =========================================================================
# CONFIG
# =========================================================================

CONFIGS = {
    "llada": {
        "outputs_path": "PipelineTest/res/LLADA_64steps_64tokens_lowconf/outputs_LLADA_64steps_64tokens_lowconf.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLADA_64steps_64tokens_lowconf.json",
        "name": "LLaDA_64steps_64tokens",
    },
    "dream": {
        "outputs_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json",
        "name": "DREAM_64steps_64tokens",
    },
}


# =========================================================================
# FEATURE SELECTION UTILITIES
# =========================================================================

def univariate_roc_auc(X, y, feature_names):
    """Compute per-feature ROC-AUC (feature as score, label as target)."""
    results = {}
    for i, name in enumerate(feature_names):
        try:
            auc = roc_auc_score(y, X[:, i])
            results[name] = auc
        except ValueError:
            results[name] = 0.5
    return results


def plot_correlation_heatmap(X, y, feature_names, save_path):
    """Plot correlation heatmap of features + label."""
    data = np.column_stack([X, y.reshape(-1, 1)])
    all_names = feature_names + ["label"]
    corr = np.corrcoef(data, rowvar=False)

    plt.figure(figsize=(max(12, len(all_names) * 0.6), max(10, len(all_names) * 0.5)))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                xticklabels=all_names, yticklabels=all_names,
                annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_pca_scree(X, feature_names, save_path):
    """PCA scree plot + cumulative variance."""
    pca = PCA()
    pca.fit(X)
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(var_explained) + 1), var_explained, alpha=0.5, color="g", label="Individual")
    ax.plot(range(1, len(cum_var) + 1), cum_var, "o--", color="b", label="Cumulative")
    ax.axhline(y=0.95, color="r", linestyle=":", label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")

    n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    print(f"    Components for 95% variance: {n_95}/{len(var_explained)}")
    return pca


def plot_univariate_ranking(auc_dict, save_path):
    """Bar plot of per-feature ROC-AUC ranked."""
    sorted_items = sorted(auc_dict.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
    names = [item[0] for item in sorted_items]
    aucs = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    colors = ["green" if a > 0.5 else "red" for a in aucs]
    ax.barh(range(len(names)), aucs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(x=0.5, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("ROC-AUC (feature as score)")
    ax.set_title("Univariate Feature Discriminability")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def l1_feature_selection(X, y, feature_names):
    """L1-regularized logistic regression for feature ranking."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        penalty="l1", solver="saga", cv=5, scoring="roc_auc",
        max_iter=2000, random_state=42, Cs=20
    )
    model.fit(X_scaled, y)

    coefs = model.coef_[0]
    ranking = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print(f"    L1 LogReg (C={model.C_[0]:.4f}):")
    for name, coef in ranking:
        marker = "***" if abs(coef) > 0.1 else "  *" if abs(coef) > 0.01 else "   "
        print(f"      {marker} {name:35s}: {coef:+.4f}")

    return {name: float(coef) for name, coef in ranking}


# =========================================================================
# MAIN
# =========================================================================

def main(config_name="llada"):
    cfg = CONFIGS[config_name]
    output_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]

    print(f"\n{'='*70}")
    print(f"  FEATURE SELECTION: {cfg['name']}")
    print(f"{'='*70}")

    # Load
    print("\n[1] Loading...")
    outputs = load_outputs(output_path)
    positions, labels, data = match_samples(eval_json, outputs)
    print(f"    Samples: {len(positions)} (correct={np.sum(labels==0)}, halluc={np.sum(labels==1)})")

    try:
        pad_token_id = get_pad_token_id(output_path)
    except Exception:
        pad_token_id = None

    # Compute features
    print("\n[2] Computing features...")
    feat_baseline, names_baseline = get_baseline_features(outputs, pad_token_id=pad_token_id)
    feat_baseline = feat_baseline[positions]

    feat_markov, names_markov = get_markovian_features(outputs, k_tokens=20)
    feat_markov = feat_markov[positions]

    # Combine all
    X = np.column_stack([feat_baseline, feat_markov])
    feature_names = names_baseline + names_markov

    # Remove NaN columns
    valid_cols = ~np.any(np.isnan(X), axis=0)
    X = X[:, valid_cols]
    feature_names = [n for n, v in zip(feature_names, valid_cols) if v]
    print(f"    Total features (after NaN removal): {X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save dir
    save_dir = os.path.join(os.path.dirname(output_path), "feature_selection")
    os.makedirs(save_dir, exist_ok=True)

    # Correlation heatmap
    print("\n[3] Correlation heatmap...")
    plot_correlation_heatmap(X_scaled, labels, feature_names,
                            os.path.join(save_dir, "correlation_heatmap.png"))

    # PCA
    print("\n[4] PCA analysis...")
    plot_pca_scree(X_scaled, feature_names,
                   os.path.join(save_dir, "pca_scree.png"))

    # Univariate ranking
    print("\n[5] Univariate ROC-AUC ranking...")
    auc_dict = univariate_roc_auc(X_scaled, labels, feature_names)
    plot_univariate_ranking(auc_dict, os.path.join(save_dir, "univariate_ranking.png"))

    print("\n    Top-10 features by |AUC - 0.5|:")
    sorted_auc = sorted(auc_dict.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
    for name, auc in sorted_auc[:10]:
        print(f"      {name:35s}: AUC={auc:.4f}")

    # L1 selection
    print("\n[6] L1 feature selection...")
    l1_coefs = l1_feature_selection(X_scaled, labels, feature_names)

    # Save results
    results = {
        "config": cfg,
        "feature_names": feature_names,
        "univariate_auc": auc_dict,
        "l1_coefficients": l1_coefs,
    }
    out_path = os.path.join(save_dir, "feature_selection_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n    Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection analysis")
    parser.add_argument("--config", type=str, default="llada", choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    main(args.config)
