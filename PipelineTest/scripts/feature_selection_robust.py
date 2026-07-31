"""
scripts/feature_selection_robust.py
====================================
Robust feature selection protocol across ALL configs, models, and sampling setups.

Finds features that are universally discriminative (Baseline + Markovian, 29 total).

Phases:
  1. Collect univariate ROC-AUC per feature & compute correlation matrix across all configs
  2. Select features by intersection (top-K across configs)
  3. Validate selected subsets with LogReg on every config
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from PipelineTest.features.utils import load_outputs, match_samples, get_pad_token_id
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features
from PipelineTest.Benchmark.data_split import load_eval_data, get_train_test_split
from PipelineTest.scripts.run_evaluation import CONFIGS


# =========================================================================
# PHASE 1: UNIVARIATE AUC & CORRELATIONS ACROSS ALL CONFIGS
# =========================================================================

def compute_univariate_auc_all_configs(config_names):
    """
    For each config, compute univariate ROC-AUC for every feature and its data matrix
    to allow global cross-config correlation tracking.
    """
    results = {}

    for config_name in config_names:
        cfg = CONFIGS[config_name]
        print(f"\n  [{config_name}] Loading...")

        outputs = load_outputs(cfg["outputs_path"])
        labels_raw, indices_raw, n_missing, _ = load_eval_data(cfg["eval_json"])

        sample_indices = outputs.sample_indices.numpy()
        idx_to_pos = {int(idx): pos for pos, idx in enumerate(sample_indices)}
        tensor_positions = [idx_to_pos[int(idx)] for idx in indices_raw]
        labels = labels_raw

        try:
            pad_token_id = get_pad_token_id(cfg["outputs_path"])
        except Exception:
            pad_token_id = None

        feat_baseline, names_baseline = get_baseline_features(outputs, pad_token_id=pad_token_id)
        feat_baseline = feat_baseline[tensor_positions]

        feat_markov, names_markov = get_markovian_features(outputs, k_tokens=64)
        feat_markov = feat_markov[tensor_positions]

        all_names = names_baseline + names_markov
        X = np.column_stack([feat_baseline, feat_markov])
        X = np.nan_to_num(X, nan=0.0)

        # Calcul des corrélations locales (Pearson) pour cette config
        # np.corrcoef attend une matrice (N_features, N_samples)
        corr_matrix = np.corrcoef(X, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        auc_scores = {}
        for j, fname in enumerate(all_names):
            try:
                auc = roc_auc_score(labels, X[:, j])
            except ValueError:
                auc = 0.5
            auc_scores[fname] = float(auc)

        n_correct = int(np.sum(labels == 0))
        n_halluc = int(np.sum(labels == 1))
        print(f"    Samples: {len(labels)} (correct={n_correct}, halluc={n_halluc}) | Features: {len(all_names)}")

        results[config_name] = {
            "feature_names": all_names,
            "auc_scores": auc_scores,
            "corr_matrix": corr_matrix.tolist(),  # Sauvegardé pour export JSON
            "n_samples": len(labels),
            "n_correct": n_correct,
            "n_halluc": n_halluc,
        }

    return results


# =========================================================================
# PHASE 2: INTERSECTION SELECTION
# =========================================================================

def compute_intersection(results, top_k, threshold=0.8):
    all_feature_names = None
    for config_name, data in results.items():
        all_feature_names = data["feature_names"]
        break

    n_configs = len(results)
    threshold_count = int(np.ceil(threshold * n_configs))

    feature_frequency = {fname: 0 for fname in all_feature_names}
    rankings = {}

    for config_name, data in results.items():
        auc_scores = data["auc_scores"]
        ranked = sorted(all_feature_names, key=lambda f: abs(auc_scores.get(f, 0.5) - 0.5), reverse=True)
        rankings[config_name] = ranked

        top_k_set = set(ranked[:top_k])
        for fname in top_k_set:
            feature_frequency[fname] += 1

    selected_features = [f for f, count in feature_frequency.items() if count >= threshold_count]
    selected_features.sort(key=lambda f: feature_frequency[f], reverse=True)

    return selected_features, feature_frequency, rankings


def find_best_k(results, k_range=range(5, 20), threshold=0.8):
    all_results_by_k = {}
    for k in k_range:
        selected, freq, _ = compute_intersection(results, top_k=k, threshold=threshold)
        all_results_by_k[k] = (selected, freq)
        if selected:
            print(f"    K={k}: {len(selected)} features selected -> {selected}")
        else:
            print(f"    K={k}: no intersection")

    valid_ks = [k for k, (sel, _) in all_results_by_k.items() if sel]
    best_k = min(valid_ks) if valid_ks else None
    return best_k, all_results_by_k


# =========================================================================
# PHASE 3: VALIDATION WITH LOGREG ON EVERY CONFIG
# =========================================================================

def validate_subset_on_config(cfg, feature_names_to_use):
    outputs = load_outputs(cfg["outputs_path"])
    labels_raw, indices_raw, n_missing, _ = load_eval_data(cfg["eval_json"])

    sample_indices = outputs.sample_indices.numpy()
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(sample_indices)}
    tensor_positions = [idx_to_pos[int(idx)] for idx in indices_raw]
    labels = labels_raw

    try:
        pad_token_id = get_pad_token_id(cfg["outputs_path"])
    except Exception:
        pad_token_id = None

    feat_baseline, names_baseline = get_baseline_features(outputs, pad_token_id=pad_token_id)
    feat_baseline = feat_baseline[tensor_positions]

    feat_markov, names_markov = get_markovian_features(outputs, k_tokens=64)
    feat_markov = feat_markov[tensor_positions]

    all_names = names_baseline + names_markov
    X = np.column_stack([feat_baseline, feat_markov])
    X = np.nan_to_num(X, nan=0.0)

    indices = [all_names.index(f) for f in feature_names_to_use if f in all_names]
    if not indices:
        return None

    X_sel = X[:, indices]
    names_sel = [all_names[i] for i in indices]

    train_idx, test_idx = get_train_test_split(len(labels))
    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42)),
    ])
    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logreg__penalty": ["l1", "l2"],
        "logreg__solver": ["saga"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_scores = best.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_scores)
    pr = average_precision_score(y_test, y_scores)
    acc = accuracy_score(y_test, y_pred)

    return {
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "test_accuracy": float(acc),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": names_sel,
    }


def validate_subsets(config_names, subsets):
    all_results = {}
    for label, features in subsets.items():
        print(f"\n{'='*70}")
        print(f"  VALIDATING: {label} ({len(features)} features)")
        print(f"{'='*70}")

        config_results = {}
        for config_name in config_names:
            cfg = CONFIGS[config_name]
            print(f"  [{config_name}]...")
            result = validate_subset_on_config(cfg, features)
            if result is not None:
                config_results[config_name] = result
                print(f"    ROC-AUC={result['test_roc_auc']:.4f}  PR-AUC={result['test_pr_auc']:.4f}")
        all_results[label] = config_results
    return all_results


# =========================================================================
# PLOTS (WITH NEW AVERAGE CORRELATION MATRIX HEATMAP)
# =========================================================================

def plot_average_correlation_matrix(results, save_dir):
    """Computes and saves the average Pearson correlation matrix across all configs."""
    config_names = list(results.keys())
    feature_names = results[config_names[0]]["feature_names"]
    n_feats = len(feature_names)

    # Accumulation des matrices de toutes les configs
    matrices = [np.array(results[cfg]["corr_matrix"]) for cfg in config_names]
    avg_matrix = np.mean(matrices, axis=0)

    fig, ax = plt.subplots(figsize=(max(12, n_feats * 0.45), max(10, n_feats * 0.4)))
    
    # Masque optionnel pour la diagonale supérieure pour alléger la lecture
    mask = np.triu(np.ones_like(avg_matrix, dtype=bool), k=1)
    
    sns.heatmap(avg_matrix, ax=ax, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0,
                xticklabels=feature_names, yticklabels=feature_names, mask=mask,
                annot=True, fmt=".2f", annot_kws={"size": 5}, cbar_kws={"shrink": 0.8})
    
    ax.set_title("Average Feature Correlation Matrix across Configs", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "average_feature_correlation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved Cross-Correlation Matrix: {path}")
    return path


def plot_univariate_auc_heatmap(results, save_dir):
    config_names = sorted(results.keys())
    feature_names = results[config_names[0]]["feature_names"]

    matrix = np.zeros((len(config_names), len(feature_names)))
    for i, cfg_name in enumerate(config_names):
        auc = results[cfg_name]["auc_scores"]
        for j, fname in enumerate(feature_names):
            matrix[i, j] = auc.get(fname, 0.5)

    fig, ax = plt.subplots(figsize=(max(14, len(feature_names) * 0.5), max(8, len(config_names) * 0.4)))
    sns.heatmap(matrix, ax=ax, cmap="RdYlGn", center=0.5, vmin=0.4, vmax=0.8,
                xticklabels=feature_names, yticklabels=config_names,
                annot=True, fmt=".2f", annot_kws={"size": 6})
    ax.set_title("Univariate ROC-AUC per Feature across Configs")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = os.path.join(save_dir, "univariate_auc_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_by_subset_size(validation_results, save_dir):
    labels, means, stds, mins = [], [], [], []
    for label, config_results in validation_results.items():
        aucs = [r["test_roc_auc"] for r in config_results.values()]
        if aucs:
            labels.append(label)
            means.append(np.mean(aucs))
            stds.append(np.std(aucs))
            mins.append(np.min(aucs))

    if not labels: return None
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    ax.bar(x, means, yerr=stds, capsize=5, color="#3498db", alpha=0.8, edgecolor="k")
    ax.scatter(x, mins, color="#e74c3c", zorder=5, s=80, marker="D", label="Worst-case (min)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 0.85)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_by_subset_size.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_frequency(feature_frequency, save_dir):
    sorted_items = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    colors = ["#2ecc71" if c >= max(counts) * 0.8 else "#f39c12" if c >= max(counts) * 0.5 else "#e74c3c" for c in counts]
    ax.barh(range(len(names)), counts, color=colors, alpha=0.8, edgecolor="k")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("# configs where feature is in top-K")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def print_summary_table(validation_results):
    print(f"\n{'='*90}\n  SUMMARY TABLE\n{'='*90}")
    print(f"  {'Subset':<30} {'#Feat':>6} {'Mean AUC':>10} {'Std':>8} {'Min AUC':>10} {'Mean PR':>10}")
    print(f"  {'-'*84}")
    for label, config_results in validation_results.items():
        aucs = [r["test_roc_auc"] for r in config_results.values()]
        prs = [r["test_pr_auc"] for r in config_results.values()]
        n_feat = len(config_results[list(config_results.keys())[0]]["features"]) if config_results else 0
        if aucs:
            print(f"  {label:<30} {n_feat:>6} {np.mean(aucs):>10.4f} {np.std(aucs):>8.4f} {np.min(aucs):>10.4f} {np.mean(prs):>10.4f}")
    print(f"  {'-'*84}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust feature selection across all configs")
    parser.add_argument("--top-k", type=int, default=None, help="Fixed K for top-K selection")
    parser.add_argument("--auto-k", action="store_true", help="Auto-find best K")
    parser.add_argument("--intersection-threshold", type=float, default=0.8, help="Min fraction of configs")
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of configs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if not args.top_k and not args.auto_k:
        parser.error("Either --top-k or --auto-k is required")

    config_names = args.configs if args.configs else list(CONFIGS.keys())
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "Benchmark", "feature_selection_robust")
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, "plots")

    print(f"\n  Configs to process: {len(config_names)}")
    # ---- PHASE 1 ----
    results = compute_univariate_auc_all_configs(config_names)
    plot_average_correlation_matrix(results, save_dir)


    # ---- PHASE 2 ----
    if args.auto_k:
        best_k, all_results_by_k = find_best_k(results, threshold=args.intersection_threshold)
        if best_k is None: return
        selected_features, feature_frequency = all_results_by_k[best_k]
    else:
        best_k = args.top_k
        selected_features, feature_frequency, _ = compute_intersection(results, top_k=best_k, threshold=args.intersection_threshold)

    if not selected_features: return

    # ---- PHASE 3 & PLOTS ----
    subsets = {
        f"Selected ({len(selected_features)} feat)": selected_features,
        "All (29 feat)": results[config_names[0]]["feature_names"],
    }
    
    if best_k and best_k > 3:
        for intermediate_k in sorted(set([3, 5, best_k])):
            if intermediate_k == best_k: continue
            sel, _, _ = compute_intersection(results, top_k=intermediate_k, threshold=args.intersection_threshold)
            if sel and sel != selected_features:
                subsets[f"K={intermediate_k} ({len(sel)} feat)"] = sel

    validation_results = validate_subsets(config_names, subsets)

    # Génération de tous les plots (incluant la nouvelle matrice de corrélation)
    plot_feature_frequency(feature_frequency, save_dir)
    plot_univariate_auc_heatmap(results, save_dir)
    plot_roc_by_subset_size(validation_results, save_dir)

    print_summary_table(validation_results)


if __name__ == "__main__":
    main()