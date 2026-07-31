"""
scripts/feature_selection.py
=============================
Pipeline clean de feature selection + régression logistique.

Pour chaque configuration de run_evaluation.CONFIGS :
  1. Compute baseline + markovian features
  2. Split train / test (indépendant)
  3. Sélection top-K features sur le train uniquement
  4. GridSearch logistic regression sur le train (avec les K features)
  5. Évaluation sur le test (jamais touché pendant la sélection)

Usage:
    python -m PipelineTest.scripts.feature_selection [--config CONFIG] [--all] [--top-k 5]
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from PipelineTest.features.utils import load_outputs, match_samples, get_pad_token_id
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features
from PipelineTest.scripts.run_evaluation import CONFIGS


# =========================================================================
# PIPELINE: SELECT TOP-K → GRIDSEARCH LOGREG → EVAL ON TEST
# =========================================================================

def select_top_k_logreg(X, y, feature_names, k=5, method="l1", test_size=0.15):
    """Pipeline complet :
      1) Split train / test
      2) Sélection top-K sur le train
      3) GridSearch LogReg sur le train (K features)
      4) Éval sur le test (K features)

    Args:
        X: (N, D) feature matrix
        y: (N,) labels
        feature_names: list de D noms
        k: nombre de features à garder
        method: "l1" (ranking par coeff L1) ou "auc" (ranking par AUC univariée)
        test_size: fraction pour le test indépendant

    Returns:
        dict avec features sélectionnées, coefficients, métriques train/test
    """
    # ------------------------------------------------------------------
    # STEP 1: Split indépendant
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # ------------------------------------------------------------------
    # STEP 2: Sélection top-K sur le TRAIN uniquement
    # ------------------------------------------------------------------
    if method == "l1":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        l1_model = LogisticRegressionCV(
            Cs=20, penalty="l1", solver="saga", cv=5, scoring="roc_auc",
            n_jobs=-1, random_state=42, max_iter=10000,
        )
        l1_model.fit(X_train_scaled, y_train)
        coefs_importance = np.abs(l1_model.coef_[0])
        ranking = np.argsort(coefs_importance)[::-1]

    elif method == "auc":
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X)
        auc_scores = np.array([
            roc_auc_score(y, X_all_scaled[:, i])
            if len(np.unique(y)) > 1 else 0.5
            for i in range(X_all_scaled.shape[1])
        ])
        ranking = np.argsort(np.abs(auc_scores - 0.5))[::-1]

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    top_k_indices = sorted(ranking[:k])
    top_k_names = [feature_names[i] for i in top_k_indices]

    print(f"    Top-{k} features sélectionnées ({method}): {top_k_names}")

    # ------------------------------------------------------------------
    # STEP 3: GridSearch LogReg sur le TRAIN (K features)
    # ------------------------------------------------------------------
    X_train_k = X_train[:, top_k_indices]
    X_test_k = X_test[:, top_k_indices]

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
    grid.fit(X_train_k, y_train)

    best = grid.best_estimator_
    logreg = best.named_steps["logreg"]
    coefs_final = logreg.coef_[0]

    # ------------------------------------------------------------------
    # STEP 4: Évaluation sur le TEST indépendant
    # ------------------------------------------------------------------
    y_scores_train = best.predict_proba(X_train_k)[:, 1]
    y_scores_test = best.predict_proba(X_test_k)[:, 1]
    y_pred_test = (y_scores_test >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    coef_details = [
        {"feature": name, "coefficient": float(c)}
        for name, c in zip(top_k_names, coefs_final)
    ]

    return {
        "k": k,
        "method": method,
        "selected_indices": [int(i) for i in top_k_indices],
        "selected_features": top_k_names,
        "best_params": {key: val for key, val in grid.best_params_.items()},
        "best_cv_roc_auc": float(grid.best_score_),
        "coefficients": coef_details,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "metrics": {
            "train_roc_auc": float(roc_auc_score(y_train, y_scores_train)),
            "train_pr_auc": float(average_precision_score(y_train, y_scores_train)),
            "test_roc_auc": float(roc_auc_score(y_test, y_scores_test)),
            "test_pr_auc": float(average_precision_score(y_test, y_scores_test)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "n_train": len(X_train_k),
            "n_test": len(X_test_k),
        },
    }


# =========================================================================
# MAIN
# =========================================================================

def run_feature_selection(config_name, top_k=5):
    cfg = CONFIGS[config_name]
    output_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]

    print(f"\n{'='*70}")
    print(f"  FEATURE SELECTION: {cfg['name']}")
    print(f"{'='*70}")

    # Load data
    print("\n[1] Loading outputs...")
    outputs = load_outputs(output_path)
    positions, labels, data = match_samples(eval_json, outputs)
    print(f"    Matched samples: {len(positions)} (correct={np.sum(labels==0)}, halluc={np.sum(labels==1)})")

    # Pad token
    try:
        pad_token_id = get_pad_token_id(output_path)
    except Exception:
        pad_token_id = None

    # Compute features
    print("\n[2] Computing baseline features...")
    feat_baseline, names_baseline = get_baseline_features(outputs, pad_token_id=pad_token_id)
    feat_baseline = feat_baseline[positions]

    print("[3] Computing markovian features...")
    feat_markov, names_markov = get_markovian_features(outputs, k_tokens=64)
    feat_markov = feat_markov[positions]

    feat_baseline = np.nan_to_num(feat_baseline, nan=0.0)
    feat_markov = np.nan_to_num(feat_markov, nan=0.0)

    all_names = names_baseline + names_markov
    X_all = np.column_stack([feat_baseline, feat_markov])

    # --- Top-K Baseline+Markovian par L1 ---
    print(f"\n[4] Top-{top_k} Baseline+Markovian (L1 ranking)")
    res_l1 = select_top_k_logreg(X_all, labels, all_names, k=top_k, method="l1")
    print(f"    Features: {res_l1['selected_features']}")
    print(f"    Best params: {res_l1['best_params']}")
    print(f"    Test ROC-AUC={res_l1['metrics']['test_roc_auc']:.4f}  "
          f"PR-AUC={res_l1['metrics']['test_pr_auc']:.4f}  "
          f"Acc={res_l1['metrics']['test_accuracy']:.4f}")
    print(f"    Confusion matrix: {res_l1['confusion_matrix']}")
    for c in res_l1["coefficients"]:
        print(f"      {c['feature']:35s}: {c['coefficient']:+.4f}")

    # --- Top-K Baseline+Markovian par AUC ---
    print(f"\n[5] Top-{top_k} Baseline+Markovian (AUC ranking)")
    res_auc = select_top_k_logreg(X_all, labels, all_names, k=top_k, method="auc")
    print(f"    Features: {res_auc['selected_features']}")
    print(f"    Best params: {res_auc['best_params']}")
    print(f"    Test ROC-AUC={res_auc['metrics']['test_roc_auc']:.4f}  "
          f"PR-AUC={res_auc['metrics']['test_pr_auc']:.4f}  "
          f"Acc={res_auc['metrics']['test_accuracy']:.4f}")
    print(f"    Confusion matrix: {res_auc['confusion_matrix']}")
    for c in res_auc["coefficients"]:
        print(f"      {c['feature']:35s}: {c['coefficient']:+.4f}")

    # --- Top-K Baseline only ---
    print(f"\n[6] Top-{min(top_k, len(names_baseline))} Baseline only (L1 ranking)")
    res_bl = select_top_k_logreg(feat_baseline, labels, names_baseline,
                                  k=min(top_k, len(names_baseline)), method="l1")
    print(f"    Features: {res_bl['selected_features']}")
    print(f"    Test ROC-AUC={res_bl['metrics']['test_roc_auc']:.4f}  "
          f"PR-AUC={res_bl['metrics']['test_pr_auc']:.4f}  "
          f"Acc={res_bl['metrics']['test_accuracy']:.4f}")

    # --- Top-K Markovian only ---
    print(f"\n[7] Top-{min(top_k, len(names_markov))} Markovian only (L1 ranking)")
    res_mk = select_top_k_logreg(feat_markov, labels, names_markov,
                                  k=min(top_k, len(names_markov)), method="l1")
    print(f"    Features: {res_mk['selected_features']}")
    print(f"    Test ROC-AUC={res_mk['metrics']['test_roc_auc']:.4f}  "
          f"PR-AUC={res_mk['metrics']['test_pr_auc']:.4f}  "
          f"Acc={res_mk['metrics']['test_accuracy']:.4f}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {cfg['name']}")
    print(f"{'='*70}")
    print(f"  {'Configuration':<48} {'ROC-AUC':>10} {'PR-AUC':>10} {'Acc':>10}")
    print(f"  {'-'*78}")
    for name, res in [
        (f"Top-{top_k} Base+Markov (L1)", res_l1),
        (f"Top-{top_k} Base+Markov (AUC)", res_auc),
        (f"Top-{res_bl['k']} Baseline only (L1)", res_bl),
        (f"Top-{res_mk['k']} Markovian only (L1)", res_mk),
    ]:
        m = res["metrics"]
        print(f"  {name:<48} {m['test_roc_auc']:>10.4f} {m['test_pr_auc']:>10.4f} {m['test_accuracy']:>10.4f}")

    # --- Save results ---
    save_dir = os.path.join(os.path.dirname(output_path), "feature_selection_results")
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "config": cfg,
        "top_k": top_k,
        "n_samples": len(positions),
        "n_correct": int(np.sum(labels == 0)),
        "n_halluc": int(np.sum(labels == 1)),
        "results": {
            f"top{top_k}_base_markov_l1": res_l1,
            f"top{top_k}_base_markov_auc": res_auc,
            f"top{res_bl['k']}_baseline_l1": res_bl,
            f"top{res_mk['k']}_markovian_l1": res_mk,
        },
    }

    out_path = os.path.join(save_dir, "feature_selection_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Feature selection pipeline")
    parser.add_argument("--config", type=str, default=None,
                        choices=list(CONFIGS.keys()),
                        help="Single config to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Run on all configs")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of features to keep (default: 5)")
    args = parser.parse_args()

    if args.all:
        for config_name in CONFIGS:
            run_feature_selection(config_name, top_k=args.top_k)
    elif args.config:
        run_feature_selection(args.config, top_k=args.top_k)
    else:
        first_key = list(CONFIGS.keys())[0]
        run_feature_selection(first_key, top_k=args.top_k)


if __name__ == "__main__":
    main()
