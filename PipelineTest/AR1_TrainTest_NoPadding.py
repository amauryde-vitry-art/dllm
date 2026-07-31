"""
AR1 Train/Test Pipeline - NO PADDING TOKENS
=============================================
Same as AR1_TrainTest.py but excludes padding tokens from the AR1 model.

Definition of padding token: for a given sample, token d is "padding" if the model
proposes the pad token (from tokenizer) at that position for ALL steps where it is
still masked (in histories_x0). This means the model always proposed padding there.

These tokens are excluded from:
- AR1 model fitting (linregress uses only non-padding tokens)
- Error computation (masked mean/var only on non-padding masked tokens)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.stats import linregress
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask
from PipelineTest.AnalyseResults import mergeOutputsList
from PipelineTest.CreateMetrics import getTestSimpleFeaturesNoPadding

# =========================================================================
# 1. LOAD DATA + INDEX MAPPING
# =========================================================================
# OUTPUTS_PATH = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt'
# EVAL_JSON = 'PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json'

OUTPUTS_PATH = 'PipelineTest/res/LLADA_64steps_64tokens_lowconf/outputs_LLADA_64steps_64tokens_lowconf.pt'
EVAL_JSON = 'PipelineTest/res/eval/results_triviaqa_LLADA_64steps_64tokens_lowconf.json'


TOKENIZER_PATH = 'PipelineTest/res/LLADA_64steps_64tokens_lowconf/tokenizer_LLADA_64steps_64tokens_lowconf.pt'

print("[1] Loading data...")
with open(EVAL_JSON, encoding="utf-8") as f:
    data = json.load(f)

outputs = mergeOutputsList(OUTPUTS_PATH)

hmap = {}
for i in range(len(outputs.sample_indices)):
    j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
    hmap[j] = i

entropies = getEntropy(outputs)
masks = getEachStepMask(outputs)

samples = []
for d in data:
    idx = d["index"]
    if idx in hmap:
        pos = hmap[idx]
        label = 0 if d["is_hallucination"] == 'no' else 1
        samples.append((pos, label))

positions = np.array([s[0] for s in samples])
labels = np.array([s[1] for s in samples])
print(f"    Total matched samples: {len(samples)} (correct={np.sum(labels==0)}, halluc={np.sum(labels==1)})")

# =========================================================================
# 2. DETECT PADDING TOKENS (using tokenizer + proposed_sequence)
# =========================================================================
print("[2] Detecting padding tokens (tokenizer-based)...")

tokenizer = torch.load(TOKENIZER_PATH, map_location="cpu", weights_only=False)
pad_token_id = tokenizer.pad_token_id
print(f"    Pad token: '{tokenizer.pad_token}' (id={pad_token_id})")


def compute_padding_mask_from_x0(outputs, positions, pad_token_id):
    """
    For each sample, identify padding tokens by checking the proposed_sequence (histories_x0).
    A token d is padding if the model proposes the pad token at that position
    at ALL steps where the token is still masked.
    
    Returns: padding_mask (N, T, D) where 1 = padding (to exclude), 0 = real token (to keep)
    """
    N = len(positions)
    T = len(outputs.histories_x0)
    D = outputs.max_new_tokens
    
    padding = np.zeros((N, D), dtype=bool)
    
    for idx_n, pos in enumerate(positions):
        start_idx = outputs.start_idx_history[pos]
        end_idx = start_idx + D
        
        for d in range(D):
            # Find steps where token d is masked
            masked_steps = []
            for t in range(T):
                if outputs.histories_mask[t][pos, start_idx + d].item() > 0:
                    masked_steps.append(t)
            
            if len(masked_steps) == 0:
                # Token is never masked → not padding (prompt token)
                continue
            
            # Check if proposed token is pad_token_id at ALL masked steps
            all_padding = True
            for t in masked_steps:
                proposed_token = outputs.histories_x0[t][pos, start_idx + d].item()
                if proposed_token != pad_token_id:
                    all_padding = False
                    break
            
            if all_padding:
                padding[idx_n, d] = True
    
    # Expand to (N, T, D)
    padding_mask = np.broadcast_to(padding[:, np.newaxis, :], (N, T, D)).copy()
    
    return padding_mask


# Compute padding for all samples
padding_mask_all = compute_padding_mask_from_x0(outputs, positions, pad_token_id)

# Stats
n_padding_per_sample = padding_mask_all[:, 0, :].sum(axis=1)  # per sample
print(f"    Padding tokens per sample: mean={n_padding_per_sample.mean():.1f}, "
      f"min={n_padding_per_sample.min()}, max={n_padding_per_sample.max()}")

# =========================================================================
# 3. SPLIT: 50% for AR1 fitting, 50% for logistic regression
# =========================================================================
print("[3] Splitting data 50/50 (stratified)...")
ar1_idx, logreg_idx = train_test_split(
    np.arange(len(samples)), test_size=0.5, stratify=labels, random_state=42
)

ar1_positions = positions[ar1_idx]
ar1_labels = labels[ar1_idx]
logreg_positions = positions[logreg_idx]
logreg_labels = labels[logreg_idx]

ar1_correct_pos = ar1_positions[ar1_labels == 0]
ar1_halluc_pos = ar1_positions[ar1_labels == 1]

print(f"    AR1 fitting half: {len(ar1_idx)} (correct={len(ar1_correct_pos)}, halluc={len(ar1_halluc_pos)})")
print(f"    LogReg half:      {len(logreg_idx)} (correct={np.sum(logreg_labels==0)}, halluc={np.sum(logreg_labels==1)})")

# Build tensors for AR1 fitting (with padding info)
entropy_train_correct = np.array([entropies[i] for i in ar1_correct_pos])
entropy_train_halluc = np.array([entropies[i] for i in ar1_halluc_pos])

# Padding masks for AR1 fitting half
padding_ar1 = padding_mask_all[ar1_idx]
padding_ar1_correct = padding_ar1[ar1_labels == 0]
padding_ar1_halluc = padding_ar1[ar1_labels == 1]

# Build tensors for logistic regression half
entropy_logreg = np.array([entropies[i] for i in logreg_positions])
mask_logreg = np.array([masks[i] for i in logreg_positions], dtype=float)
padding_logreg = padding_mask_all[logreg_idx]

# Effective mask: masked AND not padding
# effective_mask[n, t, d] = 1 if token is masked AND not padding
effective_mask_logreg = mask_logreg * (1 - padding_logreg.astype(float))

N_lr, T_max, D = entropy_logreg.shape
print(f"    Tensor shapes: T={T_max}, D={D}")

# =========================================================================
# 4. FIT TWO AR1 MODELS ON TRAINING DATA (excluding padding)
# =========================================================================

def fit_ar1_model_no_padding(entropy_tensor, padding_tensor):
    """
    Fit AR(1) per-token model on population, excluding padding tokens.
    Only uses samples where the token is NOT padding for the regression.
    
    Input: entropy (N, T, D), padding (N, T, D) where 1=padding
    Output: phi (T, D), intercept (T, D), sigma (T, D)
    """
    N, T, D = entropy_tensor.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    sigma = np.zeros((T, D))
    
    for t in range(1, T):
        X_prev = entropy_tensor[:, t-1, :]
        X_curr = entropy_tensor[:, t, :]
        
        for d in range(D):
            # Only use samples where token d is NOT padding
            valid = ~padding_tensor[:, t, d].astype(bool)
            n_valid = np.sum(valid)
            
            if n_valid < 3:
                # Not enough data to fit
                continue
            
            x = X_prev[valid, d]
            y = X_curr[valid, d]
            
            slope, intcpt, _, _, _ = linregress(x, y)
            preds = slope * x + intcpt
            resid = y - preds
            
            phi[t, d] = slope
            intercept[t, d] = intcpt
            sigma[t, d] = np.std(resid)
    
    return phi, intercept, sigma


print("[4] Fitting AR1 model on CORRECT training data (no padding)...")
phi_correct, intercept_correct, sigma_correct = fit_ar1_model_no_padding(
    entropy_train_correct, padding_ar1_correct)
print("    Done.")

print("    Fitting AR1 model on HALLUCINATION training data (no padding)...")
phi_halluc, intercept_halluc, sigma_halluc = fit_ar1_model_no_padding(
    entropy_train_halluc, padding_ar1_halluc)
print("    Done.")

# =========================================================================
# 5. EVALUATE ON TEST SET: PER-SAMPLE ERRORS (excluding padding)
# =========================================================================

def compute_per_sample_masked_mean_error(entropy_test, effective_mask, phi, intercept):
    """
    Per-sample mean absolute error on masked non-padding tokens.
    effective_mask = mask * (1 - padding)
    """
    N, T, D = entropy_test.shape
    sample_errors = np.zeros(N)
    
    for n in range(N):
        total_error = 0.0
        total_count = 0
        
        for t in range(1, T):
            x_prev = entropy_test[n, t-1, :]
            x_real = entropy_test[n, t, :]
            mask_t = effective_mask[n, t, :]
            
            x_pred = phi[t, :] * x_prev + intercept[t, :]
            error = np.abs(x_real - x_pred) * mask_t
            # total_error += np.sum(error)
            total_error += np.sum(x_pred * mask_t)  
            total_count += np.sum(mask_t)
        
        sample_errors[n] = total_error / max(total_count, 1)
    
    return sample_errors


def compute_per_sample_masked_var_error(entropy_test, effective_mask, phi, intercept):
    """
    Per-sample variance error on masked non-padding tokens.
    """
    N, T, D = entropy_test.shape
    sample_var_errors = np.zeros(N)
    
    for n in range(N):
        total_var_error = 0.0
        count_steps = 0
        
        for t in range(1, T):
            x_prev = entropy_test[n, t-1, :]
            x_real = entropy_test[n, t, :]
            mask_t = effective_mask[n, t, :]
            
            n_masked = int(np.sum(mask_t))
            if n_masked < 2:
                continue
            
            real_vals = x_real[mask_t > 0]
            var_real = np.var(real_vals)
            
            x_pred = phi[t, :] * x_prev + intercept[t, :]
            pred_vals = x_pred[mask_t > 0]
            var_pred = np.var(pred_vals)
            
            # total_var_error += np.abs(var_real - var_pred)
            total_var_error += var_pred
            count_steps += 1
        
        sample_var_errors[n] = total_var_error / max(count_steps, 1)
    
    return sample_var_errors


print("[5] Computing per-sample errors on LogReg half (no padding)...")
print("    Mean entropy error vs correct model...")
error_vs_correct = compute_per_sample_masked_mean_error(
    entropy_logreg, effective_mask_logreg, phi_correct, intercept_correct
)
print("    Mean entropy error vs hallucination model...")
error_vs_halluc = compute_per_sample_masked_mean_error(
    entropy_logreg, effective_mask_logreg, phi_halluc, intercept_halluc
)

print("    Variance error vs correct model...")
var_error_vs_correct = compute_per_sample_masked_var_error(
    entropy_logreg, effective_mask_logreg, phi_correct, intercept_correct
)
print("    Variance error vs hallucination model...")
var_error_vs_halluc = compute_per_sample_masked_var_error(
    entropy_logreg, effective_mask_logreg, phi_halluc, intercept_halluc
)

# =========================================================================
# 6. LOGISTIC REGRESSION WITH CROSS-VALIDATION + GRID SEARCH
# =========================================================================
print("[6] Logistic Regression with GridSearchCV...")
features_benchmark, feature_names_benchmark, _ = getTestSimpleFeaturesNoPadding(OUTPUTS_PATH, EVAL_JSON)
features_benchmark_logreg = features_benchmark[logreg_positions]

X_ar1 = np.column_stack([
    error_vs_correct,
    error_vs_halluc,
    var_error_vs_correct,
    var_error_vs_halluc,
])
ar1_feature_names = [
    'err_vs_correct', 
    'err_vs_halluc',
    'var_err_vs_correct', 
    'var_err_vs_halluc',
]
X_benchmark = features_benchmark_logreg
y_labels = logreg_labels

print(f"    AR1 feature matrix: {X_ar1.shape}")
print(f"    Benchmark feature matrix: {X_benchmark.shape} ({feature_names_benchmark})")
print(f"    Labels: {y_labels.shape} (correct={np.sum(y_labels==0)}, halluc={np.sum(y_labels==1)})")


def run_logreg_pipeline(X, y, feature_names_list, name):
    """Run GridSearchCV logistic regression, return results dict."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid = {
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__solver': ['saga'],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    [{name}] train: {len(X_train)}, test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"    [{name}] Best params: {grid.best_params_}")
    print(f"    [{name}] Best CV ROC-AUC (on train): {grid.best_score_:.4f}")

    best = grid.best_estimator_
    y_scores = best.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)
    pr = average_precision_score(y_test, y_scores)

    thresholds = np.linspace(0, 1, 201)
    accs = [accuracy_score(y_test, (y_scores >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    pos_rate_test = float(np.mean(y_test))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"    [{name}] Test: accuracy={acc:.4f}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  (pos_rate={pos_rate_test:.4f})")
    print(f"    [{name}] Best threshold accuracy: {best_acc:.4f} (threshold={best_thresh:.3f})")
    print(f"    [{name}] Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    coefs = best.named_steps['logreg'].coef_[0]
    print(f"    [{name}] Coefficients:")
    for fname, coef in zip(feature_names_list, coefs):
        print(f"      {fname:30s}: {coef:+.4f}")

    return {
        "name": name,
        "features": feature_names_list,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "best_cv_roc_auc": float(grid.best_score_),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(best_acc),
        "test_best_threshold": float(best_thresh),
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "test_pos_rate": pos_rate_test,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "coefficients": {fname: float(c) for fname, c in zip(feature_names_list, coefs)},
    }


print("\n  === AR1 Features (no padding) ===")
results_ar1 = run_logreg_pipeline(X_ar1, y_labels, ar1_feature_names, "AR1_NoPad")

print("\n  === Benchmark Features (no padding) ===")
results_benchmark = run_logreg_pipeline(X_benchmark, y_labels, feature_names_benchmark, "Benchmark_NoPad")

X_combined = np.column_stack([X_ar1, X_benchmark])
combined_feature_names = ar1_feature_names + feature_names_benchmark
print("\n  === Combined (AR1_NoPad + Benchmark_NoPad) Features ===")
results_combined = run_logreg_pipeline(X_combined, y_labels, combined_feature_names, "Combined_NoPad")

# --- Save results to JSON ---
save_dir = os.path.join(os.path.dirname(OUTPUTS_PATH), "AnalyseResults")
os.makedirs(save_dir, exist_ok=True)
results_all = {"AR1_NoPad": results_ar1, "Benchmark_NoPad": results_benchmark, "Combined_NoPad": results_combined}
results_json_path = os.path.join(save_dir, "AR1_NoPadding_vs_Benchmark_LogReg_results.json")
with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2, ensure_ascii=False)
print(f"\n    Results saved to: {results_json_path}")

# =========================================================================
# 7. SUMMARY
# =========================================================================
test_correct_mask = logreg_labels == 0
test_halluc_mask = logreg_labels == 1

print("\n" + "="*70)
print("SUMMARY STATISTICS (NO PADDING)")
print("="*70)
print(f"\nPadding detection: tokenizer pad_token_id={pad_token_id} in proposed_sequence (histories_x0)")
print(f"Padding tokens per sample: mean={n_padding_per_sample.mean():.1f}, "
      f"min={n_padding_per_sample.min()}, max={n_padding_per_sample.max()}")

print(f"\nMean Entropy Error vs CORRECT model:")
print(f"  True Correct:      {error_vs_correct[test_correct_mask].mean():.4f} ± {error_vs_correct[test_correct_mask].std():.4f}")
print(f"  True Hallucination:{error_vs_correct[test_halluc_mask].mean():.4f} ± {error_vs_correct[test_halluc_mask].std():.4f}")

print(f"\nMean Entropy Error vs HALLUCINATION model:")
print(f"  True Correct:      {error_vs_halluc[test_correct_mask].mean():.4f} ± {error_vs_halluc[test_correct_mask].std():.4f}")
print(f"  True Hallucination:{error_vs_halluc[test_halluc_mask].mean():.4f} ± {error_vs_halluc[test_halluc_mask].std():.4f}")

print(f"\nVariance Error vs CORRECT model:")
print(f"  True Correct:      {var_error_vs_correct[test_correct_mask].mean():.4f} ± {var_error_vs_correct[test_correct_mask].std():.4f}")
print(f"  True Hallucination:{var_error_vs_correct[test_halluc_mask].mean():.4f} ± {var_error_vs_correct[test_halluc_mask].std():.4f}")

print(f"\nVariance Error vs HALLUCINATION model:")
print(f"  True Correct:      {var_error_vs_halluc[test_correct_mask].mean():.4f} ± {var_error_vs_halluc[test_correct_mask].std():.4f}")
print(f"  True Hallucination:{var_error_vs_halluc[test_halluc_mask].mean():.4f} ± {var_error_vs_halluc[test_halluc_mask].std():.4f}")

# Simple baseline
print("\n" + "-"*70)
print("SIMPLE CLASSIFIER (baseline)")
print("-"*70)

score_entropy = error_vs_correct - error_vs_halluc
score_var = var_error_vs_correct - var_error_vs_halluc
score_combined = (error_vs_correct + var_error_vs_correct) - (error_vs_halluc + var_error_vs_halluc)

pred_entropy = (score_entropy > 0).astype(int)
acc_entropy = np.mean(pred_entropy == logreg_labels)
roc_entropy = roc_auc_score(logreg_labels, score_entropy)
pr_entropy = average_precision_score(logreg_labels, score_entropy)
print(f"  Entropy-based:  accuracy={acc_entropy:.4f}  ROC-AUC={roc_entropy:.4f}  PR-AUC={pr_entropy:.4f}")

pred_var = (score_var > 0).astype(int)
acc_var = np.mean(pred_var == logreg_labels)
roc_var = roc_auc_score(logreg_labels, score_var)
pr_var = average_precision_score(logreg_labels, score_var)
print(f"  Variance-based: accuracy={acc_var:.4f}  ROC-AUC={roc_var:.4f}  PR-AUC={pr_var:.4f}")

pred_combined = (score_combined > 0).astype(int)
acc_combined = np.mean(pred_combined == logreg_labels)
roc_combined = roc_auc_score(logreg_labels, score_combined)
pr_combined = average_precision_score(logreg_labels, score_combined)
print(f"  Combined:       accuracy={acc_combined:.4f}  ROC-AUC={roc_combined:.4f}  PR-AUC={pr_combined:.4f}")

print("\n" + "-"*70)
print("LOGISTIC REGRESSION (5-fold CV + GridSearch, test on 20%)")
print("-"*70)
print(f"  {'Method':<16} {'Accuracy':>10} {'ROC-AUC':>10} {'PR-AUC':>10} {'Pos Rate':>10}")
print(f"  {'AR1_NoPad':<16} {results_ar1['test_accuracy']:>10.4f} {results_ar1['test_roc_auc']:>10.4f} {results_ar1['test_pr_auc']:>10.4f} {results_ar1['test_pos_rate']:>10.4f}")
print(f"  {'Benchmark_NoPad':<16} {results_benchmark['test_accuracy']:>10.4f} {results_benchmark['test_roc_auc']:>10.4f} {results_benchmark['test_pr_auc']:>10.4f} {results_benchmark['test_pos_rate']:>10.4f}")
print(f"  {'Combined_NoPad':<16} {results_combined['test_accuracy']:>10.4f} {results_combined['test_roc_auc']:>10.4f} {results_combined['test_pr_auc']:>10.4f} {results_combined['test_pos_rate']:>10.4f}")

# =========================================================================
# 8. PLOTS: Error distributions
# =========================================================================
print("\n[8] Generating error distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
ax.hist(error_vs_correct[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(error_vs_correct[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Mean Masked Entropy Error vs CORRECT Model (no padding)", fontweight='bold')
ax.set_xlabel("Mean Absolute Error")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

ax = axes[0, 1]
ax.hist(error_vs_halluc[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(error_vs_halluc[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Mean Masked Entropy Error vs HALLUCINATION Model (no padding)", fontweight='bold')
ax.set_xlabel("Mean Absolute Error")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

ax = axes[1, 0]
ax.hist(var_error_vs_correct[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(var_error_vs_correct[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Variance Error vs CORRECT Model (no padding)", fontweight='bold')
ax.set_xlabel("Mean |Var_real - Var_pred|")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

ax = axes[1, 1]
ax.hist(var_error_vs_halluc[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(var_error_vs_halluc[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Variance Error vs HALLUCINATION Model (no padding)", fontweight='bold')
ax.set_xlabel("Mean |Var_real - Var_pred|")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

plt.suptitle("AR1 NoPadding: Per-Sample Error Distributions", fontsize=14, fontweight='bold')
plt.tight_layout()
save_path_dist = os.path.join(save_dir, "AR1_NoPadding_ErrorDistributions.png")
plt.savefig(save_path_dist, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_dist}")

# --- Benchmark feature distribution (one subplot per feature) ---
n_bench_features = len(feature_names_benchmark)
n_cols_bench = min(n_bench_features, 2)
n_rows_bench = (n_bench_features + n_cols_bench - 1) // n_cols_bench
fig, axes_bench = plt.subplots(n_rows_bench, n_cols_bench, figsize=(8 * n_cols_bench, 5 * n_rows_bench))
if n_bench_features == 1:
    axes_bench = np.array([axes_bench])
axes_bench = np.atleast_1d(axes_bench).flatten()
for i, fname in enumerate(feature_names_benchmark):
    ax = axes_bench[i]
    feat = features_benchmark_logreg[:, i]
    ax.hist(feat[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
    ax.hist(feat[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
    ax.set_title(f"{fname}", fontweight='bold')
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
for i in range(n_bench_features, len(axes_bench)):
    axes_bench[i].set_visible(False)
plt.suptitle("Benchmark Feature Distributions (no padding split)", fontsize=14, fontweight='bold')
plt.tight_layout()
save_path_bench = os.path.join(save_dir, "AR1_NoPadding_BenchmarkDistribution.png")
plt.savefig(save_path_bench, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_bench}")

# --- Benchmark scatter plot (first two features) ---
if n_bench_features >= 2:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(features_benchmark_logreg[test_correct_mask, 0], features_benchmark_logreg[test_correct_mask, 1],
               alpha=0.5, color='green', label='True Correct', s=20)
    ax.scatter(features_benchmark_logreg[test_halluc_mask, 0], features_benchmark_logreg[test_halluc_mask, 1],
               alpha=0.5, color='red', label='True Hallucination', s=20)
    ax.set_xlabel(feature_names_benchmark[0])
    ax.set_ylabel(feature_names_benchmark[1])
    ax.set_title(f"Benchmark Scatter: {feature_names_benchmark[0]} vs {feature_names_benchmark[1]}", fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_path_bench_scatter = os.path.join(save_dir, "AR1_NoPadding_BenchmarkScatter.png")
    plt.savefig(save_path_bench_scatter, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"    Saved: {save_path_bench_scatter}")

# --- Scatter plots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.scatter(error_vs_correct[test_correct_mask], var_error_vs_correct[test_correct_mask],
           alpha=0.5, color='green', label='True Correct', s=20)
ax.scatter(error_vs_correct[test_halluc_mask], var_error_vs_correct[test_halluc_mask],
           alpha=0.5, color='red', label='True Hallucination', s=20)
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("MeanCorrect")
ax.set_ylabel("VarCorrect")
ax.set_title("MeanCorrect VS VarCorrect", fontweight='bold')
ax.legend()
ax.grid(True)

ax = axes[1]
ax.scatter(error_vs_halluc[test_correct_mask], var_error_vs_halluc[test_correct_mask],
           alpha=0.5, color='green', label='True Correct', s=20)
ax.scatter(error_vs_halluc[test_halluc_mask], var_error_vs_halluc[test_halluc_mask],
           alpha=0.5, color='red', label='True Hallucination', s=20)
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("MeanHallucinated")
ax.set_ylabel("VarHallucinated")
ax.set_title("MeanHallucinated VS VarHallucinated", fontweight='bold')
ax.legend()
ax.grid(True)

plt.tight_layout()
save_path_scatter = os.path.join(save_dir, "AR1_NoPadding_ScatterErrors.png")
plt.savefig(save_path_scatter, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_scatter}")

print("\nDone!")
