"""
AR1 Train/Test Pipeline for Hallucination Detection
=====================================================
1. Load outputs + labels (with correct index mapping)
2. Split into train/test (stratified)
3. Fit two AR1 models on TRAIN: one for correct, one for hallucinations
4. On TEST: compute per-sample masked mean entropy error vs each model
5. On TEST: compute per-sample masked variance error vs each model
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.stats import linregress
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask
from PipelineTest.historic_scripts.AnalyseResults import mergeOutputsList
from PipelineTest.historic_scripts.CreateMetrics import getTestSimpleFeatures

# =========================================================================
# 1. LOAD DATA + INDEX MAPPING
# =========================================================================
# OUTPUTS_PATH = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt'
# EVAL_JSON = 'PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json'

OUTPUTS_PATH = 'PipelineTest/res/LLADA_64steps_64tokens_lowconf/outputs_LLADA_64steps_64tokens_lowconf.pt'
EVAL_JSON = 'PipelineTest/res/eval/results_triviaqa_LLADA_64steps_64tokens_lowconf.json'


print("[1] Loading data...")
with open(EVAL_JSON, encoding="utf-8") as f:
    data = json.load(f)

outputs = mergeOutputsList(OUTPUTS_PATH)

# IMPORTANT: entropy[i] corresponds to sample_indices[i], NOT to index i directly.
# Build mapping: sample_index -> position in outputs
hmap = {}
for i in range(len(outputs.sample_indices)):
    j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
    hmap[j] = i

# Get all entropies and masks (indexed by position in outputs, not sample_index)
entropies = getEntropy(outputs)
masks = getEachStepMask(outputs)

# Build list of (position_in_outputs, label) for all samples we can match
samples = []
for d in data:
    idx = d["index"]
    if idx in hmap:
        pos = hmap[idx]
        label = 0 if d["is_hallucination"] == 'no' else 1  # 0=correct, 1=hallucination
        samples.append((pos, label))

positions = np.array([s[0] for s in samples])
labels = np.array([s[1] for s in samples])
print(f"    Total matched samples: {len(samples)} (correct={np.sum(labels==0)}, halluc={np.sum(labels==1)})")

# =========================================================================
# 2. SPLIT: 50% for AR1 fitting, 50% for logistic regression
# =========================================================================
print("[2] Splitting data 50/50 (stratified)...")
ar1_idx, logreg_idx = train_test_split(
    np.arange(len(samples)), test_size=0.5, stratify=labels, random_state=42
)

ar1_positions = positions[ar1_idx]
ar1_labels = labels[ar1_idx]
logreg_positions = positions[logreg_idx]
logreg_labels = labels[logreg_idx]

# Separate AR1 half into correct/hallucination
ar1_correct_pos = ar1_positions[ar1_labels == 0]
ar1_halluc_pos = ar1_positions[ar1_labels == 1]

print(f"    AR1 fitting half: {len(ar1_idx)} (correct={len(ar1_correct_pos)}, halluc={len(ar1_halluc_pos)})")
print(f"    LogReg half:      {len(logreg_idx)} (correct={np.sum(logreg_labels==0)}, halluc={np.sum(logreg_labels==1)})")

# Build tensors for AR1 fitting
entropy_train_correct = np.array([entropies[i] for i in ar1_correct_pos])  # (N_c, T, D)
entropy_train_halluc = np.array([entropies[i] for i in ar1_halluc_pos])    # (N_h, T, D)

# Build tensors for logistic regression half
entropy_logreg = np.array([entropies[i] for i in logreg_positions])   # (N_lr, T, D)
mask_logreg = np.array([masks[i] for i in logreg_positions], dtype=float)  # (N_lr, T, D)

N_lr, T_max, D = entropy_logreg.shape
print(f"    Tensor shapes: T={T_max}, D={D}")

# =========================================================================
# 3. FIT TWO AR1 MODELS ON TRAINING DATA
# =========================================================================

def fit_ar1_model(entropy_tensor):
    """
    Fit AR(1) per-token model on population.
    Input: (N, T, D)
    Output: phi (T, D), intercept (T, D), sigma (T, D)
    
    Model: x_{t,d} = phi_{t,d} * x_{t-1,d} + intercept_{t,d} + noise
    """
    N, T, D = entropy_tensor.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    sigma = np.zeros((T, D))
    
    for t in range(1, T):
        X_prev = entropy_tensor[:, t-1, :]  # (N, D)
        X_curr = entropy_tensor[:, t, :]    # (N, D)
        
        for d in range(D):
            slope, intcpt, _, _, _ = linregress(X_prev[:, d], X_curr[:, d])
            preds = slope * X_prev[:, d] + intcpt
            resid = X_curr[:, d] - preds
            
            phi[t, d] = slope
            intercept[t, d] = intcpt
            sigma[t, d] = np.std(resid)
    
    return phi, intercept, sigma


print("[3] Fitting AR1 model on CORRECT training data...")
phi_correct, intercept_correct, sigma_correct = fit_ar1_model(entropy_train_correct)
print("    Done.")

print("    Fitting AR1 model on HALLUCINATION training data...")
phi_halluc, intercept_halluc, sigma_halluc = fit_ar1_model(entropy_train_halluc)
print("    Done.")

# =========================================================================
# 4. EVALUATE ON TEST SET: PER-SAMPLE ERRORS
# =========================================================================

def compute_deterministic_prediction(phi, intercept, x_prev, sigma= None):
    """
    Given phi (T, D), intercept (T, D), and x_prev (D,) at time t-1,
    predict x_t = phi_t * x_{t-1} + intercept_t
    """
    if sigma is not None: 
        return phi * x_prev + intercept + sigma * np.random.randn(*x_prev.shape)
    else:
        return phi * x_prev + intercept


def compute_per_sample_masked_mean_error(entropy_test, mask_test, phi, intercept):
    """
    For each test sample, compute mean absolute error on masked tokens
    between real entropy and AR1 deterministic prediction.
    
    Returns: (N_test,) - one scalar error per sample (averaged over all steps and masked tokens)
    """
    N, T, D = entropy_test.shape
    sample_errors = np.zeros(N)
    
    for n in range(N):
        total_error = 0.0
        total_count = 0
        
        for t in range(1, T):
            x_prev = entropy_test[n, t-1, :]  # (D,)
            x_real = entropy_test[n, t, :]     # (D,)
            mask_t = mask_test[n, t, :]        # (D,)
            
            # Deterministic AR1 prediction
            x_pred = phi[t, :] * x_prev + intercept[t, :]  # (D,)
            
            # Error only on masked tokens
            error = np.abs(x_real - x_pred) * mask_t
            total_error += np.sum(error)
            total_count += np.sum(mask_t)
        
        sample_errors[n] = total_error / max(total_count, 1)
    
    return sample_errors


def compute_per_sample_masked_var_error(entropy_test, mask_test, phi, intercept, sigma):
    """
    For each test sample, compare the variance across masked tokens
    to the theoretical AR1 variance (sigma^2).
    
    Returns: (N_test,) - one scalar per sample (mean over steps of |var_real - var_model|)
    """
    N, T, D = entropy_test.shape
    sample_var_errors = np.zeros(N)
    
    for n in range(N):
        total_var_error = 0.0
        count_steps = 0
        
        for t in range(1, T):
            x_prev = entropy_test[n, t-1, :]
            x_real = entropy_test[n, t, :]
            mask_t = mask_test[n, t, :]
            
            n_masked = int(np.sum(mask_t))
            if n_masked < 2:
                continue
            
            # Real variance on masked tokens
            real_vals = x_real[mask_t > 0]
            var_real = np.var(real_vals)
            
            # Predicted variance from AR1: residuals have std sigma
            # Theoretical variance of prediction errors on masked tokens
            x_pred = compute_deterministic_prediction(phi[t, :], intercept[t, :], x_prev, sigma=None)
            pred_vals = x_pred[mask_t > 0]
            var_pred = np.var(pred_vals)
            
            total_var_error += np.abs(var_real - var_pred)
            count_steps += 1
        
        sample_var_errors[n] = total_var_error / max(count_steps, 1)
    
    return sample_var_errors


print("[4] Computing per-sample errors on LogReg half...")
print("    Mean entropy error vs correct model...")
error_vs_correct = compute_per_sample_masked_mean_error(
    entropy_logreg, mask_logreg, phi_correct, intercept_correct
)
print("    Mean entropy error vs hallucination model...")
error_vs_halluc = compute_per_sample_masked_mean_error(
    entropy_logreg, mask_logreg, phi_halluc, intercept_halluc
)

print("    Variance error vs correct model...")
var_error_vs_correct = compute_per_sample_masked_var_error(
    entropy_logreg, mask_logreg, phi_correct, intercept_correct, sigma_correct
)
print("    Variance error vs hallucination model...")
var_error_vs_halluc = compute_per_sample_masked_var_error(
    entropy_logreg, mask_logreg, phi_halluc, intercept_halluc, sigma_halluc
)

# =========================================================================
# 5. LOGISTIC REGRESSION WITH CROSS-VALIDATION + GRID SEARCH
# =========================================================================
print("[5] Logistic Regression with GridSearchCV...")
features_benchmark, feature_names_benchmark, _ = getTestSimpleFeatures(OUTPUTS_PATH, EVAL_JSON)
# features_benchmark is indexed by position in outputs, so use logreg_positions (not logreg_idx)
features_benchmark_logreg = features_benchmark[logreg_positions]

# Build feature matrix: 8 AR1 features per sample
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

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    [{name}] train: {len(X_train)}, test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"    [{name}] Best params: {grid.best_params_}")
    print(f"    [{name}] Best CV ROC-AUC (on train): {grid.best_score_:.4f}")

    # Evaluate on held-out 20%
    best = grid.best_estimator_
    y_scores = best.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)
    pr = average_precision_score(y_test, y_scores)

    # Find threshold that maximizes accuracy
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


# --- Run on AR1 features ---
print("\n  === AR1 Features ===")
results_ar1 = run_logreg_pipeline(X_ar1, y_labels, ar1_feature_names, "AR1")

# --- Run on Benchmark features ---
print("\n  === Benchmark Features ===")
results_benchmark = run_logreg_pipeline(X_benchmark, y_labels, feature_names_benchmark, "Benchmark")

# --- Run on Combined (AR1 + Benchmark) features ---
X_combined = np.column_stack([X_ar1, X_benchmark])
combined_feature_names = ar1_feature_names + feature_names_benchmark
print("\n  === Combined (AR1 + Benchmark) Features ===")
results_combined = run_logreg_pipeline(X_combined, y_labels, combined_feature_names, "Combined")

# --- Save results to JSON ---
save_dir = os.path.join(os.path.dirname(OUTPUTS_PATH), "AnalyseResults")
os.makedirs(save_dir, exist_ok=True)
results_all = {"AR1": results_ar1, "Benchmark": results_benchmark, "Combined": results_combined}
results_json_path = os.path.join(save_dir, "AR1_vs_Benchmark_LogReg_results.json")
with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2, ensure_ascii=False)
print(f"\n    Results saved to: {results_json_path}")

# =========================================================================
# 6. VISUALIZATION
# =========================================================================
print("\n[6] Generating plots...")

# --- Compute mean unmasking step per token (step where mask transitions 1->0 most often) ---
# Using AR1 training data masks for this
masks_ar1 = np.array([masks[i] for i in ar1_positions], dtype=float)  # (N_ar1, T, D)
# Unmasking event: mask[t-1,d]=1 and mask[t,d]=0
unmask_events = masks_ar1[:, :-1, :] - masks_ar1[:, 1:, :]  # (N, T-1, D), positive = unmask
unmask_rate = np.mean(unmask_events > 0, axis=0)  # (T-1, D) - fraction of samples unmasked at each step
# For each token d, the step (integer) where it is most frequently unmasked across samples
peak_unmask_step = np.argmax(unmask_rate, axis=0)  # (D,) int in [0, T-2]

# --- Plotly heatmaps of phi and intercept ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_heatmap_plotly(data_correct, data_halluc, title, filename, peak_steps, vmin=None, vmax=None):
    """Create plotly heatmap with 2 subplots (correct vs halluc) and peak unmask markers.
    Axes: x = token d, y = step t (top=0, bottom=T-1).
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=["CORRECT model", "HALLUCINATION model"],
                        horizontal_spacing=0.08)
    
    T_plot = data_correct.shape[0]  # T-1 steps (from index 1 to T-1)
    D_plot = data_correct.shape[1]
    
    if vmin is None:
        vmax_abs = max(np.abs(data_correct).max(), np.abs(data_halluc).max())
        vmin, vmax = -vmax_abs, vmax_abs
    
    # z shape for heatmap: z[row][col] -> row=y, col=x
    # We want y=step (0..T_plot-1), x=token (0..D_plot-1)
    # data is (T_plot, D_plot), so z=data directly (row=step, col=token)
    
    # Correct model
    fig.add_trace(go.Heatmap(
        z=data_correct, x=np.arange(D_plot), y=np.arange(T_plot),
        colorscale='RdBu_r', zmin=vmin, zmax=vmax,
        colorbar=dict(x=0.45, len=0.9),
        hovertemplate='Token d=%{x}<br>Step t=%{y}<br>Value=%{z:.4f}<extra></extra>'
    ), row=1, col=1)
    
    # Hallucination model
    fig.add_trace(go.Heatmap(
        z=data_halluc, x=np.arange(D_plot), y=np.arange(T_plot),
        colorscale='RdBu_r', zmin=vmin, zmax=vmax,
        colorbar=dict(x=1.02, len=0.9),
        hovertemplate='Token d=%{x}<br>Step t=%{y}<br>Value=%{z:.4f}<extra></extra>'
    ), row=1, col=2)
    
    # Add peak unmask markers: for each token d, mark the step where unmasking is most frequent
    # x = token index, y = peak_steps + 1 (unmask_events index 0 = transition from step 0->1)
    marker_x = np.arange(D_plot)
    marker_y = peak_steps + 1  # unmask at index k means transition step k -> k+1, so mark at step k+1
    
    # Build hover text with the peak unmask info
    hover_correct = [f"Token d={d}<br>Mean unmask step={marker_y[d]:.1f}" for d in range(D_plot)]
    hover_halluc = [f"Token d={d}<br>Mean unmask step={marker_y[d]:.1f}" for d in range(D_plot)]
    
    fig.add_trace(go.Scatter(
        x=marker_x, y=marker_y,
        mode='markers',
        marker=dict(symbol='x', size=7, color='black', opacity=0.7, line=dict(width=1.5)),
        name='Peak unmask step',
        showlegend=True,
        hovertext=hover_correct,
        hoverinfo='text',
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=marker_x, y=marker_y,
        mode='markers',
        marker=dict(symbol='x', size=7, color='black', opacity=0.7, line=dict(width=1.5)),
        name='Peak unmask step',
        showlegend=False,
        hovertext=hover_halluc,
        hoverinfo='text',
    ), row=1, col=2)
    
    # Y-axis: step 0 at top, step 63 at bottom
    fig.update_yaxes(autorange='reversed')
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=700, width=1200,
        xaxis_title="Token d", yaxis_title="Step t",
        xaxis2_title="Token d", yaxis2_title="Step t",
    )
    
    fig.write_html(filename)
    print(f"    Saved: {filename}")



# Add a fictive row 0 (NaN) so the heatmap covers steps 0..63 and each token has its cross
def _add_row0(data):
    """Prepend a row of zeros at step 0 so shape goes from (T-1, D) to (T, D)."""
    return np.vstack([np.zeros((1, data.shape[1])), data])

make_heatmap_plotly(
    _add_row0(phi_correct[1:, :]), _add_row0(phi_halluc[1:, :]),
    "AR1 phi per (step, token)",
    os.path.join(save_dir, "AR1_Heatmap_Phi.html"),
    peak_unmask_step, vmin=-1, vmax=1
)

make_heatmap_plotly(
    _add_row0(intercept_correct[1:, :]), _add_row0(intercept_halluc[1:, :]),
    "AR1 intercept per (step, token)",
    os.path.join(save_dir, "AR1_Heatmap_Intercept.html"),
    peak_unmask_step
)

make_heatmap_plotly(
    _add_row0(sigma_correct[1:, :]), _add_row0(sigma_halluc[1:, :]),
    "AR1 sigma (residual std) per (step, token)",
    os.path.join(save_dir, "AR1_Heatmap_Sigma.html"),
    peak_unmask_step, vmin=0, vmax=max(sigma_correct[1:].max(), sigma_halluc[1:].max())
)

# Separate errors by true label
test_correct_mask = logreg_labels == 0
test_halluc_mask = logreg_labels == 1

# --- Plot: Mean & Variance of masked entropy (data vs AR1 reconstruction) per diffusion step ---
# For each half (AR1 fit / LogReg), for each class (correct / halluc),
# compute per-step: mean across masked tokens, then mean across samples.

def compute_per_step_masked_stats(entropy_arr, mask_arr):
    """For each sample and each step, compute mean and var of entropy on masked tokens.
    Returns: mean_per_step (T,), var_per_step (T,) — averaged across samples.
    """
    N, T, D = entropy_arr.shape
    mean_steps = np.zeros((N, T))
    var_steps = np.zeros((N, T))
    for n in range(N):
        for t in range(T):
            m = mask_arr[n, t, :] > 0
            if np.sum(m) > 0:
                vals = entropy_arr[n, t, m]
                mean_steps[n, t] = np.mean(vals)
                var_steps[n, t] = np.var(vals) if np.sum(m) > 1 else 0.0
    return np.mean(mean_steps, axis=0), np.mean(var_steps, axis=0)

def compute_per_step_masked_reconstruction_stats(entropy_arr, mask_arr, phi, intercept):
    """Same but on AR1 reconstruction x_pred = phi*x_{t-1} + intercept.
    At t=0, use actual data (initial condition)."""
    N, T, D = entropy_arr.shape
    mean_steps = np.zeros((N, T))
    var_steps = np.zeros((N, T))
    for n in range(N):
        # t=0: initial condition = actual data
        m = mask_arr[n, 0, :] > 0
        if np.sum(m) > 0:
            vals = entropy_arr[n, 0, m]
            mean_steps[n, 0] = np.mean(vals)
            var_steps[n, 0] = np.var(vals) if np.sum(m) > 1 else 0.0
        for t in range(1, T):
            m = mask_arr[n, t, :] > 0
            if np.sum(m) > 0:
                x_prev = entropy_arr[n, t-1, :]
                x_pred = phi[t, :] * x_prev + intercept[t, :]
                vals = x_pred[m]
                mean_steps[n, t] = np.mean(vals)
                var_steps[n, t] = np.var(vals) if np.sum(m) > 1 else 0.0
    return np.mean(mean_steps, axis=0), np.mean(var_steps, axis=0)

# AR1 half: split by label
entropy_ar1_correct = np.array([entropies[i] for i in ar1_correct_pos])
mask_ar1_correct = np.array([masks[i] for i in ar1_correct_pos], dtype=float)
entropy_ar1_halluc = np.array([entropies[i] for i in ar1_halluc_pos])
mask_ar1_halluc = np.array([masks[i] for i in ar1_halluc_pos], dtype=float)

# LogReg half: split by label
entropy_lr_correct = entropy_logreg[test_correct_mask]
mask_lr_correct = mask_logreg[test_correct_mask]
entropy_lr_halluc = entropy_logreg[test_halluc_mask]
mask_lr_halluc = mask_logreg[test_halluc_mask]

# Data stats
mean_data_ar1_c, var_data_ar1_c = compute_per_step_masked_stats(entropy_ar1_correct, mask_ar1_correct)
mean_data_ar1_h, var_data_ar1_h = compute_per_step_masked_stats(entropy_ar1_halluc, mask_ar1_halluc)
mean_data_lr_c, var_data_lr_c = compute_per_step_masked_stats(entropy_lr_correct, mask_lr_correct)
mean_data_lr_h, var_data_lr_h = compute_per_step_masked_stats(entropy_lr_halluc, mask_lr_halluc)

# AR1 reconstruction stats (using correct model fitted on AR1 half)
mean_recon_ar1_c, var_recon_ar1_c = compute_per_step_masked_reconstruction_stats(
    entropy_ar1_correct, mask_ar1_correct, phi_correct, intercept_correct)
mean_recon_ar1_h, var_recon_ar1_h = compute_per_step_masked_reconstruction_stats(
    entropy_ar1_halluc, mask_ar1_halluc, phi_halluc, intercept_halluc)
mean_recon_lr_c, var_recon_lr_c = compute_per_step_masked_reconstruction_stats(
    entropy_lr_correct, mask_lr_correct, phi_correct, intercept_correct)
mean_recon_lr_h, var_recon_lr_h = compute_per_step_masked_reconstruction_stats(
    entropy_lr_halluc, mask_lr_halluc, phi_halluc, intercept_halluc)

steps = np.arange(T_max)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# --- Plot 1: Mean Masked Entropy ---
ax = axes[0]
ax.plot(steps, mean_recon_ar1_c, 'o-', color='orange', markersize=3, label='Mean Masked Reconstruction Entropy (AR1) - Correct')
ax.plot(steps, mean_recon_ar1_h, 's-', color='blue', markersize=3, label='Mean Masked Reconstruction Entropy (AR1) - Hallucinations')
ax.plot(steps, mean_data_ar1_c, 'o-', color='goldenrod', markersize=3, label='Mean Masked Entropy (Data) - Correct')
ax.plot(steps, mean_data_ar1_h, 's-', color='cornflowerblue', markersize=3, label='Mean Masked Entropy (Data) - Hallucinations')
ax.set_title("Mean Masked Entropy Across Samples (AR1 Model) — AR1 fitting half", fontweight='bold')
ax.set_xlabel("Diffusion Step")
ax.set_ylabel("Mean Masked Entropy")
ax.legend(fontsize=8)
ax.grid(True)

# --- Plot 2: Variance Masked Entropy ---
ax = axes[1]
ax.plot(steps, var_recon_ar1_c, 'o-', color='red', markersize=3, label='Variance Masked Reconstruction Entropy (AR1) - Correct')
ax.plot(steps, var_recon_ar1_h, 's-', color='purple', markersize=3, label='Variance Masked Reconstruction Entropy (AR1) - Hallucinations')
ax.plot(steps, var_data_ar1_c, 'o-', color='orange', markersize=3, label='Variance Masked Entropy (Data) - Correct')
ax.plot(steps, var_data_ar1_h, 's-', color='cornflowerblue', markersize=3, label='Variance Masked Entropy (Data) - Hallucinations')
ax.set_title("Variance Masked Entropy Across Samples (AR1 Model) — AR1 fitting half", fontweight='bold')
ax.set_xlabel("Diffusion Step")
ax.set_ylabel("Variance Masked Entropy")
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "AR1_MeanVar_Reconstruction_AR1half.png"), dpi=150, bbox_inches='tight')
plt.show()

# Same plots for LogReg half
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(steps, mean_recon_lr_c, 'o-', color='orange', markersize=3, label='Mean Masked Reconstruction Entropy (AR1) - Correct')
ax.plot(steps, mean_recon_lr_h, 's-', color='blue', markersize=3, label='Mean Masked Reconstruction Entropy (AR1) - Hallucinations')
ax.plot(steps, mean_data_lr_c, 'o-', color='goldenrod', markersize=3, label='Mean Masked Entropy (Data) - Correct')
ax.plot(steps, mean_data_lr_h, 's-', color='cornflowerblue', markersize=3, label='Mean Masked Entropy (Data) - Hallucinations')
ax.set_title("Mean Masked Entropy Across Samples (AR1 Model) — LogReg half", fontweight='bold')
ax.set_xlabel("Diffusion Step")
ax.set_ylabel("Mean Masked Entropy")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[1]
ax.plot(steps, var_recon_lr_c, 'o-', color='red', markersize=3, label='Variance Masked Reconstruction Entropy (AR1) - Correct')
ax.plot(steps, var_recon_lr_h, 's-', color='purple', markersize=3, label='Variance Masked Reconstruction Entropy (AR1) - Hallucinations')
ax.plot(steps, var_data_lr_c, 'o-', color='orange', markersize=3, label='Variance Masked Entropy (Data) - Correct')
ax.plot(steps, var_data_lr_h, 's-', color='cornflowerblue', markersize=3, label='Variance Masked Entropy (Data) - Hallucinations')
ax.set_title("Variance Masked Entropy Across Samples (AR1 Model) — LogReg half", fontweight='bold')
ax.set_xlabel("Diffusion Step")
ax.set_ylabel("Variance Masked Entropy")
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "AR1_MeanVar_Reconstruction_LogReghalf.png"), dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot 1: Mean entropy error vs correct model ---
ax = axes[0, 0]
ax.hist(error_vs_correct[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(error_vs_correct[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Mean Masked Entropy Error vs CORRECT Model", fontweight='bold')
ax.set_xlabel("Mean Absolute Error")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True,)

# --- Plot 2: Mean entropy error vs hallucination model ---
ax = axes[0, 1]
ax.hist(error_vs_halluc[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(error_vs_halluc[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Mean Masked Entropy Error vs HALLUCINATION Model", fontweight='bold')
ax.set_xlabel("Mean Absolute Error")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

# --- Plot 3: Variance error vs correct model ---
ax = axes[1, 0]
ax.hist(var_error_vs_correct[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(var_error_vs_correct[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Mean Masked Variance Error vs CORRECT Model", fontweight='bold')
ax.set_xlabel("Mean |Var_real - Var_pred|")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

# --- Plot 4: Variance error vs hallucination model ---
ax = axes[1, 1]
ax.hist(var_error_vs_halluc[test_correct_mask], bins=30, alpha=0.6, color='green', label='True Correct', density=True)
ax.hist(var_error_vs_halluc[test_halluc_mask], bins=30, alpha=0.6, color='red', label='True Hallucination', density=True)
ax.set_title("Mean Masked Variance Error vs HALLUCINATION Model", fontweight='bold')
ax.set_xlabel("Mean |Var_real - Var_pred|")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

plt.suptitle("AR1 Train/Test: Per-Sample Error Distributions", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "AR1_TrainTest_ErrorDistributions.png"), dpi=150, bbox_inches='tight')
plt.show()

# --- Scatter plot: error_vs_correct vs error_vs_halluc ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.scatter(error_vs_correct[test_correct_mask], error_vs_halluc[test_correct_mask],
           alpha=0.5, color='green', label='True Correct', s=20)
ax.scatter(error_vs_correct[test_halluc_mask], error_vs_halluc[test_halluc_mask],
           alpha=0.5, color='red', label='True Hallucination', s=20)
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("Error vs Correct Model")
ax.set_ylabel("Error vs Hallucination Model")
ax.set_title("Mean Masked Entropy Error: Correct vs Halluc Model", fontweight='bold')
ax.legend()
ax.grid(True)

ax = axes[1]
ax.scatter(var_error_vs_correct[test_correct_mask], var_error_vs_halluc[test_correct_mask],
           alpha=0.5, color='green', label='True Correct', s=20)
ax.scatter(var_error_vs_correct[test_halluc_mask], var_error_vs_halluc[test_halluc_mask],
           alpha=0.5, color='red', label='True Hallucination', s=20)
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("Var Error vs Correct Model")
ax.set_ylabel("Var Error vs Hallucination Model")
ax.set_title("Mean Masked Variance Error: Correct vs Halluc Model", fontweight='bold')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "AR1_TrainTest_ScatterErrors.png"), dpi=150, bbox_inches='tight')
plt.show()

# --- Summary statistics ---
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
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

# Simple baseline: assign to model with lower error
print("\n" + "-"*70)
print("SIMPLE CLASSIFIER (baseline): assign to model with lower error")
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
print(f"  {'Method':<12} {'Accuracy':>10} {'ROC-AUC':>10} {'PR-AUC':>10} {'Pos Rate':>10}")
print(f"  {'AR1':<12} {results_ar1['test_accuracy']:>10.4f} {results_ar1['test_roc_auc']:>10.4f} {results_ar1['test_pr_auc']:>10.4f} {results_ar1['test_pos_rate']:>10.4f}")
print(f"  {'Benchmark':<12} {results_benchmark['test_accuracy']:>10.4f} {results_benchmark['test_roc_auc']:>10.4f} {results_benchmark['test_pr_auc']:>10.4f} {results_benchmark['test_pos_rate']:>10.4f}")
print(f"  {'Combined':<12} {results_combined['test_accuracy']:>10.4f} {results_combined['test_roc_auc']:>10.4f} {results_combined['test_pr_auc']:>10.4f} {results_combined['test_pos_rate']:>10.4f}")

print("\nDone!")
