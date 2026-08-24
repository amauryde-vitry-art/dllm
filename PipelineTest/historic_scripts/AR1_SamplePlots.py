"""
AR1 Per-Sample Visualization
==============================
Pick ~10 random samples and plot per-step:
  - Mean masked entropy (data vs AR1 reconstruction)
  - Variance masked entropy (data vs AR1 reconstruction)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask
from PipelineTest.historic_scripts.AnalyseResults import mergeOutputsList

# =========================================================================
# CONFIG
# =========================================================================
OUTPUTS_PATH = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt'
EVAL_JSON = 'PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json'

N_SAMPLES_TO_PLOT = 20

# =========================================================================
# LOAD DATA
# =========================================================================
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
# FIT AR1 (same split as AR1_TrainTest)
# =========================================================================
print("[2] Splitting and fitting AR1...")
ar1_idx, logreg_idx = train_test_split(
    np.arange(len(samples)), test_size=0.5, stratify=labels, random_state=42
)

ar1_positions = positions[ar1_idx]
ar1_labels = labels[ar1_idx]

ar1_correct_pos = ar1_positions[ar1_labels == 0]
ar1_halluc_pos = ar1_positions[ar1_labels == 1]

entropy_train_correct = np.array([entropies[i] for i in ar1_correct_pos])
entropy_train_halluc = np.array([entropies[i] for i in ar1_halluc_pos])


def fit_ar1_model(entropy_tensor):
    N, T, D = entropy_tensor.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    sigma = np.zeros((T, D))
    for t in range(1, T):
        X_prev = entropy_tensor[:, t-1, :]
        X_curr = entropy_tensor[:, t, :]
        for d in range(D):
            slope, intcpt, _, _, _ = linregress(X_prev[:, d], X_curr[:, d])
            phi[t, d] = slope
            intercept[t, d] = intcpt
            sigma[t, d] = np.std(X_curr[:, d] - (slope * X_prev[:, d] + intcpt))
    return phi, intercept, sigma


phi_correct, intercept_correct, sigma_correct = fit_ar1_model(entropy_train_correct)
phi_halluc, intercept_halluc, sigma_halluc = fit_ar1_model(entropy_train_halluc)
print("    AR1 models fitted.")

# =========================================================================
# SELECT SAMPLES FROM LOGREG HALF (test set, not used for AR1 fitting)
# =========================================================================
print(f"[3] Selecting {N_SAMPLES_TO_PLOT} samples from logreg half...")

logreg_positions = positions[logreg_idx]
logreg_labels = labels[logreg_idx]

# Get the original JSON indices for the logreg half
logreg_orig_indices = set()
by_index = {d["index"]: d for d in data}
for d in data:
    idx = d["index"]
    if idx in hmap:
        pos = hmap[idx]
        if pos in set(logreg_positions):
            logreg_orig_indices.add(idx)

# Filter correct/halluc to only logreg half, then shuffle
correct_sample_indices = [d["index"] for d in data if d["is_hallucination"] == 'no' and d["index"] in logreg_orig_indices]
halluc_sample_indices = [d["index"] for d in data if d["is_hallucination"] == 'yes' and d["index"] in logreg_orig_indices]
np.random.seed(42)
np.random.shuffle(correct_sample_indices)
np.random.shuffle(halluc_sample_indices)

n_per_cat = N_SAMPLES_TO_PLOT // 2
selected_correct = correct_sample_indices[:n_per_cat]
selected_halluc = halluc_sample_indices[:n_per_cat]

# Map to positions and labels
selected_all = [(idx, 0) for idx in selected_correct] + [(idx, 1) for idx in selected_halluc]
plot_positions = np.array([hmap[idx] for idx, _ in selected_all])
plot_labels = np.array([lbl for _, lbl in selected_all])

# Build question titles (truncated)
plot_questions = []
for idx, _ in selected_all:
    q = by_index[idx].get("question", "")
    q = q if len(q) <= 40 else q[:37] + "..."
    plot_questions.append(q)

print(f"    Picked {np.sum(plot_labels==0)} correct + {np.sum(plot_labels==1)} hallucination samples")
print(f"    First correct indices: {selected_correct[:5]}")
print(f"    First halluc indices: {selected_halluc[:5]}")

# =========================================================================
# COMPUTE PER-SAMPLE STATS AND PLOT
# =========================================================================
print("[4] Generating plots...")

save_dir = os.path.join(os.path.dirname(OUTPUTS_PATH), "AnalyseResults")
os.makedirs(save_dir, exist_ok=True)

fig, axes = plt.subplots(len(plot_positions), 2, figsize=(16, 4 * len(plot_positions)))

for plot_idx in range(len(plot_positions)):
    pos = plot_positions[plot_idx]
    lbl = plot_labels[plot_idx]
    label_str = "Correct" if lbl == 0 else "Hallucination"

    entropy_sample = np.array(entropies[pos])
    mask_sample = np.array(masks[pos], dtype=float)

    T, D = entropy_sample.shape

    # Compute per-step stats: data + recon with correct model + recon with halluc model
    mean_data = np.zeros(T)
    var_data = np.zeros(T)
    mean_recon_correct = np.zeros(T)
    var_recon_correct = np.zeros(T)
    mean_recon_halluc = np.zeros(T)
    var_recon_halluc = np.zeros(T)

    for t in range(T):
        m = mask_sample[t, :] > 0
        n_masked = np.sum(m)
        if n_masked > 0:
            vals = entropy_sample[t, m]
            mean_data[t] = np.mean(vals)
            var_data[t] = np.var(vals) if n_masked > 1 else 0.0

            if t == 0:
                # Initial condition: use actual data
                mean_recon_correct[t] = mean_data[t]
                var_recon_correct[t] = var_data[t]
                mean_recon_halluc[t] = mean_data[t]
                var_recon_halluc[t] = var_data[t]
            else:
                x_prev = entropy_sample[t-1, :]
                # Reconstruction with correct model
                x_pred_c = phi_correct[t, :] * x_prev + intercept_correct[t, :]
                vals_c = x_pred_c[m]
                mean_recon_correct[t] = np.mean(vals_c)
                var_recon_correct[t] = np.var(vals_c) if n_masked > 1 else 0.0
                # Reconstruction with hallucination model
                x_pred_h = phi_halluc[t, :] * x_prev + intercept_halluc[t, :]
                vals_h = x_pred_h[m]
                mean_recon_halluc[t] = np.mean(vals_h)
                var_recon_halluc[t] = np.var(vals_h) if n_masked > 1 else 0.0

    steps = np.arange(T)

    # Mean plot
    ax = axes[plot_idx, 0]
    ax.plot(steps, mean_data, 'o-', color='blue', markersize=3, label='Data')
    ax.plot(steps, mean_recon_correct, 's--', color='green', markersize=3, label='AR1 Recon (correct model)')
    ax.plot(steps, mean_recon_halluc, '^--', color='red', markersize=3, label='AR1 Recon (halluc model)')
    ax.set_title(f"Sample {plot_idx+1} [{label_str}] {plot_questions[plot_idx]} — Mean Masked Entropy", fontweight='bold')
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Mean Entropy (masked tokens)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Variance plot
    ax = axes[plot_idx, 1]
    ax.plot(steps, var_data, 'o-', color='blue', markersize=3, label='Data')
    ax.plot(steps, var_recon_correct, 's--', color='green', markersize=3, label='AR1 Recon (correct model)')
    ax.plot(steps, var_recon_halluc, '^--', color='red', markersize=3, label='AR1 Recon (halluc model)')
    ax.set_title(f"Sample {plot_idx+1} [{label_str}] {plot_questions[plot_idx]} — Var Masked Entropy", fontweight='bold')
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Variance Entropy (masked tokens)")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle("Per-Sample: Data vs AR1 Reconstruction (Masked Tokens)", fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
save_path = os.path.join(save_dir, "AR1_PerSample_MeanVar.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path}")

# =========================================================================
# PLOT: Entropy of the 60th unmasked token over time
# =========================================================================
print("[5] Generating 64th-unmasked-token plots...")

TOKEN_RANK = 64  # 64th token to be unmasked

# Use same samples as above
fig, axes = plt.subplots(len(plot_positions), 1, figsize=(14, 4 * len(plot_positions)))

for plot_idx in range(len(plot_positions)):
    pos = plot_positions[plot_idx]
    lbl = plot_labels[plot_idx]
    label_str = "Correct" if lbl == 0 else "Hallucination"

    entropy_sample = np.array(entropies[pos])  # (T, D)
    mask_sample = np.array(masks[pos], dtype=float)  # (T, D)
    T, D = entropy_sample.shape

    # Find unmasking order: for each token d, find the step where it gets unmasked
    # (first step t where mask[t-1,d]=1 and mask[t,d]=0)
    unmask_step = np.full(D, T)  # default: never unmasked
    for d in range(D):
        for t in range(1, T):
            if mask_sample[t-1, d] == 1 and mask_sample[t, d] == 0:
                unmask_step[d] = t
                break

    # Filter out tokens that were never masked (unmask_step == T)
    valid_tokens = np.where(unmask_step < T)[0]
    valid_unmask = unmask_step[valid_tokens]
    token_order = valid_tokens[np.argsort(valid_unmask)]
    
    if len(token_order) < D:
        # Add never-unmasked tokens at the end
        never_unmasked = np.where(unmask_step == T)[0]
        token_order = np.concatenate([token_order, never_unmasked])


    if TOKEN_RANK == 64:
        target_token = token_order[63]
    else:
        target_token = token_order[TOKEN_RANK - 1] 

    # Extract entropy trajectory for this token
    entropy_data = entropy_sample[:, target_token]

    # AR1 reconstruction for this token
    recon_correct = np.zeros(T)
    recon_halluc_model = np.zeros(T)
    recon_correct[0] = entropy_data[0]
    recon_halluc_model[0] = entropy_data[0]
    for t in range(1, T):
        recon_correct[t] = phi_correct[t, target_token] * entropy_sample[t-1, target_token] + intercept_correct[t, target_token]
        recon_halluc_model[t] = phi_halluc[t, target_token] * entropy_sample[t-1, target_token] + intercept_halluc[t, target_token]

    steps = np.arange(T)

    ax = axes[plot_idx]
    ax.plot(steps, entropy_data, 'o-', color='blue', markersize=3, label='Data')
    ax.plot(steps, recon_correct, 's--', color='green', markersize=3, label='AR1 Recon (correct model)')
    ax.plot(steps, recon_halluc_model, '^--', color='red', markersize=3, label='AR1 Recon (halluc model)')
    ax.axvline(x=unmask_step[target_token], color='gray', linestyle=':', alpha=0.7, label=f'Unmask step={unmask_step[target_token]}')
    ax.set_title(f"Sample {plot_idx+1} [{label_str}] {plot_questions[plot_idx]} — Entropy of {TOKEN_RANK}th unmasked token (d={target_token})", fontweight='bold')
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Entropy")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle(f"Per-Sample: Entropy of {TOKEN_RANK}th Unmasked Token — Data vs AR1", fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
save_path_token = os.path.join(save_dir, f"AR1_PerSample_Token{TOKEN_RANK}.png")
plt.savefig(save_path_token, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_token}")

# =========================================================================
# PLOT: 10 token trajectories — correct model vs halluc model side by side
# =========================================================================
print("[6] Generating 10-token trajectory plots (correct vs halluc model)...")

N_TOKENS_TRAJ = 10
colors_tokens = plt.cm.tab10(np.linspace(0, 1, N_TOKENS_TRAJ))

fig, axes = plt.subplots(len(plot_positions), 2, figsize=(18, 4 * len(plot_positions)))

for plot_idx in range(len(plot_positions)):
    pos = plot_positions[plot_idx]
    lbl = plot_labels[plot_idx]
    label_str = "Correct" if lbl == 0 else "Hallucination"

    entropy_sample = np.array(entropies[pos])  # (T, D)
    mask_sample = np.array(masks[pos], dtype=float)  # (T, D)
    T, D = entropy_sample.shape

    # Unmasking order
    unmask_step = np.full(D, T)
    for d in range(D):
        for t in range(1, T):
            if mask_sample[t-1, d] == 1 and mask_sample[t, d] == 0:
                unmask_step[d] = t
                break

    valid_tokens = np.where(unmask_step < T)[0]
    valid_unmask = unmask_step[valid_tokens]
    token_order = valid_tokens[np.argsort(valid_unmask)]

    # Pick N_TOKENS_TRAJ tokens evenly spaced in the unmasking order
    if len(token_order) >= N_TOKENS_TRAJ:
        pick_indices = np.linspace(0, len(token_order) - 1, N_TOKENS_TRAJ, dtype=int)
        selected_tokens = token_order[pick_indices]
    else:
        selected_tokens = token_order

    steps = np.arange(T)

    # Left: correct model reconstruction
    ax_left = axes[plot_idx, 0]
    for k, tok_d in enumerate(selected_tokens):
        entropy_data = entropy_sample[:, tok_d]
        recon = np.zeros(T)
        recon[0] = entropy_data[0]
        for t in range(1, T):
            recon[t] = phi_correct[t, tok_d] * entropy_sample[t-1, tok_d] + intercept_correct[t, tok_d]

        ax_left.plot(steps, entropy_data, '-', color=colors_tokens[k], alpha=0.4, linewidth=1)
        ax_left.plot(steps, recon, '--', color=colors_tokens[k], linewidth=1.5,
                     label=f'd={tok_d} (unmask={unmask_step[tok_d]})')

    ax_left.set_title(f"Sample {plot_idx+1} [{label_str}] — Correct model recon", fontweight='bold')
    ax_left.set_xlabel("Diffusion Step")
    ax_left.set_ylabel("Entropy")
    ax_left.legend(fontsize=6, ncol=2)
    ax_left.grid(True)

    # Right: hallucination model reconstruction
    ax_right = axes[plot_idx, 1]
    for k, tok_d in enumerate(selected_tokens):
        entropy_data = entropy_sample[:, tok_d]
        recon = np.zeros(T)
        recon[0] = entropy_data[0]
        for t in range(1, T):
            recon[t] = phi_halluc[t, tok_d] * entropy_sample[t-1, tok_d] + intercept_halluc[t, tok_d]

        ax_right.plot(steps, entropy_data, '-', color=colors_tokens[k], alpha=0.4, linewidth=1)
        ax_right.plot(steps, recon, '--', color=colors_tokens[k], linewidth=1.5,
                      label=f'd={tok_d} (unmask={unmask_step[tok_d]})')

    ax_right.set_title(f"Sample {plot_idx+1} [{label_str}] — Halluc model recon", fontweight='bold')
    ax_right.set_xlabel("Diffusion Step")
    ax_right.set_ylabel("Entropy")
    ax_right.legend(fontsize=6, ncol=2)
    ax_right.grid(True)

plt.suptitle("Per-Sample: 10 Token Trajectories — Data (solid) vs AR1 Recon (dashed)", fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
save_path_traj = os.path.join(save_dir, "AR1_PerSample_10Tokens_CorrectVsHalluc.png")
plt.savefig(save_path_traj, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_traj}")
