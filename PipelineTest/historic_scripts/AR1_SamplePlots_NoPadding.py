"""
AR1 Per-Sample Visualization - NO PADDING
==========================================
Same as AR1_SamplePlots.py but excludes padding tokens.
Padding = token where the model proposes pad_token_id at ALL masked steps (in histories_x0).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask
from PipelineTest.historic_scripts.AnalyseResults import mergeOutputsList

# =========================================================================
# CONFIG
# =========================================================================
OUTPUTS_PATH = 'PipelineTest/res/LLADA_64steps_64tokens_lowconf/outputs_LLADA_64steps_64tokens_lowconf.pt'
EVAL_JSON = 'PipelineTest/res/eval/results_triviaqa_LLADA_64steps_64tokens_lowconf.json'
TOKENIZER_PATH = 'PipelineTest/res/LLADA_64steps_64tokens_lowconf/tokenizer_LLADA_64steps_64tokens_lowconf.pt'

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
# DETECT PADDING TOKENS (using tokenizer + proposed_sequence)
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
    
    Returns: padding (N, D) where True = padding (to exclude)
    """
    N = len(positions)
    T = len(outputs.histories_x0)
    D = outputs.max_new_tokens
    
    padding = np.zeros((N, D), dtype=bool)
    
    for idx_n, pos in enumerate(positions):
        start_idx = outputs.start_idx_history[pos]
        
        for d in range(D):
            masked_steps = []
            for t in range(T):
                if outputs.histories_mask[t][pos, start_idx + d].item() > 0:
                    masked_steps.append(t)
            
            if len(masked_steps) == 0:
                continue
            
            all_padding = True
            for t in masked_steps:
                proposed_token = outputs.histories_x0[t][pos, start_idx + d].item()
                if proposed_token != pad_token_id:
                    all_padding = False
                    break
            
            if all_padding:
                padding[idx_n, d] = True
    
    return padding


padding_2d_all = compute_padding_mask_from_x0(outputs, positions, pad_token_id)

n_padding_per_sample = padding_2d_all.sum(axis=1)
print(f"    Padding tokens per sample: mean={n_padding_per_sample.mean():.1f}, "
      f"min={n_padding_per_sample.min()}, max={n_padding_per_sample.max()}")

# =========================================================================
# FIT AR1 (same split as AR1_TrainTest_NoPadding)
# =========================================================================
print("[3] Splitting and fitting AR1 (no padding)...")
ar1_idx, logreg_idx = train_test_split(
    np.arange(len(samples)), test_size=0.5, stratify=labels, random_state=42
)

ar1_positions = positions[ar1_idx]
ar1_labels = labels[ar1_idx]

ar1_correct_pos = ar1_positions[ar1_labels == 0]
ar1_halluc_pos = ar1_positions[ar1_labels == 1]

entropy_train_correct = np.array([entropies[i] for i in ar1_correct_pos])
entropy_train_halluc = np.array([entropies[i] for i in ar1_halluc_pos])

# Padding for AR1 fitting half
padding_ar1 = padding_2d_all[ar1_idx]
padding_ar1_correct = padding_ar1[ar1_labels == 0]
padding_ar1_halluc = padding_ar1[ar1_labels == 1]


def fit_ar1_model_no_padding(entropy_tensor, padding_2d):
    """
    Fit AR(1) per-token, excluding padding tokens.
    padding_2d: (N, D) where True = padding
    """
    N, T, D = entropy_tensor.shape
    phi = np.zeros((T, D))
    intercept = np.zeros((T, D))
    sigma = np.zeros((T, D))
    for t in range(1, T):
        X_prev = entropy_tensor[:, t-1, :]
        X_curr = entropy_tensor[:, t, :]
        for d in range(D):
            valid = ~padding_2d[:, d]
            n_valid = np.sum(valid)
            if n_valid < 3:
                continue
            x = X_prev[valid, d]
            y = X_curr[valid, d]
            slope, intcpt, _, _, _ = linregress(x, y)
            phi[t, d] = slope
            intercept[t, d] = intcpt
            sigma[t, d] = np.std(y - (slope * x + intcpt))
    return phi, intercept, sigma


phi_correct, intercept_correct, sigma_correct = fit_ar1_model_no_padding(
    entropy_train_correct, padding_ar1_correct)
phi_halluc, intercept_halluc, sigma_halluc = fit_ar1_model_no_padding(
    entropy_train_halluc, padding_ar1_halluc)
print("    AR1 models fitted (no padding).")

# =========================================================================
# SELECT SAMPLES FROM LOGREG HALF
# =========================================================================
print(f"[4] Selecting {N_SAMPLES_TO_PLOT} samples from logreg half...")

logreg_positions = positions[logreg_idx]
logreg_labels = labels[logreg_idx]

logreg_orig_indices = set()
by_index = {d["index"]: d for d in data}
for d in data:
    idx = d["index"]
    if idx in hmap:
        pos = hmap[idx]
        if pos in set(logreg_positions):
            logreg_orig_indices.add(idx)

correct_sample_indices = [d["index"] for d in data if d["is_hallucination"] == 'no' and d["index"] in logreg_orig_indices]
halluc_sample_indices = [d["index"] for d in data if d["is_hallucination"] == 'yes' and d["index"] in logreg_orig_indices]
np.random.seed(42)
np.random.shuffle(correct_sample_indices)
np.random.shuffle(halluc_sample_indices)

n_per_cat = N_SAMPLES_TO_PLOT // 2
selected_correct = correct_sample_indices[:n_per_cat]
selected_halluc = halluc_sample_indices[:n_per_cat]

selected_all = [(idx, 0) for idx in selected_correct] + [(idx, 1) for idx in selected_halluc]
plot_positions = np.array([hmap[idx] for idx, _ in selected_all])
plot_labels = np.array([lbl for _, lbl in selected_all])

# Get padding for plot samples (find their index in `positions` array)
pos_to_idx = {pos: i for i, pos in enumerate(positions)}
plot_padding = np.array([padding_2d_all[pos_to_idx[pos]] for pos in plot_positions])

plot_questions = []
for idx, _ in selected_all:
    q = by_index[idx].get("question", "")
    q = q if len(q) <= 40 else q[:37] + "..."
    plot_questions.append(q)

print(f"    Picked {np.sum(plot_labels==0)} correct + {np.sum(plot_labels==1)} hallucination samples")

# =========================================================================
# PLOT [4]: Mean/Var masked entropy (excluding padding)
# =========================================================================
print("[5] Generating plots (no padding)...")

save_dir = os.path.join(os.path.dirname(OUTPUTS_PATH), "AnalyseResults")
os.makedirs(save_dir, exist_ok=True)

fig, axes = plt.subplots(len(plot_positions), 2, figsize=(16, 4 * len(plot_positions)))

for plot_idx in range(len(plot_positions)):
    pos = plot_positions[plot_idx]
    lbl = plot_labels[plot_idx]
    label_str = "Correct" if lbl == 0 else "Hallucination"
    pad_d = plot_padding[plot_idx]  # (D,) bool

    entropy_sample = np.array(entropies[pos])
    mask_sample = np.array(masks[pos], dtype=float)

    T, D = entropy_sample.shape

    mean_data = np.zeros(T)
    var_data = np.zeros(T)
    mean_recon_correct = np.zeros(T)
    var_recon_correct = np.zeros(T)
    mean_recon_halluc = np.zeros(T)
    var_recon_halluc = np.zeros(T)

    for t in range(T):
        # Effective mask: masked AND not padding
        m = (mask_sample[t, :] > 0) & (~pad_d)
        n_masked = np.sum(m)
        if n_masked > 0:
            vals = entropy_sample[t, m]
            mean_data[t] = np.mean(vals)
            var_data[t] = np.var(vals) if n_masked > 1 else 0.0

            if t == 0:
                mean_recon_correct[t] = mean_data[t]
                var_recon_correct[t] = var_data[t]
                mean_recon_halluc[t] = mean_data[t]
                var_recon_halluc[t] = var_data[t]
            else:
                x_prev = entropy_sample[t-1, :]
                x_pred_c = phi_correct[t, :] * x_prev + intercept_correct[t, :]
                vals_c = x_pred_c[m]
                mean_recon_correct[t] = np.mean(vals_c)
                var_recon_correct[t] = np.var(vals_c) if n_masked > 1 else 0.0

                x_pred_h = phi_halluc[t, :] * x_prev + intercept_halluc[t, :]
                vals_h = x_pred_h[m]
                mean_recon_halluc[t] = np.mean(vals_h)
                var_recon_halluc[t] = np.var(vals_h) if n_masked > 1 else 0.0

    steps = np.arange(T)
    n_pad = int(np.sum(pad_d))

    ax = axes[plot_idx, 0]
    ax.plot(steps, mean_data, 'o-', color='blue', markersize=3, label='Data')
    ax.plot(steps, mean_recon_correct, 's--', color='green', markersize=3, label='AR1 Recon (correct)')
    ax.plot(steps, mean_recon_halluc, '^--', color='red', markersize=3, label='AR1 Recon (halluc)')
    ax.set_title(f"Sample {plot_idx+1} [{label_str}] {plot_questions[plot_idx]} — Mean (pad={n_pad})", fontweight='bold')
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Mean Entropy (masked, no pad)")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[plot_idx, 1]
    ax.plot(steps, var_data, 'o-', color='blue', markersize=3, label='Data')
    ax.plot(steps, var_recon_correct, 's--', color='green', markersize=3, label='AR1 Recon (correct)')
    ax.plot(steps, var_recon_halluc, '^--', color='red', markersize=3, label='AR1 Recon (halluc)')
    ax.set_title(f"Sample {plot_idx+1} [{label_str}] {plot_questions[plot_idx]} — Var (pad={n_pad})", fontweight='bold')
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Variance Entropy (masked, no pad)")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle("Per-Sample: Data vs AR1 Reconstruction (Masked, No Padding)", fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
save_path = os.path.join(save_dir, "AR1_PerSample_MeanVar_NoPadding.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path}")

# =========================================================================
# PLOT [5]: 64th unmasked token (excluding padding tokens from order)
# =========================================================================
print("[6] Generating 64th-unmasked-token plots (no padding)...")

TOKEN_RANK = 64

fig, axes = plt.subplots(len(plot_positions), 1, figsize=(14, 4 * len(plot_positions)))

for plot_idx in range(len(plot_positions)):
    pos = plot_positions[plot_idx]
    lbl = plot_labels[plot_idx]
    label_str = "Correct" if lbl == 0 else "Hallucination"
    pad_d = plot_padding[plot_idx]

    entropy_sample = np.array(entropies[pos])
    mask_sample = np.array(masks[pos], dtype=float)
    T, D = entropy_sample.shape

    unmask_step = np.full(D, T)
    for d in range(D):
        for t in range(1, T):
            if mask_sample[t-1, d] == 1 and mask_sample[t, d] == 0:
                unmask_step[d] = t
                break

    # Filter: unmasked AND not padding
    valid_tokens = np.where((unmask_step < T) & (~pad_d))[0]
    valid_unmask = unmask_step[valid_tokens]
    token_order = valid_tokens[np.argsort(valid_unmask)]

    if len(token_order) < TOKEN_RANK:
        target_token = token_order[-1] if len(token_order) > 0 else 0
    else:
        target_token = token_order[TOKEN_RANK - 1]

    entropy_data = entropy_sample[:, target_token]

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
    ax.plot(steps, recon_correct, 's--', color='green', markersize=3, label='AR1 Recon (correct)')
    ax.plot(steps, recon_halluc_model, '^--', color='red', markersize=3, label='AR1 Recon (halluc)')
    ax.axvline(x=unmask_step[target_token], color='gray', linestyle=':', alpha=0.7, label=f'Unmask step={unmask_step[target_token]}')
    ax.set_title(f"Sample {plot_idx+1} [{label_str}] {plot_questions[plot_idx]} — {TOKEN_RANK}th unmasked (no pad, d={target_token})", fontweight='bold')
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Entropy")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle(f"Per-Sample: {TOKEN_RANK}th Unmasked Token (No Padding) — Data vs AR1", fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
save_path_token = os.path.join(save_dir, f"AR1_PerSample_Token{TOKEN_RANK}_NoPadding.png")
plt.savefig(save_path_token, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_token}")

# =========================================================================
# PLOT [6]: 10 token trajectories (excluding padding)
# =========================================================================
print("[7] Generating 10-token trajectory plots (no padding)...")

N_TOKENS_TRAJ = 10
colors_tokens = plt.cm.tab10(np.linspace(0, 1, N_TOKENS_TRAJ))

fig, axes = plt.subplots(len(plot_positions), 2, figsize=(18, 4 * len(plot_positions)))

for plot_idx in range(len(plot_positions)):
    pos = plot_positions[plot_idx]
    lbl = plot_labels[plot_idx]
    label_str = "Correct" if lbl == 0 else "Hallucination"
    pad_d = plot_padding[plot_idx]

    entropy_sample = np.array(entropies[pos])
    mask_sample = np.array(masks[pos], dtype=float)
    T, D = entropy_sample.shape

    unmask_step = np.full(D, T)
    for d in range(D):
        for t in range(1, T):
            if mask_sample[t-1, d] == 1 and mask_sample[t, d] == 0:
                unmask_step[d] = t
                break

    # Only non-padding tokens
    valid_tokens = np.where((unmask_step < T) & (~pad_d))[0]
    valid_unmask = unmask_step[valid_tokens]
    token_order = valid_tokens[np.argsort(valid_unmask)]

    if len(token_order) >= N_TOKENS_TRAJ:
        pick_indices = np.linspace(0, len(token_order) - 1, N_TOKENS_TRAJ, dtype=int)
        selected_tokens = token_order[pick_indices]
    else:
        selected_tokens = token_order

    steps = np.arange(T)

    # Left: correct model
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

    ax_left.set_title(f"Sample {plot_idx+1} [{label_str}] — Correct model (no pad)", fontweight='bold')
    ax_left.set_xlabel("Diffusion Step")
    ax_left.set_ylabel("Entropy")
    ax_left.legend(fontsize=6, ncol=2)
    ax_left.grid(True)

    # Right: halluc model
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

    ax_right.set_title(f"Sample {plot_idx+1} [{label_str}] — Halluc model (no pad)", fontweight='bold')
    ax_right.set_xlabel("Diffusion Step")
    ax_right.set_ylabel("Entropy")
    ax_right.legend(fontsize=6, ncol=2)
    ax_right.grid(True)

plt.suptitle("Per-Sample: 10 Token Trajectories (No Padding) — Data (solid) vs AR1 (dashed)", fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
save_path_traj = os.path.join(save_dir, "AR1_PerSample_10Tokens_CorrectVsHalluc_NoPadding.png")
plt.savefig(save_path_traj, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Saved: {save_path_traj}")
