import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich import json
from scipy.stats import linregress

import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask
from PipelineTest.AnalyseResults import mergeOutputsList

# df_mean_entropy = pd.read_csv('PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/VarAndMeanEntropy.csv')

# df_mean_entropy_correct = df_mean_entropy[
#     df_mean_entropy['isHallucination'] == 'no'
# ][['Diffusion Step', 'mean_entropy', 'SampleIndex']]
# matrix_df_correct = df_mean_entropy_correct.pivot(
#     index='SampleIndex', columns='Diffusion Step', values='mean_entropy'
# )

# df_mean_entropy_hallucination = df_mean_entropy[
#     df_mean_entropy['isHallucination'] == 'yes'
# ][['Diffusion Step', 'mean_entropy', 'SampleIndex']]
# matrix_df_hallucination = df_mean_entropy_hallucination.pivot(
#     index='SampleIndex', columns='Diffusion Step', values='mean_entropy'
# )

# # Aligne explicitement les deux populations sur les mêmes steps (0..63 dans ce dataset).
# common_steps = sorted(set(matrix_df_correct.columns).intersection(matrix_df_hallucination.columns))
# matrix_df_correct = matrix_df_correct[common_steps]
# matrix_df_hallucination = matrix_df_hallucination[common_steps]

# entropy_matrix_correct = matrix_df_correct.to_numpy()
# entropy_matrix_hallucination = matrix_df_hallucination.to_numpy()

# # Nombre total de steps dérivé des données (évite les désalignements hardcodés).
# T_max = len(common_steps)
# t_steps = np.array(common_steps)

# # Initialisation des vecteurs de variance (conditions initiales prises des données).
# V_fabricated = np.zeros(T_max)
# V_correct = np.zeros(T_max)
# V_correct[0] = np.var(entropy_matrix_correct[:, 0])
# V_fabricated[0] = np.var(entropy_matrix_hallucination[:, 0])


# def fit_tvp_ar1(entropy_matrix):
#     """
#     entropy_matrix: array de taille (N_samples, T_steps) -> ex: (2048, 64)
#     """
#     n_samples, t_max = entropy_matrix.shape
    
#     # Tableaux pour stocker les paramètres fittés à chaque instant t
#     phi_fitted = np.zeros(t_max)
#     theta_fitted = np.zeros(t_max)
#     sigma_fitted = np.zeros(t_max)
    
#     # Le fit commence à t=1 puisqu'il faut un état précédent (t-1)
#     for t in range(1, t_max):
#         x_prev = entropy_matrix[:, t-1] # Les 2048 valeurs au step précédent
#         x_curr = entropy_matrix[:, t]   # Les 2048 valeurs au step actuel
        
#         # Régression linéaire : X_t = slope * X_{t-1} + intercept
#         slope, intercept, r_value, p_value, std_err = linregress(x_prev, x_curr)
#         print(f"Step {t}: slope={slope}, intercept={intercept}, r_value={r_value}, p_value={p_value}, std_err={std_err}")
        
#         # Calcul des résidus pour avoir le bruit sigma
#         predictions = slope * x_prev + intercept
#         residuals = x_curr - predictions
        
#         # Extraction des paramètres du modèle
#         phi_fitted[t] = slope
#         sigma_fitted[t] = np.std(residuals)
        
#         # Sécurité pour éviter la division par zéro si phi vaut 1
#         if abs(1 - slope) > 1e-5:
#             theta_fitted[t] = intercept / (1 - slope)
#         else:
#             theta_fitted[t] = np.mean(x_curr)
            
#     return phi_fitted, sigma_fitted, theta_fitted


# phi_fitted_correct, sigma_fitted_correct, theta_fitted_correct = fit_tvp_ar1(entropy_matrix_correct)
# phi_fitted_hallucination, sigma_fitted_hallucination, theta_fitted_hallucination = fit_tvp_ar1(entropy_matrix_hallucination)

# # =========================================================================
# # RECONSTRUCTION DE L'ENTROPIE (MOYENNE ET VARIANCE THEORIQUES) VIA AR(1)
# # =========================================================================

# # 1. Initialisation des vecteurs pour la Moyenne (E) et la Variance (V) reconstruites
# E_correct_rec = np.zeros(T_max)
# E_hallucination_rec = np.zeros(T_max)
# V_correct_rec = np.zeros(T_max)
# V_hallucination_rec = np.zeros(T_max)

# # Conditions initiales extraites directement du premier pas de tes matrices
# E_correct_rec[0] = np.mean(entropy_matrix_correct[:, 0])
# E_hallucination_rec[0] = np.mean(entropy_matrix_hallucination[:, 0])
# V_correct_rec[0] = np.var(entropy_matrix_correct[:, 0])
# V_fabricated[0] = np.var(entropy_matrix_hallucination[:, 0]) # Garde ton init originale
# V_hallucination_rec[0] = V_fabricated[0]

# # 2. Application des équations de récurrence de l'AR(1) temporellement hétérogène
# for t in range(1, T_max):
#     # --- Cas population SANS hallucination (Correct) ---
#     phi_c = phi_fitted_correct[t]
#     theta_c = theta_fitted_correct[t]
#     sigma_c = sigma_fitted_correct[t]
    
#     E_correct_rec[t] = phi_c * E_correct_rec[t-1] + (1 - phi_c) * theta_c
#     V_correct_rec[t] = (phi_c**2) * V_correct_rec[t-1] + sigma_c**2
    
#     # --- Cas population AVEC hallucination (Fabricated) ---
#     phi_h = phi_fitted_hallucination[t]
#     theta_h = theta_fitted_hallucination[t]
#     sigma_h = sigma_fitted_hallucination[t]
    
#     E_hallucination_rec[t] = phi_h * E_hallucination_rec[t-1] + (1 - phi_h) * theta_h
#     V_hallucination_rec[t] = (phi_h**2) * V_hallucination_rec[t-1] + sigma_h**2


# # =========================================================================
# # PLOT DES RESULTATS RECONSTRUITS
# # =========================================================================

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

# # --- Graphique 1 : Reconstruction de la Moyenne ---
# mean_correct_data = np.mean(entropy_matrix_correct, axis=0)
# mean_hallucination_data = np.mean(entropy_matrix_hallucination, axis=0)

# ax1.plot(t_steps, E_correct_rec, color='#2ecc71', linewidth=2.5, label='Correct (Fitted AR(1))')
# ax1.plot(t_steps, E_hallucination_rec, color='#e74c3c', linewidth=2.5, label='Fabricated (Fitted AR(1))')
# ax1.plot(t_steps, mean_correct_data, color='#27ae60', linestyle='--', linewidth=2.0, alpha=0.9, label='Correct (Data mean)')
# ax1.plot(t_steps, mean_hallucination_data, color='#c0392b', linestyle='--', linewidth=2.0, alpha=0.9, label='Fabricated (Data mean)')

# ax1.set_title("Reconstruction TVP-AR(1) : Mean Entropy", fontsize=11, fontweight='bold')
# ax1.set_xlabel("Diffusion Step")
# ax1.set_ylabel("MeanMaskedEntropyAcrossTokens")
# ax1.grid(True, linestyle='-', alpha=0.3)
# ax1.legend()

# ax2.plot(t_steps, V_correct_rec, color='#3498db', linewidth=2.5, label='Correct (Fitted AR(1))')
# ax2.plot(t_steps, V_hallucination_rec, color='#9b59b6', linewidth=2.5, label='Fabricated (Fitted AR(1))')
# ax2.plot(t_steps, V_correct, color='#2980b9', linestyle='--', linewidth=2.0, alpha=0.9, label='Correct (Data variance)')
# ax2.plot(t_steps, V_fabricated, color='#8e44ad', linestyle='--', linewidth=2.0, alpha=0.9, label='Fabricated (Data variance)')

# ax2.set_title("Reconstruction TVP-AR(1) : Variance Entropy", fontsize=11, fontweight='bold')
# ax2.set_xlabel("Diffusion Step")
# ax2.set_ylabel("VarianceMaskedEntropyAcrossTokens")
# ax2.grid(True, linestyle='-', alpha=0.3)
# ax2.legend()

# plt.suptitle("Validation de la Modélisation AR(1) par rapport aux Données Réelles", fontsize=12, fontweight='bold')
# plt.tight_layout()
# plt.savefig("PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/AR1_Reconstruction_Validation.png", dpi=150)
# plt.show()



def fit_tvp_var1(entropy_tensor,):
    """
    Fit a time-varying parameter VAR(1) model to the given entropy tensor.
    """
    
    N, T, D = entropy_tensor.shape  # D = 64 tokens
    
    # Initialize output arrays
    phi_fitted = np.zeros((T, D))
    theta_fitted = np.zeros((T, D))
    sigma_fitted = np.zeros((T, D))
    
    # Fit for each timestep t=1 to T-1
    for t in range(1, T):
        print(f"[VAR(1)] Fitting timestep {t}/{T-1}...", end='\r')
        
        # Extract data at time t-1 and t
        X_prev = entropy_tensor[:, t-1, :]  # (N, 64) - explanatory
        X_curr = entropy_tensor[:, t, :]    # (N, 64) - target
        
        # Regression for each output dimension (token d)
        for d in range(D):
            y_d = X_curr[:, d]      # (N,) - target for token d
            x_d = X_prev[:, d]      # (N,) - explanatory for token d
            
            slope, intercept, r_value, p_value, std_err = linregress(x_d, y_d)

            # Calcul des résidus pour avoir le bruit sigma
            predictions = slope * x_d + intercept
            residuals = y_d - predictions
            
            # Extraction des paramètres du modèle
            phi_fitted[t, d] = slope
            sigma_fitted[t, d] = np.std(residuals)

            # Sécurité pour éviter la division par zéro si phi vaut 1
            if abs(1 - slope) > 1e-5:
                theta_fitted[t, d] = intercept / (1 - slope)
            else:
                theta_fitted[t, d] = np.mean(y_d)

    return phi_fitted, sigma_fitted, theta_fitted



def reconstruct_AR1(phi_fitted, sigma_fitted, theta_fitted, init):
    """
    Reconstruct the mean and variance over time using the fitted AR(1) parameters.
    """
    T_max, D = phi_fitted.shape
    
    reconstruct = np.zeros((T_max, D))
    reconstruct[0, :] = init  # Initial condition
    for t in range(1, T_max):
        for d in range(D):
            phi = phi_fitted[t, d]
            theta = theta_fitted[t, d]
            sigma = sigma_fitted[t, d]
            
            # Update mean and variance using AR(1) equations
            reconstruct[t, d] = phi * reconstruct[t-1, d] + (1 - phi) * theta + np.random.normal(0, sigma)
    return reconstruct

def compute_reconstruction_for_each_sample(entropy_tensor):
    """
    Compute the AR(1) reconstruction for each sample in the entropy tensor.
    """
    N, T, D = entropy_tensor.shape
    reconstructions = np.zeros((N, T, D))
    phi_fitted, sigma_fitted, theta_fitted = fit_tvp_var1(entropy_tensor)

    for n in range(N):
        print(f"[Sample {n+1}/{N}] Computing reconstruction...", end='\r')
        sample_data = entropy_tensor[n:n+1, :, :]  # Shape (1, T, D)
        
        # Fit AR(1) model for this sample        
        # Reconstruct using fitted parameters
        init = sample_data[0, 0, :]  # Initial condition for this sample
        reconstructions[n] = reconstruct_AR1(phi_fitted, sigma_fitted, theta_fitted, init)
    
    return reconstructions

def getMeanReconstructionError(entropy_tensor, masks_tensor):
    """
    Compute mean reconstruction error across all samples,
    summing only on masked tokens.

    Args:
        entropy_tensor: (N, T, D)
        masks_tensor: (N, T, D) with 1/True for masked tokens
    """
    reconstructions = compute_reconstruction_for_each_sample(entropy_tensor)
    N_sample, T_max, D = entropy_tensor.shape
    masks_tensor = np.asarray(masks_tensor, dtype=float)
    if masks_tensor.shape != entropy_tensor.shape:
        raise ValueError(f"masks_tensor shape {masks_tensor.shape} must match entropy_tensor shape {entropy_tensor.shape}")

    errors = np.abs(entropy_tensor - reconstructions)
    masked_errors = errors * masks_tensor
    cum_error = np.sum(masked_errors, axis=2) / np.sum(masks_tensor, axis=2)  # Sum only on masked tokens
    mean_error = np.mean(cum_error, axis=0)  # Mean over samples
    return mean_error

def getMeanMaskedReconstructionEntropy(entropy_tensor, masks_tensor):
    """
    Compute mean and variance of masked reconstruction entropy for each sample.

    Args:
        entropy_tensor: (N, T, D)
        masks_tensor: (N, T, D) with 1/True for masked tokens
    
    Returns:
        MeanMaskedReconstructionEntropy: (T,) mean over samples
        MeanMaskedEntropy: (T,) mean over samples
        VarMaskedReconstructionEntropy: (T,) variance over samples
        VarMaskedEntropy: (T,) variance over samples
    """
    reconstructions = compute_reconstruction_for_each_sample(entropy_tensor)
    N_sample, T_max, D = entropy_tensor.shape
    masks_tensor = np.asarray(masks_tensor, dtype=float)
    if masks_tensor.shape != entropy_tensor.shape:
        raise ValueError(f"masks_tensor shape {masks_tensor.shape} must match entropy_tensor shape {entropy_tensor.shape}")

    # Compute masked mean for each sample at each timestep
    # masked_reconstructions[n, t] = mean of reconstruction[n, t, :] over masked tokens
    masked_reconstructions = reconstructions * masks_tensor
    masked_entropies = entropy_tensor * masks_tensor
    
    # Avoid division by zero
    mask_counts = np.sum(masks_tensor, axis=2)
    mask_counts = np.maximum(mask_counts, 1e-6)
    
    # Mean over masked tokens (per sample per timestep)
    mean_recon_per_sample = np.sum(masked_reconstructions, axis=2) / mask_counts  # (N, T)
    mean_entropy_per_sample = np.sum(masked_entropies, axis=2) / mask_counts  # (N, T)
    
    # Variance over masked tokens (per sample per timestep)
    var_recon_per_sample = np.zeros((N_sample, T_max))
    var_entropy_per_sample = np.zeros((N_sample, T_max))
    
    for n in range(N_sample):
        for t in range(T_max):
            masked_recon_t = reconstructions[n, t, :] * masks_tensor[n, t, :]
            masked_ent_t = entropy_tensor[n, t, :] * masks_tensor[n, t, :]
            mask_t = masks_tensor[n, t, :]
            
            n_masked = np.sum(mask_t)
            if n_masked > 0:
                # Variance of masked values only
                recon_vals = reconstructions[n, t, mask_t > 0]
                ent_vals = entropy_tensor[n, t, mask_t > 0]
                var_recon_per_sample[n, t] = np.var(recon_vals) if len(recon_vals) > 1 else 0
                var_entropy_per_sample[n, t] = np.var(ent_vals) if len(ent_vals) > 1 else 0
    
    # Mean and variance over samples (per timestep)
    MeanMaskedReconstructionEntropy = np.mean(mean_recon_per_sample, axis=0)  # (T,)
    MeanMaskedEntropy = np.mean(mean_entropy_per_sample, axis=0)  # (T,)
    VarMaskedReconstructionEntropy = np.mean(var_recon_per_sample, axis=0)  # (T,)
    VarMaskedEntropy = np.mean(var_entropy_per_sample, axis=0)  # (T,)
    
    return MeanMaskedReconstructionEntropy, MeanMaskedEntropy, VarMaskedReconstructionEntropy, VarMaskedEntropy

OUTPUTS_PATH  = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt'
EVAL_JSON     = 'PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json'


with open(EVAL_JSON, encoding="utf-8") as f:
    data = json.load(f)

outputs = mergeOutputsList(OUTPUTS_PATH)


by_index = {d["index"]: d for d in data}
hmap = {}
for i in range(len(outputs.sample_indices)):
    j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
    hmap[j] = i


correct = [d["index"] for d in data if d["is_hallucination"] == 'no']
correct = [hmap[j] for j in correct if j in hmap]
hallucinations = [d["index"] for d in data if d["is_hallucination"] == 'yes']
hallucinations = [hmap[j] for j in hallucinations if j in hmap]
print('nb correct: ', len(correct))
print('nb hallucinations: ', len(hallucinations))

entropies = getEntropy(outputs)
masks = getEachStepMask(outputs)
entropies_correct = [entropies[i] for i in correct]
entropies_hallucinations = [entropies[i] for i in hallucinations]
masks_correct = [masks[i] for i in correct]
masks_hallucinations = [masks[i] for i in hallucinations]
# transform entropies to shape (N_samples, T_steps, D_tokens)
entropy_tensor_correct = np.array(entropies_correct)  # Assuming entropies is a list of arrays of shape (T_steps, D_tokens) for each sample
entropy_tensor_hallucinations = np.array(entropies_hallucinations)  # Assuming entropies is a list of arrays of shape (T_steps, D_tokens) for each sample
mask_tensor_correct = np.array(masks_correct)
mask_tensor_hallucinations = np.array(masks_hallucinations)
Reconstruct_error_correct = getMeanReconstructionError(entropy_tensor_correct, mask_tensor_correct)
Reconstruct_error_hallucinations = getMeanReconstructionError(entropy_tensor_hallucinations, mask_tensor_hallucinations)

plt.figure(figsize=(15, 6))
plt.plot(Reconstruct_error_correct, color='#e67e22', linewidth=2.5, label='Mean Reconstruction Error (AR(1)) - Correct', marker='o', markersize=4)
plt.plot(Reconstruct_error_hallucinations, color='#3498db', linewidth=2.5, label='Mean Reconstruction Error (AR(1)) - Hallucinations', marker='o', markersize=4)
plt.title("Mean Reconstruction Error Across Samples (AR(1) Model)", fontsize=12, fontweight='bold')
plt.xlabel("Diffusion Step")
plt.ylabel("Mean Absolute Error")
plt.grid(True,)
plt.legend()
plt.tight_layout()
plt.savefig("PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/MeanReconstructionError_AR1.png", dpi=150)
plt.show()



MeanMaskedReconstructionEntropy_correct, MeanMaskedEntropy_correct, VarMaskedReconstructionEntropy_correct, VarMaskedEntropy_correct = getMeanMaskedReconstructionEntropy(entropy_tensor_correct, mask_tensor_correct)
MeanMaskedReconstructionEntropy_hallucinations, MeanMaskedEntropy_hallucinations, VarMaskedReconstructionEntropy_hallucinations, VarMaskedEntropy_hallucinations = getMeanMaskedReconstructionEntropy(entropy_tensor_hallucinations, mask_tensor_hallucinations)

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Graphe 1: Mean Masked Entropy
ax = axes[0]
ax.plot(MeanMaskedReconstructionEntropy_correct, color='#e67e22', linewidth=2.5, label='Mean Masked Reconstruction Entropy (AR(1)) - Correct', marker='o', markersize=4)
ax.plot(MeanMaskedReconstructionEntropy_hallucinations, color='#3498db', linewidth=2.5, label='Mean Masked Reconstruction Entropy (AR(1)) - Hallucinations', marker='o', markersize=4)
ax.plot(MeanMaskedEntropy_correct, color='#f39c12', linewidth=2.5, label='Mean Masked Entropy (Data) - Correct', marker='x', markersize=4)
ax.plot(MeanMaskedEntropy_hallucinations, color='#2980b9', linewidth=2.5, label='Mean Masked Entropy (Data) - Hallucinations', marker='x', markersize=4)
ax.set_title("Mean Masked Entropy Across Samples (AR(1) Model)", fontsize=12, fontweight='bold')
ax.set_xlabel("Diffusion Step")
ax.set_ylabel("Mean Masked Entropy")
ax.grid(True, alpha=0.3)
ax.legend()

# Graphe 2: Variance of Masked Entropy
ax = axes[1]
ax.plot(VarMaskedReconstructionEntropy_correct, color='#e74c3c', linewidth=2.5, label='Variance Masked Reconstruction Entropy (AR(1)) - Correct', marker='o', markersize=4)
ax.plot(VarMaskedReconstructionEntropy_hallucinations, color='#9b59b6', linewidth=2.5, label='Variance Masked Reconstruction Entropy (AR(1)) - Hallucinations', marker='o', markersize=4)
ax.plot(VarMaskedEntropy_correct, color='#e67e22', linewidth=2.5, label='Variance Masked Entropy (Data) - Correct', marker='x', markersize=4)
ax.plot(VarMaskedEntropy_hallucinations, color='#3498db', linewidth=2.5, label='Variance Masked Entropy (Data) - Hallucinations', marker='x', markersize=4)
ax.set_title("Variance Masked Entropy Across Samples (AR(1) Model)", fontsize=12, fontweight='bold')
ax.set_xlabel("Diffusion Step")
ax.set_ylabel("Variance Masked Entropy")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/MeanVarMaskedReconstructionEntropy_AR1.png", dpi=150)
plt.show()  