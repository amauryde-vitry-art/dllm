import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getLogProbs, getEachStepMask, getEachStepChange, getEntropy
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
import seaborn as sns

import numpy as np
from scipy.stats import norm
try:
    from SemanticEntropy import build_token_to_cluster
except ModuleNotFoundError:
    from PipelineTest.SemanticEntropy import build_token_to_cluster
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEachStepProposedTokenIdSequence, getSemanticDispersion, getSemanticEntropy, getEachStepGeneratedSequence, getEntropyJustUnmasked

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from scipy.stats import skew, kurtosis
from scipy.stats import linregress



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

def _load_outputs(output_path):
    # Lazy import avoids circular dependency with AnalyseResults.
    try:
        from AnalyseResults import mergeOutputsList
    except ModuleNotFoundError:
        from PipelineTest.AnalyseResults import mergeOutputsList

    return mergeOutputsList(output_path)


def getFromOutputsVarEntropy(outputs):
    entropies = getEntropy(outputs)
    var_entropy_all = np.asarray([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=0)))
        for i in range(len(entropies))
    ])
    return var_entropy_all

def getFromOutputVarEntropyAcrossTokens(outputs):
    entropies = getEntropy(outputs)
    var_entropy_all = np.asarray([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=1)))
        for i in range(len(entropies))
    ])
    return var_entropy_all


def getFromOutputsVarEntropyMasked(outputs):
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    var_entropy_all = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        # For each token, compute variance across the steps where it is still masked
        token_vars = []
        for j in range(ent.shape[1]):
            masked_vals = ent[mask[:, j], j]
            if len(masked_vals) > 1:
                token_vars.append(np.var(masked_vals))
        var_entropy_all.append(float(np.mean(token_vars)) if token_vars else 0.0)
    return np.asarray(var_entropy_all)




def getFromOutputsVarSemanticEntropyMasked(outputs):
    entropies = getSemanticEntropy(outputs)
    masks = getEachStepMask(outputs)
    var_entropy_all = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        # For each token, compute variance across the steps where it is still masked
        token_vars = []
        for j in range(ent.shape[1]):
            masked_vals = ent[mask[:, j], j]
            if len(masked_vals) > 1:
                token_vars.append(np.var(masked_vals))
        var_entropy_all.append(float(np.mean(token_vars)) if token_vars else 0.0)
    return np.asarray(var_entropy_all)

def getMeanEntropyJustUnmasked(outputs):
    entropies = getEntropyJustUnmasked(outputs)
    mean_entropy_all = []
    for i in range(len(entropies)):
        values = np.asarray(entropies[i], dtype=float)[1:]
        mean_entropy_all.append(float(np.mean(values)) if values.size > 0 else 0.0)
    mean_entropy_all = np.asarray(mean_entropy_all)
    return mean_entropy_all

def getVarEntropyJustUnmasked(outputs):
    entropies = getEntropyJustUnmasked(outputs)
    var_entropy_all = []
    for i in range(len(entropies)):
        values = np.asarray(entropies[i], dtype=float)[1:]
        var_entropy_all.append(float(np.var(values)))
    var_entropy_all = np.asarray(var_entropy_all)
    return var_entropy_all

def getFromOutputsVarEntropyMaskedAcrossTokens(outputs):
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    var_entropy_all = []
    
        
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        var_entropy_all.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.asarray(var_entropy_all)


def getFromOutputsVarEntropyMaskedAcrossTokensNoPadding(outputs, pad_token_id):
    """Variance of entropy across masked non-padding tokens, at each step, averaged over steps."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask_from_x0(outputs, positions, pad_token_id)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    var_entropy_all = []

    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        pad = padding_mask_all[i, 0, :]                # [tokens] bool, True=padding
        step_vars = []
        for s in range(ent.shape[0]):
            valid = mask[s] & (~pad)
            masked_vals = ent[s, valid]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        var_entropy_all.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.asarray(var_entropy_all)



def getFromOutputsVarSemanticEntropyMaskedAcrossTokens(outputs):
    entropies = getSemanticEntropy(outputs)
    masks = getEachStepMask(outputs)
    var_entropy_all = []
    
        
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        var_entropy_all.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.asarray(var_entropy_all)


def getTauFromEntropy(outputs, window=10):
    entropies = getEntropy(outputs)
    list_mean_tau = []
    list_var_tau = []
    for i in range(len(entropies)):
        # entropies[i] shape: [steps, tokens]
        ent = np.asarray(entropies[i], dtype=float)
        ent = np.mean(ent, axis=1)  # moyenne sur les tokens
        n_steps = len(ent)
        # tau at steps 0, window, 2*window, ... : (entropy[t+window] - entropy[t]) / window
        t_indices = np.arange(0, n_steps - window, window)
        tau = (ent[t_indices + window] - ent[t_indices]) / window  # [n_steps//window]
        list_mean_tau.append(np.mean(tau))
        list_var_tau.append(np.var(tau))
    return list_mean_tau, list_var_tau

def getAlphaAndBetaFromEntropy(outputs):
    entropies = getEntropy(outputs)
    alpha = []
    beta = []
    for i in range(len(entropies)):
        # entropies[i] shape: [steps, tokens]
        ent = np.asarray(entropies[i], dtype=float)
        ln_ent = np.log(ent + 1e-10)
        n_steps, n_tokens = ent.shape
        # fit linear reg ln_ent[t] = alpha * t + beta
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha_i = []
        beta_i = []
        for j in range(n_tokens):
            y = ln_ent[:, j]
            alpha_j, beta_j = np.linalg.lstsq(A, y, rcond=None)[0]
            alpha_i.append(alpha_j)
            beta_i.append(beta_j)
        alpha.append(np.mean(alpha_i))
        beta.append(np.mean(beta_i))
    return alpha, beta


def getAlphaBetaFromEntropyAvg(outputs, k_tokens=1):
    entropies = getEntropy(outputs)
    list_alpha = []
    list_beta = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mean_ent = np.mean(ent, axis=0)
        # selection des top k tokens avec plus grande moyenne
        if k_tokens:
            indices_large_var = np.argsort(mean_ent)[-k_tokens:]
            mean_ent = np.mean(ent[:, indices_large_var], axis=1)
        else:
            mean_ent = np.mean(ent, axis=1)
        ln_mean_ent = np.log(mean_ent + 1e-10)
        n_steps = len(mean_ent)
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha, beta = np.linalg.lstsq(A, ln_mean_ent, rcond=None)[0]
        list_alpha.append(alpha)
        list_beta.append(beta)

    return list_alpha, list_beta




def getInterceptCAndTauFromEntropyAvg(outputs, k_tokens=1):
    """Fit Entropy(t) = C * (t - m) * exp(-(t-m) / tau) on averaged entropy trajectories."""

    def model(t, c, tau, m):
        return c * t * np.exp(-(t - m) / tau)

    entropies = getEntropy(outputs)
    list_c = []
    list_tau = []
    list_m = []

    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mean_ent_per_token = np.mean(ent, axis=0)

        if k_tokens:
            indices_large_var = np.argsort(mean_ent_per_token)[-k_tokens:]
            mean_ent = np.mean(ent[:, indices_large_var], axis=1)
        else:
            mean_ent = np.mean(ent, axis=1)

        t = np.arange(len(mean_ent), dtype=float)
        valid = np.isfinite(mean_ent) & (mean_ent > 0)
        t_fit = t[valid]
        y_fit = mean_ent[valid]

        if len(y_fit) < 3:
            list_c.append(np.nan)
            list_tau.append(np.nan)
            list_m.append(np.nan)
            continue

        c0 = float(max(y_fit.max(), 1e-3))
        tau0 = float(max(len(y_fit) / 4.0, 1.0))
        m0 = float(len(y_fit) / 2.0)

        try:
            params, _ = curve_fit(
                model,
                t_fit,
                y_fit,
                p0=[c0, tau0, m0],
                bounds=([0.0, 1e-6, -len(y_fit)], [np.inf, np.inf, 2*len(y_fit)]),
                maxfev=20000,
            )
            c_i, tau_i, m_i = params
        except Exception:
            c_i, tau_i, m_i = np.nan, np.nan, np.nan

        list_c.append(float(c_i))
        list_tau.append(float(tau_i))
        list_m.append(float(m_i))

    return list_c, list_tau, list_m

def getInterceptCAndTauFromVarMaskedEntropyAcrossTokens(outputs, k_tokens=1):
    """Fit Entropy(t) = C * (t - 53) * exp(-2 (t-53) / tau) on averaged entropy trajectories."""
    
    # Correction 1 : Retrait du paramètre 'm' inutile
    def model(t, c, tau):
        # Attention : si t < 53, (t - 53) est négatif, l'exponentielle exp(-2*(t-53)/tau) EXPLOSE.
        # On s'assure de clipper ou de comprendre que t doit être supérieur à 53.
        with np.errstate(over='ignore', invalid='ignore'):
            val = c * (t - 53) * np.exp(-2 * (t - 53) / tau)
        return np.nan_to_num(val, nan=0.0, posinf=1e10, neginf=-1e10)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)

    list_c = []
    list_tau = []

    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        
        step_vars = []
        actual_steps = []
        
        # Correction 2 : On garde une trace du vrai pas de temps 's'
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
                actual_steps.append(s)

        step_vars = np.array(step_vars)
        t = np.array(actual_steps, dtype=float)
        
        # Filtrage des valeurs valides
        valid = np.isfinite(step_vars) & (step_vars > 0)
        t_fit = t[valid]
        y_fit = step_vars[valid]

        # Inutile de tenter un fit si on n'a pas de données après le step 53 
        # car le modèle y est hautement instable ou négatif.
        valid_post_53 = t_fit > 53
        if np.sum(valid_post_53) < 3:
            list_c.append(np.nan)
            list_tau.append(np.nan)
            continue

        # Optionnel mais recommandé : fitter uniquement là où le modèle fait du sens (t > 53)
        # t_fit = t_fit[valid_post_53]
        # y_fit = y_fit[valid_post_53]

        # Initialisation des guesses
        c0 = float(max(y_fit.max(), 1e-3))
        tau0 = float(max(len(y_fit) / 4.0, 1.0))

        try:
            params, _ = curve_fit(
                model,
                t_fit,
                y_fit,
                p0=[c0, tau0],
                bounds=([0.0, 1e-6], [np.inf, np.inf]),
                maxfev=20000,
            )
            c_i, tau_i = params
        except Exception as e:
            # En cas d'erreur, on peut print(e) pour débugger si besoin
            c_i, tau_i = np.nan, np.nan

        list_c.append(float(c_i))
        list_tau.append(float(tau_i))

    return list_c, list_tau


def getAlphaBetaFromEntropyAvgStudy(outputs):
    entropies = getEntropy(outputs)
    alpha_for_top_k = {}
    beta_for_top_k = {}
    for k_tokens in range(1, 65):
        list_alpha = []
        list_beta = []
        for i in range(len(entropies)):
                
            ent = np.asarray(entropies[i], dtype=float)
            mean_ent_across_tokens = np.mean(ent, axis=0)
            # selection des top k indices avec plus grande moyenne
            indices_large_entropy_across_tokens = np.argsort(mean_ent_across_tokens)[-k_tokens:]
            

            mean_ent = np.mean(ent[:, indices_large_entropy_across_tokens], axis=1)
            ln_mean_ent = np.log(mean_ent + 1e-10)
            n_steps = len(mean_ent)
            t = np.arange(n_steps)
            A = np.vstack([t, np.ones(n_steps)]).T
            alpha, beta = np.linalg.lstsq(A, ln_mean_ent, rcond=None)[0]
            list_alpha.append(alpha)
            list_beta.append(beta)

        alpha_for_top_k[k_tokens] = list_alpha
        beta_for_top_k[k_tokens] = list_beta
    return alpha_for_top_k, beta_for_top_k


def getAlphaBetaFromEntropyAvgStudyv2(outputs):
    entropies = getEntropy(outputs)
    alpha_for_top_k = {}
    beta_for_top_k = {}
    
    for k_tokens in range(1, 65):
        list_alpha = []
        list_beta = []
        
        for i in range(len(entropies)):
            ent = np.asarray(entropies[i], dtype=float)  # Shape attendue : [n_steps, seq_len]
            
            mean_ent_across_tokens = np.mean(ent, axis=0)
            indices_large_entropy_across_tokens = np.argsort(mean_ent_across_tokens)[-k_tokens:]

            mean_ent_across_diff = np.mean(ent, axis=1)
            indices_large_entropy_across_diff = np.argsort(mean_ent_across_diff)[-k_tokens:]

            sub_matrix_indices = np.ix_(indices_large_entropy_across_diff, indices_large_entropy_across_tokens)
            sub_matrix = ent[sub_matrix_indices]  # Shape : [k_tokens, k_tokens]
            
            mean_ent = np.mean(sub_matrix, axis=1)
            
            ln_mean_ent = np.log(mean_ent + 1e-10)
            n_steps = len(mean_ent)  # Ici n_steps sera égal à k_tokens
            
            t = np.arange(n_steps)
            A = np.vstack([t, np.ones(n_steps)]).T
            
            alpha, beta = np.linalg.lstsq(A, ln_mean_ent, rcond=None)[0]
            list_alpha.append(alpha)
            list_beta.append(beta)

        alpha_for_top_k[k_tokens] = list_alpha
        beta_for_top_k[k_tokens] = list_beta
        
    return alpha_for_top_k, beta_for_top_k

def getAlphaAndBetaFromLogProbs(outputs):
    logprobs = getLogProbs(outputs)
    alpha = []
    beta = []
    for i in range(len(logprobs)):
        # logprobs[i] shape: [steps, tokens]
        logp = np.asarray(logprobs[i], dtype=float)
        ln_logp = np.log(logp + 1e-10)
        n_steps, n_tokens = logp.shape
        # fit linear reg ln_logp[t] = alpha * t + beta
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha_i = []
        beta_i = []
        for j in range(n_tokens):
            y = ln_logp[:, j]
            alpha_j, beta_j = np.linalg.lstsq(A, y, rcond=None)[0]
            alpha_i.append(alpha_j)
            beta_i.append(beta_j)
        alpha.append(np.mean(alpha_i))
        beta.append(np.mean(beta_i))
    return alpha, beta

def getAlphaBetaFromLogProbsAvg(outputs):
    logprobs = getLogProbs(outputs)
    list_alpha = []
    list_beta = []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        mean_logp = np.mean(logp, axis=1)
        ln_mean_logp = np.log(mean_logp + 1e-10)
        n_steps = len(mean_logp)
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha, beta = np.linalg.lstsq(A, ln_mean_logp, rcond=None)[0]
        list_alpha.append(alpha)
        list_beta.append(beta)

    return list_alpha, list_beta

def getTauFromLogProb(outputs, window=10):
    logprobs = getLogProbs(outputs)
    list_mean_tau = []
    list_var_tau = []
    for i in range(len(logprobs)):
        # logprobs[i] shape: [steps, tokens]
        logp = np.asarray(logprobs[i], dtype=float)
        n_steps, n_tokens = logp.shape
        # tau at steps 0, window, 2*window, ... : (logprob[t+window] - logprob[t]) / window
        t_indices = np.arange(0, n_steps - window, window)
        tau = (logp[t_indices + window, :] - logp[t_indices, :]) / window  # [n_steps//window, tokens]
        list_mean_tau.append(np.mean(tau))
        list_var_tau.append(np.mean(np.var(tau, axis=1)))
    return list_mean_tau, list_var_tau
    

def fromOutputsVarLogprobs(outputs):
    logprobs = getLogProbs(outputs)
    var_logprobs_all = np.asarray([
        float(np.mean(np.var(np.asarray(logprobs[i], dtype=float), axis=0)))
        for i in range(len(logprobs))
    ])
    return var_logprobs_all



def fromOutputsGetMeanEntropy(outputs):
    entropies = getEntropy(outputs)

    mean_entropy_sequence_all = []
    for i in range(len(entropies)):
        mean_entropy_sequence = np.mean(np.asarray(entropies[i], dtype=float))
        mean_entropy_sequence_all.append(mean_entropy_sequence)
    mean_entropy_sequence_all = np.asarray(mean_entropy_sequence_all)
    return mean_entropy_sequence_all


def fromOutputsGetMeanEntropyMasked(outputs):
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    mean_entropy_sequence_all = []
    for i in range(len(entropies)):
        masked_entropy = np.asarray(entropies[i], dtype=float) * np.asarray(masks[i], dtype=float)
        mean_entropy_sequence = np.mean(masked_entropy)
        mean_entropy_sequence_all.append(mean_entropy_sequence)
    mean_entropy_sequence_all = np.asarray(mean_entropy_sequence_all)
    return mean_entropy_sequence_all

def fromOutputsGetMeanEntropyMaskedNoPadding(outputs, pad_token_id):
    """Mean entropy over masked non-padding tokens only."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask_from_x0(outputs, positions, pad_token_id)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    mean_entropy_sequence_all = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)    # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)         # [steps, tokens]
        pad = padding_mask_all[i, 0, :]                 # [tokens] bool, True=padding
        valid = mask & (~pad[np.newaxis, :])
        vals = ent[valid]
        mean_entropy_sequence_all.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    mean_entropy_sequence_all = np.asarray(mean_entropy_sequence_all)
    return mean_entropy_sequence_all

def fromOutputsGetMeanLogprob(outputs):
    logprobs = getLogProbs(outputs)
    mean_logprob_sequence_all = []
    for i in range(len(logprobs)):
        mean_logprob_sequence = np.mean(np.asarray(logprobs[i], dtype=float))
        mean_logprob_sequence_all.append(mean_logprob_sequence)
    mean_logprob_sequence_all = np.asarray(mean_logprob_sequence_all)
    return mean_logprob_sequence_all



def getLabels(eval_json_path=None):
    if eval_json_path is None:
        eval_json_path = os.path.join(os.path.dirname(__file__), "res", "eval", "results_triviaqa_semantic_dispersion_and_semantic_entropy_3k_stc_2048_samples.json")
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    data.sort(key=lambda x: x["index"])
    labels = np.array([1 if d["is_hallucination"] == "yes" else 0 for d in data], dtype=np.int64)
    return labels

def fromOutputsGetVarSemanticDispersion(output_path):
    outputs = _load_outputs(output_path)
    semantic_dispersion = getSemanticDispersion(outputs)
    var_semantic_dispersion_all = np.asarray([
        float(np.mean(np.var(np.asarray(semantic_dispersion[i], dtype=float), axis=0)))
        for i in range(len(semantic_dispersion))
    ])
    return var_semantic_dispersion_all
    

def fromOutputsGetVarSemanticDispersionAcrossToken(output_path):
    outputs = _load_outputs(output_path)
    semantic_dispersion = getSemanticDispersion(outputs)
    var_semantic_dispersion_all = np.asarray([
        float(np.mean(np.var(np.asarray(semantic_dispersion[i], dtype=float), axis=1)))
        for i in range(len(semantic_dispersion))
    ])
    return var_semantic_dispersion_all
    

def fromOutputsGetVarSemanticEntropy(output_path):
    outputs = _load_outputs(output_path)
    semantic_entropy = getSemanticEntropy(outputs)
    var_semantic_entropy_all = np.asarray([
        float(np.mean(np.var(np.asarray(semantic_entropy[i], dtype=float), axis=0)))
        for i in range(len(semantic_entropy))
    ])
    return var_semantic_entropy_all


def fromOutputsGetVarSemanticEntropyAcrossToken(output_path):
    outputs = _load_outputs(output_path)
    semantic_entropy = getSemanticEntropy(outputs)
    var_semantic_entropy_all = np.asarray([
        float(np.mean(np.var(np.asarray(semantic_entropy[i], dtype=float), axis=1)))
        for i in range(len(semantic_entropy))
    ])
    return var_semantic_entropy_all


def getMaskedEntropyAcrossTokens(outputs):
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    var_entropy_all = []
    
        
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)   # [steps, tokens]
        mask = np.asarray(masks[i], dtype=bool)        # [steps, tokens]
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        var_entropy_all.append(step_vars)

    return var_entropy_all


def getAUCMaskedEntropyAcrossTokens(outputs):
    MaskedEntropyAcrossTokens = getMaskedEntropyAcrossTokens(outputs)
    aucs = []
    for i in range(len(MaskedEntropyAcrossTokens)):
        auc = integrate.trapezoid(MaskedEntropyAcrossTokens[i])
        aucs.append(auc)
    return np.asarray(aucs)


def getARGMAXMaskedEntropyAcrossTokens(outputs):
    MaskedEntropyAcrossTokens = getMaskedEntropyAcrossTokens(outputs)
    argmaxs = []
    for i in range(len(MaskedEntropyAcrossTokens)):
        argmax = np.argmax(MaskedEntropyAcrossTokens[i])
        argmaxs.append(argmax)
    return np.asarray(argmaxs)

def getMAXMaskedEntropyAcrossTokens(outputs):
    MaskedEntropyAcrossTokens = getMaskedEntropyAcrossTokens(outputs)
    maxs = []
    for i in range(len(MaskedEntropyAcrossTokens)):
        max_val = np.max(MaskedEntropyAcrossTokens[i])
        maxs.append(max_val)
    return np.asarray(maxs)


def getKurtosisskedEntropyAcrossTokens(outputs):
    MaskedEntropyAcrossTokens = getMaskedEntropyAcrossTokens(outputs)
    kurtosis_vals = []
    for i in range(len(MaskedEntropyAcrossTokens)):
        kurt = kurtosis(MaskedEntropyAcrossTokens[i])
        kurtosis_vals.append(kurt)
    return np.asarray(kurtosis_vals)


def getSkewnessMaskedEntropyAcrossTokens(outputs):
    MaskedEntropyAcrossTokens = getMaskedEntropyAcrossTokens(outputs)
    skewness_vals = []
    for i in range(len(MaskedEntropyAcrossTokens)):
        skew_val = skew(MaskedEntropyAcrossTokens[i])
        skewness_vals.append(skew_val)
    return np.asarray(skewness_vals)

def getMeanCurvatureMaskedEntropyAcrossTokens(outputs):
    MaskedEntropyAcrossTokens = getMaskedEntropyAcrossTokens(outputs)
    mean_curvature_vals = []
    for i in range(len(MaskedEntropyAcrossTokens)):
        y = np.asarray(MaskedEntropyAcrossTokens[i])
        x = np.arange(len(y))
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        curvature = np.abs(d2y) / (1 + dy**2)**(3/2)
        mean_curvature = np.mean(curvature)
        mean_curvature_vals.append(mean_curvature)
    return np.asarray(mean_curvature_vals)

    
   
# def getTestSimpleFeatures(output_path, eval_json_path, k_tokens=None):
#     outputs = _load_outputs(output_path)
    
#     VarEntropy = getFromOutputsVarEntropy(outputs)
#     VarEntropyAcrossTokens = getFromOutputVarEntropyAcrossTokens(outputs)



#     VarMaskedEntropy = getFromOutputsVarEntropyMasked(outputs)
#     VarMaskedEntropyAcrossTokens = getFromOutputsVarEntropyMaskedAcrossTokens(outputs)

#     VarSemanticEntropy = fromOutputsGetVarSemanticEntropy(output_path)
#     VarSemanticEntropyAcrossToken = fromOutputsGetVarSemanticEntropyAcrossToken(output_path)

#     VarSemanticDispersion = fromOutputsGetVarSemanticDispersion(output_path)
#     VarSemanticDispersionAcrossToken = fromOutputsGetVarSemanticDispersionAcrossToken(output_path)
#     varLogProbs = fromOutputsVarLogprobs(outputs)



#     VarSemanticEntropyMasked = getFromOutputsVarSemanticEntropyMasked(outputs)
#     VarSemanticEntropyMaskedAcrossTokens = getFromOutputsVarSemanticEntropyMaskedAcrossTokens(outputs)
    
#     VarEntropyJustUnmasked = getVarEntropyJustUnmasked(outputs)



#     MeanEntropyJustUnmasked = getMeanEntropyJustUnmasked(outputs)

#     meanLogProbs = fromOutputsGetMeanLogprob(outputs)
#     meanEntropy = fromOutputsGetMeanEntropy(outputs)
#     meanMaskedEntropy = fromOutputsGetMeanEntropyMasked(outputs)

#     mean_tau, var_tau = getTauFromEntropy(outputs, window=10)

#     alphaEntropy, betaEntropy = getAlphaAndBetaFromEntropy(outputs)
#     alphaLogProbs, betaLogProbs = getAlphaAndBetaFromLogProbs(outputs)

#     alphaEntropyAvg, betaEntropyAvg = getAlphaBetaFromEntropyAvg(outputs, k_tokens=k_tokens)
#     alphaLogProbsAvg, betaLogProbsAvg = getAlphaBetaFromLogProbsAvg(outputs, )

#     C_ct, tau_ct, m_ct = getInterceptCAndTauFromEntropyAvg(outputs, k_tokens=20)
#     C_ct_varMaskedEntropy, tau_ct_varMaskedEntropy, m_ct_varMaskedEntropy = getInterceptCAndTauFromVarMaskedEntropyAcrossTokens(outputs, k_tokens=20)

#     AUCMaskedEntropyAcrossTokens = getAUCMaskedEntropyAcrossTokens(outputs)
#     MAXMaskedEntropyAcrossTokens = getMAXMaskedEntropyAcrossTokens(outputs)
#     ArgmaxMaskedEntropyAcrossTokens = getARGMAXMaskedEntropyAcrossTokens(outputs)
#     kurtosisMaskedEntropyAcrossTokens = getKurtosisskedEntropyAcrossTokens(outputs)
#     skewnessMaskedEntropyAcrossTokens = getSkewnessMaskedEntropyAcrossTokens(outputs)
#     meanCurvatureMaskedEntropyAcrossTokens = getMeanCurvatureMaskedEntropyAcrossTokens(outputs)


#     labels = getLabels(eval_json_path)
#     labels = labels[outputs.sample_indices.numpy()] if outputs.sample_indices is not None else labels
    
#     # return np.stack([VarEntropy,VarSemanticEntropy, VarEntropy+VarSemanticEntropy,  VarSemanticDispersion, MeanEntropy, meanLogProbs, varLogProbs, mean_tau, var_tau, alphaEntropy, betaEntropy, alphaLogProbs, betaLogProbs, alphaEntropyAvg, betaEntropyAvg], axis=1), [ "VarEntropy", "VarSemanticEntropy","VarEntropy+VarSemanticEntropy", "VarSemanticDispersion", "MeanEntropy", "MeanLogProbs","VarLogProbs", "MeanTau", "VarTau", "AlphaEntropy", "BetaEntropy", "AlphaLogProbs", "BetaLogProbs", "AlphaEntropyAvg", "BetaEntropyAvg"], labels
#     return np.stack([VarEntropy, VarEntropyAcrossTokens, VarMaskedEntropy,VarMaskedEntropyAcrossTokens, VarSemanticEntropy, VarSemanticEntropyAcrossToken, VarSemanticDispersion, VarSemanticDispersionAcrossToken, varLogProbs, meanLogProbs, meanEntropy, mean_tau, var_tau, alphaLogProbs, betaLogProbs, alphaEntropy, betaEntropy,  alphaEntropyAvg, betaEntropyAvg, alphaLogProbsAvg, betaLogProbsAvg, C_ct, tau_ct, m_ct, VarEntropyJustUnmasked, MeanEntropyJustUnmasked, meanMaskedEntropy, VarSemanticEntropyMasked, VarSemanticEntropyMaskedAcrossTokens, C_ct_varMaskedEntropy, tau_ct_varMaskedEntropy, m_ct_varMaskedEntropy, AUCMaskedEntropyAcrossTokens, MAXMaskedEntropyAcrossTokens, ArgmaxMaskedEntropyAcrossTokens, kurtosisMaskedEntropyAcrossTokens, skewnessMaskedEntropyAcrossTokens, meanCurvatureMaskedEntropyAcrossTokens], axis=1), [ "VarEntropy", "VarEntropyAcrossTokens", "VarMaskedEntropy", "VarMaskedEntropyAcrossTokens", "VarSemanticEntropy", "VarSemanticEntropyAcrossToken", "VarSemanticDispersion", "VarSemanticDispersionAcrossToken", "VarLogProbs", "MeanLogProbs", "MeanEntropy", "MeanTau", "VarTau", "AlphaLogProbs", "BetaLogProbs", "AlphaEntropy", "BetaEntropy",  "AlphaEntropyAvg", "BetaEntropyAvg", "AlphaLogProbsAvg", "BetaLogProbsAvg", "C_ct", "Tau_ct", "M_ct", "VarEntropyJustUnmasked", "MeanEntropyJustUnmasked", "MeanMaskedEntropy", "VarSemanticEntropyMasked", "VarSemanticEntropyMaskedAcrossTokens", "C_ct_varMaskedEntropy", "tau_ct_varMaskedEntropy", "m_ct_varMaskedEntropy", "AUCMaskedEntropyAcrossTokens", "MAXMaskedEntropyAcrossTokens", "ArgmaxMaskedEntropyAcrossTokens", "kurtosisMaskedEntropyAcrossTokens", "skewnessMaskedEntropyAcrossTokens", "meanCurvatureMaskedEntropyAcrossTokens"], labels
#     # return np.stack([VarEntropy,VarSemanticEntropy, meanLogProbs, meanEntropy, mean_tau, var_tau, alphaLogProbs, betaLogProbs, alphaEntropy, betaEntropy,  alphaEntropyAvg, betaEntropyAvg, alphaLogProbsAvg, betaLogProbsAvg], axis=1), [ "VarEntropy", "VarSemanticEntropy", "VarSemanticDispersion", "VarLogProbs", "MeanLogProbs", "MeanEntropy", "MeanTau", "VarTau", "AlphaLogProbs", "BetaLogProbs", "AlphaEntropy", "BetaEntropy",  "AlphaEntropyAvg", "BetaEntropyAvg", "AlphaLogProbsAvg", "BetaLogProbsAvg"], labels

    # return np.stack([VarEntropy, VarLogprobs, VarSemanticDispersion, VarSemanticEntropy, MeanLogProbs], axis=1), ["VarEntropy", "VarLogprobs", "VarSemanticDispersion", "VarSemanticEntropy", "MeanLogProbs",]
    # return np.stack([VarEntropy, VarLogprobs,VarSemanticDispersion, VarSemanticEntropy, VarSkewnessEntropy, MeanSemanticEntropy, MeanLogProbs, MeanEntropy], axis=1), ["VarEntropy", "VarLogProbs", "VarSemanticDispersion", "VarSemanticEntropy", "VarSkewnessEntropy", "MeanSemanticEntropy", "MeanLogProbs", "MeanEntropy"]



   
def _get_pad_token_id_from_output_path(output_path):
    """Derive pad_token_id from the tokenizer file next to the outputs file."""
    import torch
    dir_path = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    tokenizer_name = basename.replace("outputs_", "tokenizer_")
    tokenizer_path = os.path.join(dir_path, tokenizer_name)
    tokenizer = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    return tokenizer.pad_token_id


def getTestSimpleFeatures(output_path, eval_json_path, k_tokens=None):
    outputs = _load_outputs(output_path)
    
   
    VarMaskedEntropyAcrossTokens = getFromOutputsVarEntropyMaskedAcrossTokens(outputs)

   
    meanMaskedEntropy = fromOutputsGetMeanEntropyMasked(outputs)

   

    labels = getLabels(eval_json_path)
    labels = labels[outputs.sample_indices.numpy()] if outputs.sample_indices is not None else labels
    return np.stack([VarMaskedEntropyAcrossTokens, meanMaskedEntropy], axis=1), ["VarMaskedEntropyAcrossTokens", "meanMaskedEntropy"], labels
    # return np.stack([VarEntropy,VarSemanticEntropy, VarEntropy+VarSemanticEntropy,  VarSemanticDispersion, MeanEntropy, meanLogProbs, varLogProbs, mean_tau, var_tau, alphaEntropy, betaEntropy, alphaLogProbs, betaLogProbs, alphaEntropyAvg, betaEntropyAvg], axis=1), [ "VarEntropy", "VarSemanticEntropy","VarEntropy+VarSemanticEntropy", "VarSemanticDispersion", "MeanEntropy", "MeanLogProbs","VarLogProbs", "MeanTau", "VarTau", "AlphaEntropy", "BetaEntropy", "AlphaLogProbs", "BetaLogProbs", "AlphaEntropyAvg", "BetaEntropyAvg"], labels
    # return np.stack([VarEntropy, VarEntropyAcrossTokens, VarMaskedEntropy, VarMaskedEntropyAcrossTokens, varLogProbs, meanLogProbs, meanEntropy, mean_tau, var_tau, alphaLogProbs, betaLogProbs, alphaEntropy, betaEntropy, alphaEntropyAvg, betaEntropyAvg, alphaLogProbsAvg, betaLogProbsAvg, C_ct, tau_ct, m_ct, VarEntropyJustUnmasked, MeanEntropyJustUnmasked, meanMaskedEntropy, C_ct_varMaskedEntropy, tau_ct_varMaskedEntropy,  AUCMaskedEntropyAcrossTokens, MAXMaskedEntropyAcrossTokens, ArgmaxMaskedEntropyAcrossTokens, kurtosisMaskedEntropyAcrossTokens, skewnessMaskedEntropyAcrossTokens, meanCurvatureMaskedEntropyAcrossTokens], axis=1), ["VarEntropy", "VarEntropyAcrossTokens", "VarMaskedEntropy", "VarMaskedEntropyAcrossTokens", "VarLogProbs", "MeanLogProbs", "MeanEntropy", "MeanTau", "VarTau", "AlphaLogProbs", "BetaLogProbs", "AlphaEntropy", "BetaEntropy", "AlphaEntropyAvg", "BetaEntropyAvg", "AlphaLogProbsAvg", "BetaLogProbsAvg", "C_ct", "Tau_ct", "M_ct", "VarEntropyJustUnmasked", "MeanEntropyJustUnmasked", "MeanMaskedEntropy", "C_ct_varMaskedEntropy", "tau_ct_varMaskedEntropy", "AUCMaskedEntropyAcrossTokens", "MAXMaskedEntropyAcrossTokens", "ArgmaxMaskedEntropyAcrossTokens", "kurtosisMaskedEntropyAcrossTokens", "skewnessMaskedEntropyAcrossTokens", "meanCurvatureMaskedEntropyAcrossTokens"], labels
    # return np.stack([VarEntropy,VarSemanticE




def getTestSimpleFeaturesNoPadding(output_path, eval_json_path, k_tokens=None):
    outputs = _load_outputs(output_path)
    pad_id = _get_pad_token_id_from_output_path(output_path)
    VarMaskedEntropyAcrossTokensNoPadding = getFromOutputsVarEntropyMaskedAcrossTokensNoPadding(outputs, pad_id)

    
    meanMaskedEntropyNoPadding = fromOutputsGetMeanEntropyMaskedNoPadding(outputs, pad_id)


    labels = getLabels(eval_json_path)
    labels = labels[outputs.sample_indices.numpy()] if outputs.sample_indices is not None else labels
    return np.stack([VarMaskedEntropyAcrossTokensNoPadding, meanMaskedEntropyNoPadding], axis=1), ["VarMaskedEntropyAcrossTokensNoPadding", "meanMaskedEntropyNoPadding"], labels


def getTestSimpleFeaturesEvaluation(output_path, eval_json_path, k_tokens=None):
    outputs = _load_outputs(output_path)

    VarEntropyJustUnmasked = getVarEntropyJustUnmasked(outputs)
    AlphaEntropyAvg, BetaEntropyAvg = getAlphaBetaFromEntropyAvg(outputs, k_tokens=k_tokens)
    VarMaskedEntropyAcrossTokens = getFromOutputsVarEntropyMaskedAcrossTokens(outputs)
    AlphaEntropy, BetaEntropy = getAlphaAndBetaFromEntropy(outputs)
    _, _, m_ct = getInterceptCAndTauFromEntropyAvg(outputs, k_tokens=20)
    VarSemanticEntropyAcrossToken = fromOutputsGetVarSemanticEntropyAcrossToken(output_path)
    MeanTau, _ = getTauFromEntropy(outputs, window=10)
    AlphaLogProbs , _ = getAlphaAndBetaFromLogProbs(outputs)
    MAXMaskedEntropyAcrossTokens = getMAXMaskedEntropyAcrossTokens(outputs)
    ArgmaxMaskedEntropyAcrossTokens = getARGMAXMaskedEntropyAcrossTokens(outputs)
    

    pad_token_id = _get_pad_token_id_from_output_path(output_path)

    VarMaskedEntropyAcrossTokensNoPadding = getFromOutputsVarEntropyMaskedAcrossTokensNoPadding(outputs, pad_token_id)
    meanMaskedEntropyNoPadding = fromOutputsGetMeanEntropyMaskedNoPadding(outputs, pad_token_id)


    labels = getLabels(eval_json_path)
    labels = labels[outputs.sample_indices.numpy()] if outputs.sample_indices is not None else labels
    
    
    
    return np.stack([VarEntropyJustUnmasked, AlphaEntropyAvg, BetaEntropyAvg, VarMaskedEntropyAcrossTokens,VarMaskedEntropyAcrossTokensNoPadding, meanMaskedEntropyNoPadding, AlphaEntropy, BetaEntropy, m_ct, VarSemanticEntropyAcrossToken, MeanTau, AlphaLogProbs, MAXMaskedEntropyAcrossTokens, ArgmaxMaskedEntropyAcrossTokens], axis=1), ["VarEntropyJustUnmasked", "AlphaEntropyAvg", "BetaEntropyAvg", "VarMaskedEntropyAcrossTokens", "VarMaskedEntropyAcrossTokensNoPadding", "MeanMaskedEntropyNoPadding", "AlphaEntropy", "BetaEntropy", "M_ct", "VarSemanticEntropyAcrossToken", "MeanTau", "AlphaLogProbs", "MAXMaskedEntropyAcrossTokens", "ArgmaxMaskedEntropyAcrossTokens"], labels

    