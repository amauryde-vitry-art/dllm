"""
features.baseline
=================
Baseline (non-dynamic) features: mean and variance of entropy/logprob.
These are simple aggregations over the diffusion trajectory.
"""

import numpy as np

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getEntropy, getEachStepMask, getLogProbs, getEntropyJustUnmasked,
)
from PipelineTest.features.utils import compute_padding_mask, get_pad_token_id


# =========================================================================
# MEAN FEATURES
# =========================================================================

def mean_entropy(outputs):
    """Mean entropy across all steps and tokens."""
    entropies = getEntropy(outputs)
    return np.array([
        float(np.mean(np.asarray(entropies[i], dtype=float)))
        for i in range(len(entropies))
    ])


def mean_masked_entropy(outputs):
    """Mean entropy restricted to masked positions."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=float)
        masked_entropy = ent * mask
        result.append(float(np.sum(masked_entropy) / max(np.sum(mask), 1)))
    return np.array(result)


def mean_masked_entropy_no_padding(outputs, pad_token_id):
    """Mean entropy over masked non-padding tokens only."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask(outputs, positions, pad_token_id)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        pad = padding_mask_all[i, 0, :]
        valid = mask & (~pad[np.newaxis, :])
        vals = ent[valid]
        result.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    return np.array(result)

def mean_entropy_just_unmasked(outputs):
    """Mean entropy at the step each token is unmasked (Flat global average)."""
    entropies = getEntropyJustUnmasked(outputs) # Contient maintenant une liste de tableaux 1D propres
    result = []
    for i in range(len(entropies)):
        vals = entropies[i]
        
        # Plus besoin de slice [1:] car le tableau ne contient aucun placeholder d'étape vide.
        # np.mean(vals) fait maintenant la vraie moyenne à plat : Somme(entropies) / 32
        result.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
    return np.array(result)


def mean_logprob(outputs):
    """Mean log-probability across all steps and tokens."""
    logprobs = getLogProbs(outputs)
    return np.array([
        float(np.mean(np.asarray(logprobs[i], dtype=float)))
        for i in range(len(logprobs))
    ])


# =========================================================================
# VARIANCE FEATURES
# =========================================================================

def var_entropy(outputs):
    """Variance of entropy across steps (per token), averaged over tokens."""
    entropies = getEntropy(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=0)))
        for i in range(len(entropies))
    ])


def var_entropy_across_tokens(outputs):
    """Variance of entropy across tokens (per step), averaged over steps."""
    entropies = getEntropy(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=1)))
        for i in range(len(entropies))
    ])


def var_masked_entropy(outputs):
    """Variance of entropy across masked steps only (per token), averaged."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        token_vars = []
        for j in range(ent.shape[1]):
            masked_vals = ent[mask[:, j], j]
            if len(masked_vals) > 1:
                token_vars.append(np.var(masked_vals))
        result.append(float(np.mean(token_vars)) if token_vars else 0.0)
    return np.array(result)


def var_masked_entropy_across_tokens(outputs):
    """Variance of entropy among masked tokens at each step, averaged over steps."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        result.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.array(result)


def var_masked_entropy_across_tokens_no_padding(outputs, pad_token_id):
    """Variance of entropy across masked non-padding tokens, averaged over steps."""
    positions = list(range(len(outputs.sample_indices)))
    padding_mask_all = compute_padding_mask(outputs, positions, pad_token_id)

    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    result = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        pad = padding_mask_all[i, 0, :]
        step_vars = []
        for s in range(ent.shape[0]):
            valid = mask[s] & (~pad)
            masked_vals = ent[s, valid]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        result.append(float(np.mean(step_vars)) if step_vars else 0.0)
    return np.array(result)


def var_entropy_just_unmasked(outputs):
    """Variance of entropy at unmasking across steps."""
    entropies = getEntropyJustUnmasked(outputs)
    result = []
    for i in range(len(entropies)):
        values = np.asarray(entropies[i], dtype=float)[1:]
        result.append(float(np.var(values)))
    return np.array(result)


def var_logprob(outputs):
    """Variance of log-probabilities across steps (per token), averaged."""
    logprobs = getLogProbs(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(logprobs[i], dtype=float), axis=0)))
        for i in range(len(logprobs))
    ])


# =========================================================================
# COMBINED EXTRACTION
# =========================================================================

def get_baseline_features(outputs, pad_token_id=None):
    """Extract all baseline features.
    
    Args:
        outputs: merged sampler outputs
        pad_token_id: if provided, also computes no-padding variants
        
    Returns:
        features: np.ndarray (N, F)
        feature_names: list of str
    """
    feats = {
        "MeanEntropy": mean_entropy(outputs),
        "MeanMaskedEntropy": mean_masked_entropy(outputs),
        "MeanEntropyJustUnmasked": mean_entropy_just_unmasked(outputs),
        "MeanLogProb": mean_logprob(outputs),
        "VarEntropy": var_entropy(outputs),
        "VarEntropyAcrossTokens": var_entropy_across_tokens(outputs),
        "VarMaskedEntropy": var_masked_entropy(outputs),
        "VarMaskedEntropyAcrossTokens": var_masked_entropy_across_tokens(outputs),
        "VarEntropyJustUnmasked": var_entropy_just_unmasked(outputs),
        "VarLogProb": var_logprob(outputs),
    }

    if pad_token_id is not None:
        feats["MeanMaskedEntropyNoPad"] = mean_masked_entropy_no_padding(outputs, pad_token_id)
        feats["VarMaskedEntropyAcrossTokensNoPad"] = var_masked_entropy_across_tokens_no_padding(outputs, pad_token_id)

    feature_names = list(feats.keys())
    features = np.column_stack([feats[k] for k in feature_names])
    return features, feature_names
