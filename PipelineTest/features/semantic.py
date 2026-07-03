"""
features.semantic
=================
Semantic token clustering and semantic entropy/dispersion features.
Wraps the existing SemanticEntropy.py and clusteringTokens.py logic.
"""

import numpy as np

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getSemanticEntropy, getSemanticDispersion, getEachStepMask,
)


# =========================================================================
# SEMANTIC VARIANCE FEATURES
# =========================================================================

def var_semantic_entropy(outputs):
    """Variance of semantic entropy across steps (per token), averaged."""
    semantic_entropy = getSemanticEntropy(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(semantic_entropy[i], dtype=float), axis=0)))
        for i in range(len(semantic_entropy))
    ])


def var_semantic_entropy_across_tokens(outputs):
    """Variance of semantic entropy across tokens (per step), averaged."""
    semantic_entropy = getSemanticEntropy(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(semantic_entropy[i], dtype=float), axis=1)))
        for i in range(len(semantic_entropy))
    ])


def var_semantic_entropy_masked(outputs):
    """Variance of semantic entropy across masked steps (per token), averaged."""
    entropies = getSemanticEntropy(outputs)
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


def var_semantic_entropy_masked_across_tokens(outputs):
    """Variance of semantic entropy among masked tokens (per step), averaged."""
    entropies = getSemanticEntropy(outputs)
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


def var_semantic_dispersion(outputs):
    """Variance of semantic dispersion across steps (per token), averaged."""
    disp = getSemanticDispersion(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(disp[i], dtype=float), axis=0)))
        for i in range(len(disp))
    ])


def var_semantic_dispersion_across_tokens(outputs):
    """Variance of semantic dispersion across tokens (per step), averaged."""
    disp = getSemanticDispersion(outputs)
    return np.array([
        float(np.mean(np.var(np.asarray(disp[i], dtype=float), axis=1)))
        for i in range(len(disp))
    ])


# =========================================================================
# COMBINED EXTRACTION
# =========================================================================

def get_semantic_features(outputs):
    """Extract all semantic features.
    
    Returns:
        features: np.ndarray (N, F)
        feature_names: list of str
    """
    feats = {
        "VarSemanticEntropy": var_semantic_entropy(outputs),
        "VarSemanticEntropyAcrossTokens": var_semantic_entropy_across_tokens(outputs),
        "VarSemanticEntropyMasked": var_semantic_entropy_masked(outputs),
        "VarSemanticEntropyMaskedAcrossTokens": var_semantic_entropy_masked_across_tokens(outputs),
        "VarSemanticDispersion": var_semantic_dispersion(outputs),
        "VarSemanticDispersionAcrossTokens": var_semantic_dispersion_across_tokens(outputs),
    }

    feature_names = list(feats.keys())
    features = np.column_stack([feats[k] for k in feature_names])
    return features, feature_names
