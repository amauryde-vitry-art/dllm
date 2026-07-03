"""
features.markovian
==================
Dynamic features motivated by the Markovian/OU modelisation of entropy decay.

Features:
- alpha/beta from exponential decay fit: ln H(t) ~ alpha*t + beta
- Top-k averaged decay
- Parametric trajectory fit (C, tau, m)
- Finite-difference rate of change (MeanTau, VarTau)
- Shape features on the masked variance curve (AUC, max, argmax, skewness, kurtosis, curvature)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from scipy import integrate

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getEntropy, getEachStepMask, getLogProbs,
)


# =========================================================================
# EXPONENTIAL DECAY FIT (alpha, beta)
# =========================================================================

def alpha_beta_entropy(outputs):
    """Fit ln H(t,d) = alpha_d * t + beta_d per token, return mean alpha, mean beta."""
    entropies = getEntropy(outputs)
    alpha_list, beta_list = [], []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        ln_ent = np.log(ent + 1e-10)
        n_steps, n_tokens = ent.shape
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha_i, beta_i = [], []
        for j in range(n_tokens):
            a, b = np.linalg.lstsq(A, ln_ent[:, j], rcond=None)[0]
            alpha_i.append(a)
            beta_i.append(b)
        alpha_list.append(np.mean(alpha_i))
        beta_list.append(np.mean(beta_i))
    return np.array(alpha_list), np.array(beta_list)


def alpha_beta_entropy_avg(outputs, k_tokens=1):
    """Fit on top-k tokens (by mean entropy) averaged trajectory."""
    entropies = getEntropy(outputs)
    alpha_list, beta_list = [], []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mean_ent = np.mean(ent, axis=0)
        if k_tokens:
            indices = np.argsort(mean_ent)[-k_tokens:]
            mean_ent_traj = np.mean(ent[:, indices], axis=1)
        else:
            mean_ent_traj = np.mean(ent, axis=1)
        ln_mean_ent = np.log(mean_ent_traj + 1e-10)
        n_steps = len(mean_ent_traj)
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        a, b = np.linalg.lstsq(A, ln_mean_ent, rcond=None)[0]
        alpha_list.append(a)
        beta_list.append(b)
    return np.array(alpha_list), np.array(beta_list)


def alpha_beta_logprob(outputs):
    """Fit ln logprob(t,d) = alpha_d * t + beta_d per token."""
    logprobs = getLogProbs(outputs)
    alpha_list, beta_list = [], []
    for i in range(len(logprobs)):
        logp = np.asarray(logprobs[i], dtype=float)
        ln_logp = np.log(logp + 1e-10)
        n_steps, n_tokens = logp.shape
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        alpha_i, beta_i = [], []
        for j in range(n_tokens):
            a, b = np.linalg.lstsq(A, ln_logp[:, j], rcond=None)[0]
            alpha_i.append(a)
            beta_i.append(b)
        alpha_list.append(np.mean(alpha_i))
        beta_list.append(np.mean(beta_i))
    return np.array(alpha_list), np.array(beta_list)


# =========================================================================
# PARAMETRIC TRAJECTORY FIT: H(t) = C * t * exp(-(t-m)/tau)
# =========================================================================

def parametric_fit_entropy(outputs, k_tokens=20):
    """Fit H(t) = C * t * exp(-(t-m)/tau) on averaged top-k entropy trajectories.
    
    Returns: C, tau, m arrays of shape (N,)
    """
    def model(t, c, tau, m):
        return c * t * np.exp(-(t - m) / tau)

    entropies = getEntropy(outputs)
    list_c, list_tau, list_m = [], [], []

    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mean_ent_per_token = np.mean(ent, axis=0)

        if k_tokens:
            indices = np.argsort(mean_ent_per_token)[-k_tokens:]
            mean_ent = np.mean(ent[:, indices], axis=1)
        else:
            mean_ent = np.mean(ent, axis=1)

        t = np.arange(len(mean_ent), dtype=float)
        valid = np.isfinite(mean_ent) & (mean_ent > 0)
        t_fit, y_fit = t[valid], mean_ent[valid]

        if len(y_fit) < 3:
            list_c.append(np.nan); list_tau.append(np.nan); list_m.append(np.nan)
            continue

        c0 = float(max(y_fit.max(), 1e-3))
        tau0 = float(max(len(y_fit) / 4.0, 1.0))
        m0 = float(len(y_fit) / 2.0)

        try:
            params, _ = curve_fit(
                model, t_fit, y_fit, p0=[c0, tau0, m0],
                bounds=([0.0, 1e-6, -len(y_fit)], [np.inf, np.inf, 2*len(y_fit)]),
                maxfev=20000,
            )
            c_i, tau_i, m_i = params
        except Exception:
            c_i, tau_i, m_i = np.nan, np.nan, np.nan

        list_c.append(float(c_i))
        list_tau.append(float(tau_i))
        list_m.append(float(m_i))

    return np.array(list_c), np.array(list_tau), np.array(list_m)


# =========================================================================
# FINITE-DIFFERENCE RATE OF CHANGE (Tau)
# =========================================================================

def mean_var_tau_entropy(outputs, window=10):
    """Discrete rate of change of step-averaged entropy."""
    entropies = getEntropy(outputs)
    list_mean_tau, list_var_tau = [], []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        ent_avg = np.mean(ent, axis=1)
        n_steps = len(ent_avg)
        t_indices = np.arange(0, n_steps - window, window)
        tau = (ent_avg[t_indices + window] - ent_avg[t_indices]) / window
        list_mean_tau.append(np.mean(tau))
        list_var_tau.append(np.var(tau))
    return np.array(list_mean_tau), np.array(list_var_tau)


# =========================================================================
# SHAPE FEATURES ON MASKED VARIANCE CURVE V(t)
# =========================================================================

def _masked_entropy_across_tokens_curve(outputs):
    """Compute V(t) = Var_d(H(t,d) | masked) for each sample."""
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    all_curves = []
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        all_curves.append(step_vars)
    return all_curves


def auc_masked_entropy_curve(outputs):
    """AUC (integral) of the V(t) curve."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([integrate.trapezoid(c) for c in curves])


def max_masked_entropy_curve(outputs):
    """Maximum of V(t) curve."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([np.max(c) for c in curves])


def argmax_masked_entropy_curve(outputs):
    """Step of maximum V(t)."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([np.argmax(c) for c in curves])


def skewness_masked_entropy_curve(outputs):
    """Skewness of V(t) distribution."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([skew(c) for c in curves])


def kurtosis_masked_entropy_curve(outputs):
    """Kurtosis of V(t) distribution."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    return np.array([kurtosis(c) for c in curves])


def mean_curvature_masked_entropy_curve(outputs):
    """Mean curvature of V(t) curve."""
    curves = _masked_entropy_across_tokens_curve(outputs)
    result = []
    for c in curves:
        y = np.asarray(c)
        x = np.arange(len(y))
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        curvature = np.abs(d2y) / (1 + dy**2)**(3/2)
        result.append(np.mean(curvature))
    return np.array(result)


# =========================================================================
# COMBINED EXTRACTION
# =========================================================================

def get_markovian_features(outputs, k_tokens=20):
    """Extract all Markovian dynamic features.
    
    Returns:
        features: np.ndarray (N, F)
        feature_names: list of str
    """
    alpha_ent, beta_ent = alpha_beta_entropy(outputs)
    alpha_ent_avg, beta_ent_avg = alpha_beta_entropy_avg(outputs, k_tokens=k_tokens)
    alpha_lp, beta_lp = alpha_beta_logprob(outputs)
    C, tau, m = parametric_fit_entropy(outputs, k_tokens=k_tokens)
    mean_tau, var_tau = mean_var_tau_entropy(outputs)
    auc = auc_masked_entropy_curve(outputs)
    max_v = max_masked_entropy_curve(outputs)
    argmax_v = argmax_masked_entropy_curve(outputs)
    skew_v = skewness_masked_entropy_curve(outputs)
    kurt_v = kurtosis_masked_entropy_curve(outputs)
    curv_v = mean_curvature_masked_entropy_curve(outputs)

    feats = {
        "AlphaEntropy": alpha_ent,
        "BetaEntropy": beta_ent,
        "AlphaEntropyAvg": alpha_ent_avg,
        "BetaEntropyAvg": beta_ent_avg,
        "AlphaLogProb": alpha_lp,
        "BetaLogProb": beta_lp,
        "C_fit": C,
        "Tau_fit": tau,
        "M_fit": m,
        "MeanTau": mean_tau,
        "VarTau": var_tau,
        "AUC_V": auc,
        "Max_V": max_v,
        "Argmax_V": argmax_v,
        "Skewness_V": skew_v,
        "Kurtosis_V": kurt_v,
        "MeanCurvature_V": curv_v,
    }

    feature_names = list(feats.keys())
    features = np.column_stack([feats[k] for k in feature_names])
    return features, feature_names
