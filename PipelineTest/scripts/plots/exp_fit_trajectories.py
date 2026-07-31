"""
plots.exp_fit_trajectories
===========================
Fit exponential decay c * exp(-t/tau) on mean and variance of masked entropy
trajectories (correct vs hallucination), via log-linear regression:
    ln(y) = alpha * t + beta  =>  tau = -1/alpha, c = exp(beta)

Produces:
- Plot with data + fitted curves
- Prints R^2 for each fit

Usage:
    python -m PipelineTest.scripts.plots.exp_fit_trajectories --config llada
    python -m PipelineTest.scripts.plots.exp_fit_trajectories --config dream
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from .utils import PlotContext


def compute_per_step_stats(ctx: PlotContext, pos_arr):
    """Compute mean and variance of masked entropy per diffusion step."""
    ent_arr = np.array([ctx.entropies[i] for i in pos_arr])
    mask_arr = np.array([ctx.masks[i] for i in pos_arr], dtype=float)
    N, T, D = ent_arr.shape
    mean_per_step = np.zeros(T)
    var_per_step = np.zeros(T)
    for t in range(T):
        step_means = []
        step_vars = []
        for n in range(N):
            m = mask_arr[n, t, :] > 0
            if np.sum(m) > 0:
                vals = ent_arr[n, t, m]
                step_means.append(np.mean(vals))
                if np.sum(m) > 1:
                    step_vars.append(np.var(vals))
        mean_per_step[t] = np.mean(step_means) if step_means else 0
        var_per_step[t] = np.mean(step_vars) if step_vars else 0
    return mean_per_step, var_per_step


def fit_exponential(t, y, label=""):
    """
    Fit y = c * exp(-t/tau) via ln(y) = alpha*t + beta.
    Only uses points where y > 0.
    Returns (alpha, beta, tau, c, r_squared, t_valid, y_fitted).
    """
    valid = y > 0
    t_valid = t[valid]
    y_valid = y[valid]
    
    if len(t_valid) < 3:
        print(f"  [{label}] Not enough positive points for fit.")
        return None
    
    ln_y = np.log(y_valid)
    slope, intercept, r_value, p_value, std_err = linregress(t_valid, ln_y)
    
    r_squared = r_value ** 2
    tau = -1.0 / slope if slope != 0 else np.inf
    c = np.exp(intercept)
    
    # Fitted curve over all valid t
    y_fitted = c * np.exp(slope * t_valid)
    
    return {
        "alpha": slope,
        "beta": intercept,
        "tau": tau,
        "c": c,
        "r_squared": r_squared,
        "t_valid": t_valid,
        "y_fitted": y_fitted,
    }


def plot_exp_fit(ctx: PlotContext):
    """Plot mean/var trajectories with exponential fits."""
    save_dir = ctx.get_save_dir("exp_fit")

    correct_pos = ctx.positions[ctx.labels == 0]
    halluc_pos = ctx.positions[ctx.labels == 1]

    mean_c, var_c = compute_per_step_stats(ctx, correct_pos)
    mean_h, var_h = compute_per_step_stats(ctx, halluc_pos)

    T = len(mean_c)
    t = np.arange(T, dtype=float)

    # --- Fits ---
    fits = {}
    fits["mean_correct"] = fit_exponential(t, mean_c, "Mean Correct")
    fits["mean_halluc"] = fit_exponential(t, mean_h, "Mean Halluc")
    fits["var_correct"] = fit_exponential(t, var_c, "Var Correct")
    fits["var_halluc"] = fit_exponential(t, var_h, "Var Halluc")

    # --- Print results ---
    print("\n" + "=" * 60)
    print(f"  Exponential fit results: {ctx.cfg['name']}")
    print("=" * 60)
    for name, fit in fits.items():
        if fit is not None:
            print(f"  {name:20s} | R² = {fit['r_squared']:.4f} | "
                  f"tau = {fit['tau']:.2f} | c = {fit['c']:.4f} | "
                  f"alpha = {fit['alpha']:.6f}")
        else:
            print(f"  {name:20s} | FIT FAILED")
    print("=" * 60)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Mean entropy
    ax = axes[0]
    ax.plot(t, mean_c, "o-", color="green", markersize=3, label="Correct (data)")
    ax.plot(t, mean_h, "s-", color="red", markersize=3, label="Hallucination (data)")
    if fits["mean_correct"] is not None:
        f = fits["mean_correct"]
        ax.plot(f["t_valid"], f["y_fitted"], "--", color="darkgreen", linewidth=2,
                label=f"Correct fit (R²={f['r_squared']:.3f}, τ={f['tau']:.1f})")
    if fits["mean_halluc"] is not None:
        f = fits["mean_halluc"]
        ax.plot(f["t_valid"], f["y_fitted"], "--", color="darkred", linewidth=2,
                label=f"Halluc fit (R²={f['r_squared']:.3f}, τ={f['tau']:.1f})")
    ax.set_title("Mean Masked Entropy + Exp Fit: $c \\cdot e^{-t/\\tau}$", fontweight="bold")
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Mean Masked Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance
    ax = axes[1]
    ax.plot(t, var_c, "o-", color="green", markersize=3, label="Correct (data)")
    ax.plot(t, var_h, "s-", color="red", markersize=3, label="Hallucination (data)")
    if fits["var_correct"] is not None:
        f = fits["var_correct"]
        ax.plot(f["t_valid"], f["y_fitted"], "--", color="darkgreen", linewidth=2,
                label=f"Correct fit (R²={f['r_squared']:.3f}, τ={f['tau']:.1f})")
    if fits["var_halluc"] is not None:
        f = fits["var_halluc"]
        ax.plot(f["t_valid"], f["y_fitted"], "--", color="darkred", linewidth=2,
                label=f"Halluc fit (R²={f['r_squared']:.3f}, τ={f['tau']:.1f})")
    ax.set_title("Variance of Masked Entropy + Exp Fit: $c \\cdot e^{-t/\\tau}$", fontweight="bold")
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "exp_fit_mean_var.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Saved: {path}")

    return fits


def plot(config_name="llada"):
    """Main entry point."""
    print(f"\n[exp_fit] Loading data ({config_name})...")
    ctx = PlotContext(config_name, use_padding=True)
    print("[exp_fit] Computing fits...")
    fits = plot_exp_fit(ctx)
    print("[exp_fit] Done.")
    return fits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    args = parser.parse_args()
    plot(args.config)
