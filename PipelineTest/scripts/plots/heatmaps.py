"""
plots.heatmaps
==============
Mode: heatmaps

Generates:
- AR(1) parameter heatmaps (phi, intercept, sigma) for correct and halluc models
- Correlation heatmaps between features

Usage:
    python -m PipelineTest.scripts.plots.heatmaps --config llada
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import PlotContext


def plot_ar1_parameter_heatmaps(ctx: PlotContext, no_padding=True):
    """Heatmaps of phi, intercept, sigma for both correct and halluc AR1 models."""
    save_dir = ctx.get_save_dir("heatmaps")
    ar1_models = ctx.get_ar1_models(no_padding=no_padding)

    suffix = "_nopad" if no_padding and ctx.padding_2d is not None else ""

    for model_name, (phi, intercept, sigma) in ar1_models.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im0 = axes[0].imshow(phi[1:, :], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_title(f"$\\phi$ ({model_name})")
        axes[0].set_xlabel("Token position $d$")
        axes[0].set_ylabel("Diffusion step $t$")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(intercept[1:, :], aspect="auto", cmap="RdBu_r")
        axes[1].set_title(f"intercept ({model_name})")
        axes[1].set_xlabel("Token position $d$")
        axes[1].set_ylabel("Diffusion step $t$")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(sigma[1:, :], aspect="auto", cmap="viridis", vmin=0)
        axes[2].set_title(f"$\\sigma$ ({model_name})")
        axes[2].set_xlabel("Token position $d$")
        axes[2].set_ylabel("Diffusion step $t$")
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(f"AR(1) Parameters — {model_name} model", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, f"ar1_heatmap_{model_name}{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {path}")


def plot_correlation_heatmap(ctx: PlotContext):
    """Correlation heatmap of all features."""
    save_dir = ctx.get_save_dir("heatmaps")

    from PipelineTest.features.baseline import get_baseline_features
    from PipelineTest.features.markovian import get_markovian_features

    pad_id = ctx.pad_token_id if ctx.use_padding else None
    feat_b, names_b = get_baseline_features(ctx.outputs, pad_token_id=pad_id)
    feat_b = feat_b[ctx.positions]

    feat_m, names_m = get_markovian_features(ctx.outputs, k_tokens=20)
    feat_m = feat_m[ctx.positions]

    # Combine and remove NaN columns
    features = np.hstack([feat_b, feat_m])
    names = names_b + names_m
    valid = ~np.any(np.isnan(features), axis=0)
    features = features[:, valid]
    names = [n for n, v in zip(names, valid) if v]

    corr = np.corrcoef(features.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot(config_name="llada", no_padding=True):
    """Main entry point for heatmaps mode."""
    print("\n[heatmaps] Loading data...")
    ctx = PlotContext(config_name, use_padding=no_padding)

    print("[heatmaps] AR(1) parameter heatmaps...")
    plot_ar1_parameter_heatmaps(ctx, no_padding=no_padding)

    if no_padding and ctx.padding_2d is not None:
        print("[heatmaps] AR(1) parameter heatmaps (with padding for comparison)...")
        plot_ar1_parameter_heatmaps(ctx, no_padding=False)

    print("[heatmaps] Correlation heatmap...")
    plot_correlation_heatmap(ctx)

    print("[heatmaps] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llada", choices=["llada", "dream"])
    parser.add_argument("--no-padding", action="store_true", default=True)
    args = parser.parse_args()
    plot(args.config, args.no_padding)
