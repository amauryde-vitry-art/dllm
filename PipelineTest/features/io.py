
import sys
import os
import re

import numpy as np
from scipy.optimize import curve_fit

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getNumTransferTokens, getSemanticEntropy, getUnmaskLogProbs, getSemanticBestLogprobsLabel
import torch 
from dataclasses import fields
from dllm.core.samplers.base import BaseSamplerOutputCompleteHistory
from GenerateBaseSamplerOutputsAndExtractInfo.PlotResults import PlotlyEntropy, PlotlyH, plotH, plotLogProbs, plotMasks, plotChanges, plotLevenshtein, plotAttentionMask, plotNumTransferredTokens, plotUnmaskLogProbs, PlotlyLogProbs, PlotlyUnmaskLogProbs, PlotlyChanges, PlotlySemanticBestLabels, getTxt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from PipelineTest.CreateMetrics import getInterceptCAndTauFromVarMaskedEntropyAcrossTokens






def _pad_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(tensor.shape) == target_shape:
        return tensor

    padded = tensor.new_zeros(target_shape)
    slices = tuple(slice(0, s) for s in tensor.shape)
    padded[slices] = tensor
    return padded


def _pad_and_cat_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]

    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        if t.dim() != len(max_shape):
            raise RuntimeError("Cannot merge tensors with different ranks")
        for dim_idx, dim_size in enumerate(t.shape):
            if dim_idx == 0:
                continue
            max_shape[dim_idx] = max(max_shape[dim_idx], dim_size)

    padded_tensors = []
    for t in tensors:
        target_shape = tuple([t.shape[0]] + max_shape[1:])
        padded_tensors.append(_pad_to_shape(t, target_shape))

    return torch.cat(padded_tensors, dim=0)

def mergeOutputsList(ouputpath='PipelineTest/results/outputs_no_remasking.pt'):
    outputs_list = torch.load(ouputpath, map_location="cpu", weights_only=False)

    if isinstance(outputs_list, BaseSamplerOutputCompleteHistory):
        return outputs_list
    if not isinstance(outputs_list, list):
        raise TypeError(f"Expected a list of BaseSampler outputs, got {type(outputs_list)}")
    if len(outputs_list) == 0:
        raise ValueError("outputs_list is empty")

    first = outputs_list[0]
    # Handle list of tuples (int, BaseSamplerOutputCompleteHistory, int)
    if isinstance(first, tuple):
        outputs_list = [item for item in outputs_list if isinstance(item, tuple)]
        outputs_list = [item[1] for item in outputs_list]
        first = outputs_list[0]
    if not isinstance(first, BaseSamplerOutputCompleteHistory):
        raise TypeError(
            "Expected list items of type BaseSamplerOutputCompleteHistory, "
            f"got {type(first)}"
        )

    merged = BaseSamplerOutputCompleteHistory()
    for f in fields(BaseSamplerOutputCompleteHistory):
        name = f.name
        values = [getattr(o, name) for o in outputs_list]
        non_none_values = [v for v in values if v is not None]

        if len(non_none_values) == 0:
            setattr(merged, name, None)
            continue

        sample = non_none_values[0]

        if isinstance(sample, torch.Tensor):
            setattr(merged, name, _pad_and_cat_tensors(non_none_values))
            continue

        if isinstance(sample, list):
            if len(sample) == 0:
                setattr(merged, name, [])
                continue

            if all(isinstance(v, list) for v in non_none_values):
                # Histories are lists of tensors per step; merge step-by-step across batches.
                if all(len(v) == len(non_none_values[0]) for v in non_none_values):
                    if all(
                        len(v) > 0 and isinstance(v[0], torch.Tensor)
                        for v in non_none_values
                    ):
                        merged_history = []
                        for step_idx in range(len(non_none_values[0])):
                            merged_history.append(
                                _pad_and_cat_tensors([v[step_idx] for v in non_none_values])
                            )
                        setattr(merged, name, merged_history)
                    else:
                        flat = []
                        for v in non_none_values:
                            flat.extend(v)
                        setattr(merged, name, flat)
                else:
                    flat = []
                    for v in non_none_values:
                        flat.extend(v)
                    setattr(merged, name, flat)
                continue

        # Scalar metadata (block_size, max_new_tokens, step_per_block): keep first non-None.
        setattr(merged, name, sample)

    return merged

def getEntropicCost(masks, changes, entropies):
    entropic_cost = []
    for j in range(len(masks)):
        changes_before_unmask = np.array(changes[j]) * (np.array(masks[j]))
        cost = ((changes_before_unmask) * np.array(entropies[j])).cumsum(axis=0) # cum sum
        entropic_cost.append(cost)
    return entropic_cost

def plotMeanEntropyAcrossTimeIncreaseOrder(list_index, entropies, save_path):
    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)  
            
        mean_ent_across_tokens = np.mean(ent, axis=0)
        indices_large_entropy_across_tokens = np.argsort(mean_ent_across_tokens)[-64:]

        mean_ent_across_diff = np.mean(ent, axis=1)
        indices_large_entropy_across_diff = np.argsort(mean_ent_across_diff)[-64:]

        sub_matrix_indices = np.ix_(indices_large_entropy_across_diff, indices_large_entropy_across_tokens)
        sub_matrix = ent[sub_matrix_indices]  
        
        mean_ent = np.mean(sub_matrix, axis=1)
        
        ln_mean_ent = np.log(mean_ent + 1e-10)
        n_steps = len(mean_ent)  
        
        t = np.arange(n_steps)
        A = np.vstack([t, np.ones(n_steps)]).T
        
        alpha, beta = np.linalg.lstsq(A, ln_mean_ent, rcond=None)[0]

        plt.figure(figsize=(15, 6))
        plt.plot(mean_ent, label="Mean Entropy across tokens (top 64 by mean)")
        plt.plot(np.exp(alpha * t + beta), label=f"Fitted exp(alpha t + beta) (alpha={alpha:.4f}, beta={beta:.4f})", linestyle='--')
        plt.xlabel("Generation Step")
        plt.ylabel("Mean Entropy across tokens")
        plt.title(f"Mean Entropy across tokens vs Generation Step for sample {list_index[i]}")
        plt.legend()
        plt.grid(True)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"MeanEntropy_across_time_sample_{list_index[i]}.png"))
        plt.show()


def plotMeanEntropyAndVarEntropyWithFittedCurve(list_index, entropies, masks, save_path):
    def model_power(t, a, p):
        return a * np.power(t, -p)

    def model_ct_offset(t, c, tau, m):
        return c * t * np.exp(-(t - m) / tau)

    def fit_ct_offset(y):
        valid = np.isfinite(y) & (y > 0)
        if np.sum(valid) < 3:
            return None, None, None
        t_fit = np.arange(len(y), dtype=float)[valid]
        y_fit = y[valid]
        c0 = float(max(np.max(y_fit), 1e-3))
        tau0 = float(max(len(y_fit) / 4.0, 1.0))
        m0 = float(len(y_fit) / 2.0)
        try:
            params, _ = curve_fit(
                model_ct_offset,
                t_fit,
                y_fit,
                p0=[c0, tau0, m0],
                bounds=([0.0, 1e-6, -len(y_fit)], [np.inf, np.inf, 2 * len(y_fit)]),
                maxfev=20000,
            )
            return float(params[0]), float(params[1]), float(params[2])
        except Exception:
            return None, None, None

    def fit_exponential(y):
        valid = np.isfinite(y) & (y > 0)
        if np.sum(valid) < 3:
            return None, None
        t_fit = np.arange(len(y), dtype=float)[valid]
        y_fit = y[valid]
        alpha, beta = np.polyfit(t_fit, np.log(y_fit), 1)
        return float(alpha), float(beta)

    def fit_power_law(y):
        valid = np.isfinite(y) & (y > 0)
        if np.sum(valid) < 3:
            return None, None
        t_pow = np.arange(len(y), dtype=float) + 1.0
        t_fit = t_pow[valid]
        y_fit = y[valid]
        a0 = float(max(np.max(y_fit), 1e-3))
        p0 = 1.0
        try:
            params, _ = curve_fit(
                model_power,
                t_fit,
                y_fit,
                p0=[a0, p0],
                bounds=([0.0, 0.0], [np.inf, 10.0]),
                maxfev=20000,
            )
            return float(params[0]), float(params[1])
        except Exception:
            return None, None

    for i in range(len(entropies)):
        ent = np.asarray(entropies[i], dtype=float)
        mask = np.asarray(masks[i], dtype=bool)

        n_steps = ent.shape[0]
        t = np.arange(n_steps, dtype=float)
        t_pow = t + 1.0

        mean_entropy = np.mean(ent, axis=1)
        # var_entropy = np.var(ent, axis=1)
        step_vars = []
        for s in range(ent.shape[0]):
            masked_vals = ent[s, mask[s]]
            if len(masked_vals) > 1:
                step_vars.append(np.var(masked_vals))
        var_entropy_masked_across_tokens = np.array(step_vars, dtype=float)
        

        token_vars = []
        for j in range(ent.shape[1]):
            masked_vals = ent[mask[:, j], j]
            if len(masked_vals) > 1:
                token_vars.append(np.var(masked_vals))
        

        k_tokens = min(20, ent.shape[1])
        mean_ent_per_token = np.mean(ent, axis=0)
        indices_top_k = np.argsort(mean_ent_per_token)[-k_tokens:]

        mean_entropy_top_k = np.mean(ent[:, indices_top_k], axis=1)

        c_mean, tau_mean, m_mean = fit_ct_offset(mean_entropy)
        c_mean_top, tau_mean_top, m_mean_top = fit_ct_offset(mean_entropy_top_k)

        c_var, tau_var, m_var = fit_ct_offset(var_entropy_masked_across_tokens)


        fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

        axes[0].plot(mean_entropy, label="Mean Entropy", linewidth=2)
        axes[0].plot(
            mean_entropy_top_k,
            label=f"Mean Entropy top {k_tokens} tokens",
            linestyle=':',
            linewidth=2,
        )
        if c_mean is not None and tau_mean is not None and m_mean is not None:
            axes[0].plot(
                t,
                model_ct_offset(t, c_mean, tau_mean, m_mean),
                label=f"E(t)=C*t*exp(-(t-m)/tau) (C={c_mean:.4f}, tau={tau_mean:.4f}, m={m_mean:.4f})",
                linestyle='--',
            )
        if c_mean_top is not None and tau_mean_top is not None and m_mean_top is not None:
            axes[0].plot(
                t,
                model_ct_offset(t, c_mean_top, tau_mean_top, m_mean_top),
                label=f"Top-{k_tokens}: C*t*exp(-(t-m)/tau) (C={c_mean_top:.4f}, tau={tau_mean_top:.4f}, m={m_mean_top:.4f})",
                linestyle='-.',
            )
        axes[0].set_title("Mean Entropy")
        axes[0].set_xlabel("Generation Step")
        axes[0].set_ylabel("Entropy")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(var_entropy_masked_across_tokens, label="Var(Entropy across tokens)", linewidth=2)
        # axes[1].plot(var_entropy_masked_across_diff, label="Var(Entropy across different steps)", linewidth=2)

        if c_var is not None and tau_var is not None and m_var is not None:
            axes[1].plot(
                t,
                model_ct_offset(t, c_var, tau_var, m_var),
                label=f"E(t)=C*t*exp(-(t-m)/tau) (C={c_var:.4f}, tau={tau_var:.4f}, m={m_var:.4f})",
                linestyle='--',
            )
        
        axes[1].set_title("Variance of Entropy")
        axes[1].set_xlabel("Generation Step")
        axes[1].set_ylabel("Variance")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle(f"Entropy and Variance fits for sample {list_index[i]}")
        fig.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"MeanVarEntropy_Fitted_sample_{list_index[i]}.png"))
        plt.show()

def plotEntropyAndLogProbsAcrossTime(list_index, entropies, logprobs, masks, k_tokens, save_path):

    for i in range(len(entropies)):
        # fit ln(np.mean(entropies[i], axis=1)) with a linear regression: ln(E) = alpha * t + beta
        ent = np.asarray(entropies[i], dtype=float)
        
        mean_ent = np.mean(ent, axis=0)

        indices_large_var = np.argsort(mean_ent)[-k_tokens:]
        mean_ent_top_k = np.mean(ent[:, indices_large_var], axis=1)

        mean_entropy = np.mean(ent, axis=1)
        t = np.arange(len(mean_entropy))
        ln_mean_entropy = np.log(mean_entropy + 1e-8)  # add
        alpha_i, beta_i = np.polyfit(t, ln_mean_entropy, 1)

        # Entropy at the exact step where tokens become unmasked.
        # mask=1 means masked, mask=0 means unmasked.
        masked = np.asarray(masks[i], dtype=bool)
        just_unmasked = np.zeros_like(masked, dtype=bool)
        just_unmasked[1:, :] = masked[:-1, :] & (~masked[1:, :])

        entropy_at_unmask_time = np.full(masked.shape[0], np.nan, dtype=float)
        for s in range(masked.shape[0]):
            selected = ent[s, just_unmasked[s]]
            if selected.size > 0:
                entropy_at_unmask_time[s] = float(np.mean(selected))

        # Fit with intercept: E(t) = C * t * exp(-(t-m)/tau)
        c_b_i, tau_b_i, m_b_i = None, None, None
        valid_b = np.isfinite(mean_entropy) & (mean_entropy > 0)
        if np.sum(valid_b) >= 3:
            t_b = t[valid_b].astype(float)
            y_b = mean_entropy[valid_b].astype(float)

            def model_b(tt, c, tau, m):
                return c * tt * np.exp(-(tt - m) / tau)

            c0 = float(max(y_b.max(), 1e-3))
            tau0 = float(max(len(y_b) / 4.0, 1.0))
            m0 = float(len(y_b) / 2.0)
            try:
                params, _ = curve_fit(
                    model_b,
                    t_b,
                    y_b,
                    p0=[c0, tau0, m0],
                    bounds=([0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf]),
                    maxfev=20000,
                )
                c_b_i, tau_b_i, m_b_i = [float(p) for p in params]
            except Exception:
                pass


        plt.figure(figsize=(15, 6))
        plt.plot((np.mean(ent, axis=1)), label=" Mean Entropy")
        plt.plot((np.mean(logprobs[i], axis=1)), label=" Mean LogProbs")
        plt.plot(mean_ent_top_k, label=f"Mean Entropy of top {k_tokens} tokens with largest mean", linestyle=':')
        plt.plot(entropy_at_unmask_time, label="Entropy at unmask time", linestyle='-', linewidth=2)

        #plot exp(alpha t + beta) as a dashed line
        if alpha_i is not None and beta_i is not None:
            t = np.arange(len(entropies[i]))
            exp_curve = np.exp(alpha_i * t + beta_i)
            plt.plot(t, exp_curve, label=f"exp(alpha t + beta) (alpha={alpha_i:.4f}, beta={beta_i:.4f})", linestyle='--')

       

        # plot C * t * exp(-t/tau)
        if c_b_i is not None and tau_b_i is not None and m_b_i is not None:
            t_ct_b = np.arange(len(entropies[i]), dtype=float)
            ct_b_curve = c_b_i * t_ct_b * np.exp(-(t_ct_b - m_b_i) / tau_b_i)
            plt.plot(
                t_ct_b,
                ct_b_curve,
                label=f"C*t*exp(-(t-m)/tau) (C={c_b_i:.4f}, tau={tau_b_i:.4f}, m={m_b_i:.4f})",
                linestyle='-.',
            )
        plt.xlabel("Generation Step")
        plt.ylabel("Mean Entropy across tokens")
        plt.title(f"Mean Entropy across tokens vs Generation Step for sample {list_index[i]}")
        plt.legend()
        plt.grid(True)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"MeanEntropy_MeanLogProbs_across_time_sample_{list_index[i]}.png"))
        plt.show()
        

def GetInfoFromIndex(listIndex, tokenizer, ouputpath='PipelineTest/results/outputs_no_remasking.pt'):
    from PipelineTest.CreateMetrics import getAlphaAndBetaFromEntropy

    outputs = mergeOutputsList(ouputpath)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    remasks = getEachStepMask(outputs, remask=True) if outputs.histories_remasking is not None else None
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    entropies = getEntropy(outputs) 
    if outputs.attention_mask is None:
        raise RuntimeError("Merged outputs has no attention_mask")
    attention_masks = outputs.attention_mask.detach().cpu().numpy()
    H = getH(outputs) if outputs.histories_H is not None else None
    entropic_costs = getEntropicCost(masks, changes, entropies)

    hmap = {}
    for i in range(len(outputs.sample_indices)):
        j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
        hmap[j] = i
    
    l_index = [hmap[i] for i in listIndex]


    block_size = outputs.block_size 

    logprobs = [logprobs[i] for i in l_index]
    masks = [masks[i] for i in l_index]
    if remasks is not None:
        remasks = [remasks[i] for i in l_index]
    Generated_sequences = [Generated_sequences[i] for i in l_index]
    Proposed_sequences = [Proposed_sequences[i] for i in l_index]
    changes = [changes[i] for i in l_index]
    levenshtein = [levenshtein[i] for i in l_index]
    unmaskLogProbs = [unmaskLogProbs[i] for i in l_index]
    entropies = [entropies[i] for i in l_index]
    attention_masks = attention_masks[l_index]
    if H is not None:
        H = [H[i] for i in l_index]
    entropic_costs = [entropic_costs[i] for i in l_index]


    return logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_costs, 


def PlotInfo(listIndex, tokenizer, title_list, save_path=f"PipelineTest/PlotResults",  ouputpath='PipelineTest/results/outputs.pt'):
    logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_costs = GetInfoFromIndex(listIndex, tokenizer, ouputpath)
    # getTxt(Proposed_sequences, save_path, proposed_sequence=True)
    # getTxt(Generated_sequences, save_path, proposed_sequence=False)

    # plotLogProbs(logprobs, title_list, save_path, block_size, Generated_sequences, masks)
    # plotMasks(masks, title_list, save_path, block_size, Generated_sequences)
    # plotChanges(changes, title_list, save_path, block_size, Generated_sequences)
    # plotUnmaskLogProbs(unmaskLogProbs, title_list, save_path, block_size, Generated_sequences, masks)
    # plotLevenshtein(levenshtein, title_list, save_path, block_size, Generated_sequences)
    # plotAttentionMask(attention_masks, title_list, save_path)
    # plotMeanEntropyAndVarEntropyWithFittedCurve(listIndex, entropies, masks, save_path)
    # plotEntropyAndLogProbsAcrossTime(listIndex, entropies, logprobs, masks, 20, save_path)
    # plotMeanEntropyAcrossTimeIncreaseOrder(listIndex, entropies, save_path)
   
    PlotlyLogProbs(logprobs, title_list, save_path, block_size, Proposed_sequences, masks, entropies=entropies)
    # PlotlyUnmaskLogProbs(unmaskLogProbs, title_list, save_path, block_size, Generated_sequences, masks)
    # PlotlyChanges(changes, title_list, save_path, block_size, Proposed_sequences)
    PlotlyEntropy(entropies, title_list, save_path, block_size, Proposed_sequences, masks,)

    # PlotlyEntropy(entropic_costs, ['cost_'+title for title in title_list], save_path, block_size, Proposed_sequences, masks,)

    # if outputs.histories_semantic_bestlogprobs_label is not None:
    #     best_labels = getSemanticBestLogprobsLabel(outputs)
    #     best_labels = [best_labels[i] for i in listIndex]
    #     PlotlySemanticBestLabels(best_labels, title_list, save_path, block_size, Proposed_sequences, masks)

    # if outputs.histories_semantic_entropy is not None:
    #     semantic_entropies = getSemanticEntropy(outputs)
    #     semantic_entropies = [semantic_entropies[i] for i in listIndex]
    #     PlotlyEntropy(semantic_entropies, ['semantic_entropy_'+title for title in title_list], save_path, block_size, Proposed_sequences, masks,)


def PlotInfoRemasking(listIndex, tokenizer, title_list, save_path=f"PipelineTest/PlotResultsRemasking",  ouputpath='PipelineTest/results/outputs.pt'):
    outputs = mergeOutputsList(ouputpath)
    logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_cost = GetInfoFromIndex(listIndex, tokenizer, ouputpath)
    getTxt(Proposed_sequences, save_path, proposed_sequence=True)
    getTxt(Generated_sequences, save_path, proposed_sequence=False)

    plotLogProbs(logprobs, title_list, save_path, block_size, Generated_sequences, masks, remasking_masks=remasks,)
    plotMasks(masks, title_list, save_path, block_size, Generated_sequences, remasking=False)
    plotMasks(remasks, title_list, save_path, block_size, Generated_sequences, remasking=True)
    plotChanges(changes, title_list, save_path, block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, title_list, save_path, block_size, Generated_sequences, masks, remasking_masks=remasks)
    plotLevenshtein(levenshtein, title_list, save_path, block_size, Generated_sequences)
    plotAttentionMask(attention_masks, title_list, save_path)

    PlotlyLogProbs(logprobs, title_list, save_path, block_size, Proposed_sequences, masks, remasking_masks=remasks, entropies=entropies)
    PlotlyUnmaskLogProbs(unmaskLogProbs, title_list, save_path, block_size, Generated_sequences, masks, remasking_masks=remasks)
    PlotlyChanges(changes, title_list, save_path, block_size, Proposed_sequences, remasking_masks=remasks)
    PlotlyH(H, title_list, save_path, block_size, Proposed_sequences, masks, remasking_masks=remasks)
    PlotlyEntropy(entropies, title_list, save_path, block_size, Proposed_sequences, masks, remasking_masks=remasks)

    if outputs.histories_semantic_bestlogprobs_label is not None:
        best_labels = getSemanticBestLogprobsLabel(outputs)
        best_labels = [best_labels[i] for i in listIndex]
        PlotlySemanticBestLabels(best_labels, title_list, save_path, block_size, Proposed_sequences, masks, remasking_masks=remasks)




def PlotPipeline(listIndex, tokenizer, title_list, remasking=False, save_path=f"PipelineTest/PlotResults", save_path_remasking=f"PipelineTest/PlotResultsRemasking",  ouputpath='PipelineTest/results/outputs.pt'):
    if remasking:
        PlotInfoRemasking(listIndex, tokenizer, title_list, save_path_remasking, ouputpath)
    else:
        PlotInfo(listIndex, tokenizer, title_list, save_path, ouputpath)


def PlotScatterVarMaskedEntropyAcrossTokensVsAlphaEntropyAvg(correct_indices, fabricated_indices, save_path, ouputpath):
    """
    Scatter plot: mean(Var(Entropy across diffusion steps)) vs mean(Var(logprobs across diffusion steps))
    Binary labels: correct (green) vs hallucinated (red).
    """
    from PipelineTest.CreateMetrics import getAlphaBetaFromEntropyAvg, getFromOutputsVarEntropyMaskedAcrossTokens

    outputs = mergeOutputsList(ouputpath)
    sample_id = outputs.sample_indices.detach().cpu().numpy() if outputs.sample_indices is not None else np.arange(len(outputs.histories_entropy))
    hmap = {sample_id[i]: i for i in range(len(sample_id))}
    correct_indices = [hmap[i] for i in correct_indices if i in hmap]
    fabricated_indices = [hmap[i] for i in fabricated_indices if i in hmap]
    VarMaskedEntropyAcrossTokens = getFromOutputsVarEntropyMaskedAcrossTokens(outputs)
    alphaEntropyAvg, betaEntropyAvg = getAlphaBetaFromEntropyAvg(outputs)
    

    all_indices = sorted(correct_indices + fabricated_indices)
    correct_set = set(correct_indices)

    plt.figure(figsize=(12, 7))
    for i in all_indices:
        if i < len(VarMaskedEntropyAcrossTokens):
            color = '#2ecc71' if i in correct_set else '#e74c3c'
            plt.scatter(alphaEntropyAvg[i], VarMaskedEntropyAcrossTokens[i], c=color, edgecolor='k', s=50, alpha=0.7)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, markeredgecolor='k', label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, markeredgecolor='k', label='Hallucination'),
    ]
    plt.legend(handles=legend_handles)
    plt.xlabel("Mean Alpha Masked Entropy across tokens")
    plt.ylabel("Mean Var(Entropy across diffusion steps)")
    plt.title("Scatter Plot: Mean Var(Entropy) vs Mean Alpha Entropy")

    plt.grid(True, alpha=0.3)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "scatter_var_entropy_vs_var_logprobs.png"), dpi=150, bbox_inches='tight')
    plt.show()


def PlotMeanVarMaskedAcrossTokens(correct_indices, fabricated_indices, save_path, ouputpath):
    """
    Plot 3 curves across diffusion steps, fit the asymptotic decay phase,
    and display the estimated tau_max for each group.
    """
    outputs = mergeOutputsList(ouputpath)
    sample_id = outputs.sample_indices.detach().cpu().numpy() if outputs.sample_indices is not None else np.arange(len(outputs.histories_entropy))
    hmap = {sample_id[i]: i for i in range(len(sample_id))}
    correct_indices = [hmap[i] for i in correct_indices if i in hmap]
    fabricated_indices = [hmap[i] for i in fabricated_indices if i in hmap]
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)

    def _per_step_var_masked(ent, mask):
        ent = np.asarray(ent, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        n_steps = ent.shape[0]
        out = np.full(n_steps, np.nan, dtype=float)
        for s in range(n_steps):
            masked_vals = ent[s, mask[s]]
            if masked_vals.size > 1:
                out[s] = float(np.var(masked_vals))
        return out

    def _build_group_matrix(indices):
        sequences = []
        for i in indices:
            if 0 <= i < len(entropies):
                sequences.append(_per_step_var_masked(entropies[i], masks[i]))

        if not sequences:
            return np.empty((0, 0), dtype=float)

        max_len = max(len(seq) for seq in sequences)
        mat = np.full((len(sequences), max_len), np.nan, dtype=float)
        for row, seq in enumerate(sequences):
            mat[row, : len(seq)] = seq
        return mat

    mat_correct = _build_group_matrix(correct_indices)
    mat_fabricated = _build_group_matrix(fabricated_indices)

    if mat_correct.size == 0 and mat_fabricated.size == 0:
        raise ValueError("No valid sample indices found in correct_indices or fabricated_indices")

    if mat_correct.size == 0:
        mean_correct = np.full(mat_fabricated.shape[1], np.nan, dtype=float)
    else:
        mean_correct = np.nanmean(mat_correct, axis=0)

    if mat_fabricated.size == 0:
        mean_fabricated = np.full(mat_correct.shape[1], np.nan, dtype=float)
    else:
        mean_fabricated = np.nanmean(mat_fabricated, axis=0)

    target_len = max(len(mean_correct), len(mean_fabricated))
    if len(mean_correct) < target_len:
        padded = np.full(target_len, np.nan, dtype=float)
        padded[: len(mean_correct)] = mean_correct
        mean_correct = padded
    if len(mean_fabricated) < target_len:
        padded = np.full(target_len, np.nan, dtype=float)
        padded[: len(mean_fabricated)] = mean_fabricated
        mean_fabricated = padded

    mean_both = np.nanmean(np.vstack([mean_correct, mean_fabricated]), axis=0)

    count_correct = np.zeros(target_len, dtype=int)
    if mat_correct.size != 0:
        count_correct[: mat_correct.shape[1]] = np.sum(~np.isnan(mat_correct), axis=0)

    count_fabricated = np.zeros(target_len, dtype=int)
    if mat_fabricated.size != 0:
        count_fabricated[: mat_fabricated.shape[1]] = np.sum(~np.isnan(mat_fabricated), axis=0)

    count_total = count_correct + count_fabricated

    t = np.arange(target_len)
    plt.figure(figsize=(15, 7))
    
    # --- 1. Tracé des courbes originales ---
    plt.plot(t, mean_correct, label="Correct (mean)", linewidth=2, color="#2ecc71", marker='o', markersize=4)
    plt.plot(t, mean_fabricated, label="Fabricated (mean)", linewidth=2, color="#e74c3c", marker='o', markersize=4)
    plt.plot(t, mean_both, label="Mean of both", linewidth=2, color="#3498db", linestyle="--", marker='o', markersize=4)

    # --- 2. Module de Fit Physique (Asymptotique) ---
    def fit_ct_exp(t_arr, y_arr, start_fit_idx=53, m=53):
        """ 
        Fit y = C * (t-m) * exp(-2*(t-m) / tau) sur la portion [start_fit_idx:], avec m fixé.
        """
        if len(y_arr) <= start_fit_idx:
            return None, None, None
            
        t_fit = t_arr[start_fit_idx:].astype(float)
        y_fit = y_arr[start_fit_idx:]
        
        valid = (~np.isnan(y_fit)) & (y_fit > 0)
        t_fit = t_fit[valid]
        y_fit = y_fit[valid]
        
        if len(t_fit) < 3:
            return None, None, None

        def model_ct(tt, c, tau):
            return c * (tt - m) * np.exp(-2 * (tt - m) / tau)

        idx_peak = np.argmax(y_fit)
        tau0 = 5.0
        c0 = float(np.max(y_fit)) / (tau0 * np.exp(-1))

        try:
            params, _ = curve_fit(
                model_ct,
                t_fit,
                y_fit,
                p0=[c0, tau0],
                bounds=([0.0, 0.1], [np.inf, np.inf]),
                maxfev=20000,
            )
            c_i, tau_i = params
            y_pred = model_ct(t_fit, c_i, tau_i)
            return t_fit, y_pred, tau_i
        except Exception:
            return None, None, None

    # Application du fit sur le groupe "Correct"
    t_c, y_c, tau_c = fit_ct_exp(t, mean_correct)
    if tau_c is not None:
        plt.plot(t_c, y_c, color="#1abc9c", linestyle=":", linewidth=2.5, 
                 label=r"Fit Correct: $C(t-53)e^{-2(t-53)/\tau}$" + f" (τ={tau_c:.2f})")

    # Application du fit sur le groupe "Fabricated"
    t_f, y_f, tau_f = fit_ct_exp(t, mean_fabricated)
    if tau_f is not None:
        plt.plot(t_f, y_f, color="#d35400", linestyle=":", linewidth=2.5, 
                 label=r"Fit Fabricated: $C(t-53)e^{-2(t-53)/\tau}$" + f" (τ={tau_f:.2f})")

    # --- 3. Cosmétique du graphique ---
    plt.xlabel("Generation Step")
    plt.ylabel("VarMaskedEntropyAcrossTokens")
    plt.title("VarMaskedEntropyAcrossTokens over time")
    plt.grid()
    plt.legend(fontsize=11, loc="upper left")

    tick_positions = np.arange(0, target_len, 5)
    tick_labels = [f"{step}\n(n={count_total[step]})" for step in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "mean_var_masked_entropy_across_tokens_fitted.png"),
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


def PlotMeanMaskedEntropyAcrossTokens(correct_indices, fabricated_indices, save_path, ouputpath):
    """
    Plot mean of masked entropy across tokens for correct vs fabricated, with exponential and custom fits.
    """
    outputs = mergeOutputsList(ouputpath)
    sample_id = outputs.sample_indices.detach().cpu().numpy() if outputs.sample_indices is not None else np.arange(len(outputs.histories_entropy))
    hmap = {sample_id[i]: i for i in range(len(sample_id))}
    correct_indices = [hmap[i] for i in correct_indices if i in hmap]
    fabricated_indices = [hmap[i] for i in fabricated_indices if i in hmap]
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)

    def _per_step_mean_masked(ent, mask):
        ent = np.asarray(ent, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        n_steps = ent.shape[0]
        out = np.full(n_steps, np.nan, dtype=float)
        for s in range(n_steps):
            masked_vals = ent[s, mask[s]]
            if masked_vals.size > 0:
                out[s] = float(np.mean(masked_vals))
        return out

    def _build_group_matrix(indices):
        sequences = []
        for i in indices:
            if 0 <= i < len(entropies):
                sequences.append(_per_step_mean_masked(entropies[i], masks[i]))

        if not sequences:
            return np.empty((0, 0), dtype=float)

        max_len = max(len(seq) for seq in sequences)
        mat = np.full((len(sequences), max_len), np.nan, dtype=float)
        for row, seq in enumerate(sequences):
            mat[row, : len(seq)] = seq
        return mat

    mat_correct = _build_group_matrix(correct_indices)
    mat_fabricated = _build_group_matrix(fabricated_indices)

    if mat_correct.size == 0 and mat_fabricated.size == 0:
        raise ValueError("No valid sample indices found in correct_indices or fabricated_indices")

    if mat_correct.size == 0:
        mean_correct = np.full(mat_fabricated.shape[1], np.nan, dtype=float)
    else:
        mean_correct = np.nanmean(mat_correct, axis=0)

    if mat_fabricated.size == 0:
        mean_fabricated = np.full(mat_correct.shape[1], np.nan, dtype=float)
    else:
        mean_fabricated = np.nanmean(mat_fabricated, axis=0)

    target_len = max(len(mean_correct), len(mean_fabricated))
    if len(mean_correct) < target_len:
        padded = np.full(target_len, np.nan, dtype=float)
        padded[: len(mean_correct)] = mean_correct
        mean_correct = padded
    if len(mean_fabricated) < target_len:
        padded = np.full(target_len, np.nan, dtype=float)
        padded[: len(mean_fabricated)] = mean_fabricated
        mean_fabricated = padded

    mean_both = np.nanmean(np.vstack([mean_correct, mean_fabricated]), axis=0)

    count_correct = np.zeros(target_len, dtype=int)
    if mat_correct.size != 0:
        count_correct[: mat_correct.shape[1]] = np.sum(~np.isnan(mat_correct), axis=0)

    count_fabricated = np.zeros(target_len, dtype=int)
    if mat_fabricated.size != 0:
        count_fabricated[: mat_fabricated.shape[1]] = np.sum(~np.isnan(mat_fabricated), axis=0)

    count_total = count_correct + count_fabricated

    t = np.arange(target_len)
    plt.figure(figsize=(15, 7))

    plt.plot(t, mean_correct, label="Correct (mean)", linewidth=2, color="#2ecc71", marker='o', markersize=4)
    plt.plot(t, mean_fabricated, label="Fabricated (mean)", linewidth=2, color="#e74c3c", marker='o', markersize=4)
    plt.plot(t, mean_both, label="Mean of both", linewidth=2, color="#3498db", linestyle="--", marker='o', markersize=4)
    
    # --- ANCIEN FIT EXPONENTIEL ---
    def fit_ct_exp(t_arr, y_arr, start_fit_idx=0):
        if len(y_arr) <= start_fit_idx:
            return None, None, None
        t_fit = t_arr[start_fit_idx:].astype(float)
        y_fit = y_arr[start_fit_idx:]
        valid = (~np.isnan(y_fit)) & (y_fit > 0)
        t_fit = t_fit[valid]
        y_fit = y_fit[valid]
        if len(t_fit) < 3:
            return None, None, None

        def model_ct(tt, c, tau, m):
            return c * (tt - m) * np.exp(-(tt - m) / tau)

        tau0 = 5.0
        m_0 = 0
        c0 = float(np.max(y_fit)) / (tau0 * np.exp(-1))
        try:
            params, _ = curve_fit(
                model_ct, t_fit, y_fit,
                p0=[c0, tau0, m_0],
                bounds=([0.0, 0.1, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            c_i, tau_i, m_i = params
            y_pred = model_ct(t_fit, c_i, tau_i, m_i)
            return t_fit, y_pred, tau_i
        except Exception:
            return None, None, None

    # t_c, y_c, tau_c = fit_ct_exp(t, mean_correct)
    # if tau_c is not None:
    #     plt.plot(t_c, y_c, color="#1abc9c", linestyle=":", linewidth=2.5,
    #              label=r"Fit Correct Exp: $C(t-m)e^{-(t-m)/\tau}$" + f" (τ={tau_c:.2f})")

    # t_f, y_f, tau_f = fit_ct_exp(t, mean_fabricated)
    # if tau_f is not None:
    #     plt.plot(t_f, y_f, color="#d35400", linestyle=":", linewidth=2.5,
    #              label=r"Fit Fabricated Exp: $C(t-m)e^{-(t-m)/\tau}$" + f" (τ={tau_f:.2f})")

    # --- NOUVEAU FIT : S0 * exp(-lambda*t) / (N - t) ---
    # Modèle physique: compétition entre la décroissance diffusive (numérateur)
    # et le biais de sélection / survivorship (dénominateur qui diminue)
    N_tokens = 64.0

    def fit_survivorship(t_arr, y_arr, start_idx=5, end_idx=55):
        if len(y_arr) <= start_idx:
            return None, None, None, None
        
        actual_end = min(end_idx, len(y_arr))
        t_fit = t_arr[start_idx:actual_end].astype(float)
        y_fit = y_arr[start_idx:actual_end]
        
        valid = ~np.isnan(y_fit)
        t_fit = t_fit[valid]
        y_fit = y_fit[valid]
        
        if len(t_fit) < 3:
            return None, None, None, None

        def model_surv(tt, S0, lam):
            return S0 * np.exp(-lam * tt) / (N_tokens - tt)

        # Initial guesses
        S0_0 = float(y_fit[0]) * (N_tokens - t_fit[0])
        lam_0 = 0.01
        try:
            params, _ = curve_fit(
                model_surv, t_fit, y_fit,
                p0=[S0_0, lam_0],
                bounds=([0.0, 0.0], [np.inf, 1.0]),
                maxfev=20000,
            )
            S0_opt, lam_opt = params
            # Predict on full range for visualization
            t_pred = t_arr[start_idx:min(end_idx, len(y_arr))].astype(float)
            y_pred = model_surv(t_pred, S0_opt, lam_opt)
            return t_pred, y_pred, S0_opt, lam_opt
        except Exception:
            return None, None, None, None

    t_surv_c, y_surv_c, S0_c, lam_c = fit_survivorship(t, mean_correct, start_idx=2, end_idx=60)
    if S0_c is not None:
        plt.plot(t_surv_c, y_surv_c, color="#27ae60", linestyle="-.", linewidth=2.5,
                 label=r"Fit Correct: $\frac{S_0 e^{-\lambda t}}{N-t}$" + f" (S₀={S0_c:.1f}, λ={lam_c:.4f})")

    t_surv_f, y_surv_f, S0_f, lam_f = fit_survivorship(t, mean_fabricated, start_idx=2, end_idx=60)
    if S0_f is not None:
        plt.plot(t_surv_f, y_surv_f, color="#c0392b", linestyle="-.", linewidth=2.5,
                 label=r"Fit Fabricated: $\frac{S_0 e^{-\lambda t}}{N-t}$" + f" (S₀={S0_f:.1f}, λ={lam_f:.4f})")


    plt.xlabel("Generation Step")
    plt.ylabel("MeanMaskedEntropyAcrossTokens")
    plt.title("MeanMaskedEntropyAcrossTokens over time")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper left")

    tick_positions = np.arange(0, target_len, 5)
    tick_labels = [f"{step}\n(n={count_total[step]})" for step in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "mean_masked_entropy_across_tokens_fitted.png"),
        dpi=150,
        bbox_inches='tight',
    )
    plt.show()


def PlotDistributionTau(correct_indices, fabricated_indices, save_path, ouputpath):
    """
    Plot the distribution of tau values for correct vs fabricated samples.
    """
    outputs = mergeOutputsList(ouputpath)
    _, all_tau = getInterceptCAndTauFromVarMaskedEntropyAcrossTokens(outputs, k_tokens=None)
    sample_id = outputs.sample_indices.detach().cpu().numpy() if outputs.sample_indices is not None else np.arange(len(outputs.histories_entropy))
    hmap = {sample_id[i]: i for i in range(len(sample_id))}
    correct_indices = [hmap[i] for i in correct_indices if i in hmap]
    fabricated_indices = [hmap[i] for i in fabricated_indices if i in hmap]
    
    # Assuming tau values are stored or can be computed from the outputs
    tau_values_correct = [all_tau[i] for i in correct_indices if not np.isnan(all_tau[i])]
    tau_values_fabricated = [all_tau[i] for i in fabricated_indices if not np.isnan(all_tau[i])]
    print(f"Number of valid tau values for correct samples: {len(tau_values_correct)}")
    print(f"Number of valid tau values for fabricated samples: {len(tau_values_fabricated)}")
    plt.figure(figsize=(12, 7))
    plt.hist(tau_values_correct, bins=20, alpha=0.7, label='Correct', color='#2ecc71', edgecolor='k')
    plt.hist(tau_values_fabricated, bins=20, alpha=0.7, label='Fabricated', color='#e74c3c', edgecolor='k')
    
    plt.xlabel("Tau Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tau Values: Correct vs Fabricated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "distribution_tau_values.png"), dpi=150, bbox_inches='tight')
    plt.show()



def plotEntropyForSteps(correct_indices, fabricated_indices, start_id, end_id, save_path, outputpath):
    """
    Plot the entropy distribution using Violin Plots to clearly capture density per step.
    """
    outputs = mergeOutputsList(outputpath)
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)

    sample_id = outputs.sample_indices.detach().cpu().numpy() if outputs.sample_indices is not None else np.arange(len(outputs.histories_entropy))
    hmap = {sample_id[i]: i for i in range(len(sample_id))}
    correct_indices = [hmap[i] for i in correct_indices if i in hmap]
    fabricated_indices = [hmap[i] for i in fabricated_indices if i in hmap]
    
    d = {}
    meanVar_correct = []
    meanVar_fabricated = []
    for step in range(start_id, end_id):
        d[step] = {'correct': [], 'fabricated': []}
        correct_var = []
        fabricated_var = []
        for i in range(len(entropies)):
            masked_vals = [entropies[i][step][j] for j in range(len(entropies[i][step])) if masks[i][step][j]]
            if i in correct_indices:
                d[step]['correct'].extend(masked_vals)
                if len(masked_vals) > 1:
                    correct_var.append(np.var(masked_vals))
            elif i in fabricated_indices:
                d[step]['fabricated'].extend(masked_vals)
                if len(masked_vals) > 1:
                    fabricated_var.append(np.var(masked_vals))
        meanVar_correct.append(np.mean(correct_var) if correct_var else np.nan)
        meanVar_fabricated.append(np.mean(fabricated_var) if fabricated_var else np.nan)
        d[step]['n_correct_samples'] = len(correct_var)
        d[step]['n_fabricated_samples'] = len(fabricated_var)

    
    steps = list(d.keys())
    os.makedirs(save_path, exist_ok=True)

    # -------------------------------------------------------------
    # --- FIGURE 1 : CORRECT SAMPLES (VIOLIN PLOT) ---
    # -------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    
    # Préparation des données sous forme de liste de listes pour matplotlib
    data_correct = [d[step]['correct'] for step in steps]
    
    # Génération du Violin Plot
    # showmeans=False, showmedians=False car on trace la moyenne proprement après
    v1 = plt.violinplot(data_correct, positions=steps, showextrema=False, widths=0.7)
    
    # Personnalisation de la couleur du violon
    for body in v1['bodies']:
        body.set_facecolor('#2ecc71')
        body.set_edgecolor('#27ae60')
        body.set_alpha(0.6) # L'opacité permet de voir les zones plus sombres / denses

    # Tracé de la ligne de moyenne
    mean_correct = [np.mean(d[step]['correct']) if len(d[step]['correct']) > 0 else np.nan for step in steps]
    plt.plot(steps, mean_correct, color="#1e272e", linewidth=2, linestyle='--', zorder=5)
    plt.scatter(steps, mean_correct, color="#ffffff", edgecolor="#1e272e", s=50, marker='D', linewidth=2, label='Mean Correct', zorder=6)
    
    plt.plot(steps, meanVar_correct, color="#1f2e1e", linewidth=2, linestyle='--', zorder=5)
    plt.scatter(steps, meanVar_correct, color="#6fc858", edgecolor="#1e272e", s=50, marker='D', linewidth=2, label='Mean Var Correct', zorder=6)
    
    plt.xlabel("Generation Step", fontsize=11)
    plt.ylabel("Entropy", fontsize=11)
    plt.title(f"Entropy Density Distribution (Violin) - Correct Samples", fontsize=13, fontweight='semibold')
    plt.grid(True, linestyle='--', alpha=0.4)
    tick_labels_correct = [f"{step}\n(n={len(d[step]['correct'])})" for step in steps]
    plt.xticks(steps, tick_labels_correct, fontsize=9)
    plt.xlim(min(steps) - 0.5, max(steps) + 0.5)
    plt.legend()
    
    plt.savefig(os.path.join(save_path, f"entropy_violin_from_{start_id}_to_{end_id}_correct.png"), dpi=200, bbox_inches='tight')

    # -------------------------------------------------------------
    # --- FIGURE 2 : FABRICATED SAMPLES (VIOLIN PLOT) ---
    # -------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    
    data_fabricated = [d[step]['fabricated'] for step in steps]
    
    v2 = plt.violinplot(data_fabricated, positions=steps, showextrema=False, widths=0.7)
    
    for body in v2['bodies']:
        body.set_facecolor('#e74c3c')
        body.set_edgecolor('#c0392b')
        body.set_alpha(0.6)

    mean_fabricated = [np.mean(d[step]['fabricated']) if len(d[step]['fabricated']) > 0 else np.nan for step in steps]
    plt.plot(steps, mean_fabricated, color="#1e272e", linewidth=2, linestyle='--', zorder=5)
    plt.scatter(steps, mean_fabricated, color="#ffffff", edgecolor="#1e272e", s=50, marker='D', linewidth=2, label='Mean Fabricated', zorder=6)
    
    plt.plot(steps, meanVar_fabricated, color="#1e272e", linewidth=2, linestyle='--', zorder=5)
    plt.scatter(steps, meanVar_fabricated, color="#c12424", edgecolor="#1e272e", s=50, marker='D', linewidth=2, label='Mean Var Fabricated', zorder=6)

    plt.xlabel("Generation Step", fontsize=11)
    plt.ylabel("Entropy", fontsize=11)
    plt.title(f"Entropy Density Distribution (Violin) - Fabricated Samples", fontsize=13, fontweight='semibold')
    plt.grid(True, linestyle='--', alpha=0.4)
    tick_labels_fabricated = [f"{step}\n(n={len(d[step]['fabricated'])})" for step in steps]
    plt.xticks(steps, tick_labels_fabricated, fontsize=9)
    plt.xlim(min(steps) - 0.5, max(steps) + 0.5)
    plt.legend()
    
    plt.savefig(os.path.join(save_path, f"entropy_violin_from_{start_id}_to_{end_id}_fabricated.png"), dpi=200, bbox_inches='tight')

    