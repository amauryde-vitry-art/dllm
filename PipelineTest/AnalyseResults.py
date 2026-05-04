
import sys
import os
import re

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from StudyAutoRegressionBehavior.GetInfoFromBaseSamplerOutput import getEntropy, getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getNumTransferTokens, getUnmaskLogProbs
import torch 
from dataclasses import fields
from dllm.core.samplers.base import BaseSamplerOutputCompleteHistory
from StudyAutoRegressionBehavior.PlotResults import PlotlyEntropy, PlotlyH, plotH, plotLogProbs, plotMasks, plotChanges, plotLevenshtein, plotAttentionMask, plotNumTransferredTokens, plotUnmaskLogProbs, PlotlyLogProbs, PlotlyUnmaskLogProbs, PlotlyChanges , getTxt
import matplotlib.pyplot as plt


def _safe_filename_component(value, max_len=120):
    text = str(value)
    text = re.sub(r'[\\/:*?"<>|]+', '_', text)
    text = re.sub(r'\s+', '_', text).strip('._')
    if not text:
        text = "untitled"
    return text[:max_len]


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


def getPCI(masks, changes, entropies, alpha = 0.7):
    pci_list = []
    for j in range(len(masks)):
        k = np.sum(1 - np.array(masks[j]),axis = 1)
        changes_after_unmask = np.sum(np.array(changes[j]) * (1 - np.array(masks[j])),axis = 1) / k
        entropy_after_unmask = np.sum(np.array(entropies[j]) * (1 - np.array(masks[j])),axis = 1) / k
        pci = alpha * changes_after_unmask + (1-alpha) * entropy_after_unmask
        pci_list.append(np.mean(pci))
    return pci_list
        

def GetInfoFromIndex(listIndex, tokenizer, ouputpath='PipelineTest/results/outputs_no_remasking.pt'):
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


    block_size = outputs.block_size 

    logprobs = [logprobs[i] for i in listIndex]
    masks = [masks[i] for i in listIndex]
    if remasks is not None:
        remasks = [remasks[i] for i in listIndex]
    Generated_sequences = [Generated_sequences[i] for i in listIndex]
    Proposed_sequences = [Proposed_sequences[i] for i in listIndex]
    changes = [changes[i] for i in listIndex]
    levenshtein = [levenshtein[i] for i in listIndex]
    unmaskLogProbs = [unmaskLogProbs[i] for i in listIndex]
    entropies = [entropies[i] for i in listIndex]
    attention_masks = attention_masks[listIndex]
    if H is not None:
        H = [H[i] for i in listIndex]
    entropic_costs = [entropic_costs[i] for i in listIndex]

    return logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_costs


def getMeanVarEntropyVSMeanVarLogProbs(tokenizer, alpha=0.7, beta=0.7, ouputpath='PipelineTest/results/outputs_no_remasking.pt'):
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
    attention_masks = outputs.attention_mask.detach().cpu().numpy()
    H = getH(outputs) 
    entropic_costs = getEntropicCost(masks, changes, entropies)
    logprobs_decay = [np.polyfit(np.arange(len(e)), np.array(e), deg=1)[0] for e in logprobs]
    pci = getPCI(masks, changes, entropies, beta)
    var_entropies = [np.mean(np.var(e, axis=1)) for e in entropies]
    var_logprobs = [(np.var(lp)) for lp in logprobs]
    CSG = [np.mean(lp) - alpha * pci for lp, pci in zip(logprobs, pci)]
    varH = [np.mean(np.var(h, axis=1)) for h in H] 
    return var_entropies, var_logprobs, CSG, logprobs, pci, varH

def getScatterPlotVarEntropiesVSVarLogProbs(tokenizer, save_path, category_colors, category_labels, ouputpath='PipelineTest/results_no_remasking.pt'):
    from matplotlib.lines import Line2D
    
    var_entropies, var_logprobs, CSG, logprobs, pci, varH = getMeanVarEntropyVSMeanVarLogProbs(tokenizer, ouputpath=ouputpath)
    
    color_map = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c', 3: '#95a5a6'}  # green, orange, red, gray
    actual_colors = [color_map.get(c, '#000000') for c in category_colors]
    
    unique_cats = {}
    for color_idx, label in zip(category_colors, category_labels):
        if color_idx not in unique_cats:
            unique_cats[color_idx] = (label, color_map.get(color_idx, '#000000'))

    plt.figure(figsize=(15, 6))
    plt.scatter(var_logprobs, var_entropies, alpha=0.7, c=actual_colors)
    
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color, markersize=8, alpha=0.7)
        for cat_idx in sorted(unique_cats.keys()) for label, color in [unique_cats[cat_idx]]
    ]
    
    plt.title('Mean Variance of Entropy vs Mean Variance of LogProbs - variance done across generation steps, mean across token id')
    plt.xlabel('Mean Variance of LogProbs')
    plt.ylabel('Mean Variance of Entropy')
    plt.grid()
    plt.legend(handles=legend_handles)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'VarEntropy_vs_VarLogProbs_Sample.png'))
    plt.close()
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            var_entropies, var_logprobs, CSG, logprobs, pci, varH = getMeanVarEntropyVSMeanVarLogProbs(tokenizer, alpha=alpha, beta=beta, ouputpath=ouputpath)

            CSG = [np.mean(lp) - alpha * pci for lp, pci in zip(logprobs, pci)]
            plt.figure(figsize=(15, 6))
            plt.scatter(CSG, varH, alpha=0.7, c=actual_colors)
            plt.title(f'CSG vs Mean Variance of (p_max - p_current) - variance done across generation steps, mean across token id | alpha={alpha}')
            plt.xlabel(f'CSG (mean logprobs - {alpha} * pci)')
            plt.ylabel('Mean Variance of (p_max - p_current)')
            plt.grid()
            plt.legend(handles=legend_handles)
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'CSG_vs_VarPmax-Pcurr_Sample_alpha_{alpha}_beta_{beta}.png'))
            plt.close()
        
    



def plottestMeasure(measures, title_list, xlabel, ylabel, save_path):
    for j in range(len(measures)):
        safe_title = _safe_filename_component(title_list[j])

        plt.figure(figsize=(15, 6))
        plt.scatter(np.arange(len(measures[j])), measures[j], marker='o', color='red', alpha=0.7)
        plt.title(title_list[j])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.xticks(ticks=np.arange(len(measures[j])), rotation=90)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{safe_title}_.png"))
        plt.close()

def PlotInfo(listIndex, tokenizer, title_list, save_path=f"PipelineTest/PlotResults",  ouputpath='PipelineTest/results/outputs.pt'):
    logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_costs = GetInfoFromIndex(listIndex, tokenizer, ouputpath)
    getTxt(Proposed_sequences, save_path, proposed_sequence=True)
    getTxt(Generated_sequences, save_path, proposed_sequence=False)

    plotLogProbs(logprobs, title_list, save_path, block_size, Generated_sequences, masks)
    plotMasks(masks, title_list, save_path, block_size, Generated_sequences)
    plotChanges(changes, title_list, save_path, block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, title_list, save_path, block_size, Generated_sequences, masks)
    plotLevenshtein(levenshtein, title_list, save_path, block_size, Generated_sequences)
    plotAttentionMask(attention_masks, title_list, save_path)

    var_entropies_token_id = [np.var(e, axis=0) for e in entropies]
    plottestMeasure(var_entropies_token_id, [f'Variance_{title}_token_id' for title in title_list], xlabel="Token id", ylabel="Variance of Entropy across Tokens", save_path=save_path)

    var_entropies_generation_step = [np.var(e, axis=1) for e in entropies]
    plottestMeasure(var_entropies_generation_step, [f'Variance_{title}_generation_step' for title in title_list], xlabel="Generation Step", ylabel="Variance of Entropy across Generation Steps", save_path=save_path)
    
    logprobs_decay = [np.polyfit(np.arange(len(e)), np.array(e), deg=1)[0] for e in logprobs]
    plottestMeasure(logprobs_decay, [f'LogProbs_Decay_{title}' for title in title_list], xlabel="Generation Step", ylabel="Decay of LogProbs across Generation Steps", save_path=save_path)


    PlotlyLogProbs(logprobs, title_list, save_path, block_size, Proposed_sequences, masks, entropies=entropies)
    PlotlyUnmaskLogProbs(unmaskLogProbs, title_list, save_path, block_size, Generated_sequences, masks)
    PlotlyChanges(changes, title_list, save_path, block_size, Proposed_sequences)
    PlotlyEntropy(entropies, title_list, save_path, block_size, Proposed_sequences, masks,)

    PlotlyEntropy(entropic_costs, ['cost_'+title for title in title_list], save_path, block_size, Proposed_sequences, masks,)


def PlotInfoRemasking(listIndex, tokenizer, title_list, save_path=f"PipelineTest/PlotResultsRemasking",  ouputpath='PipelineTest/results/outputs.pt'):
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




def PlotPipeline(listIndex, tokenizer, title_list, remasking=False, save_path=f"PipelineTest/PlotResults", save_path_remasking=f"PipelineTest/PlotResultsRemasking",  ouputpath='PipelineTest/results/outputs.pt'):
    if remasking:
        PlotInfoRemasking(listIndex, tokenizer, title_list, save_path_remasking, ouputpath)
    else:
        PlotInfo(listIndex, tokenizer, title_list, save_path, ouputpath)