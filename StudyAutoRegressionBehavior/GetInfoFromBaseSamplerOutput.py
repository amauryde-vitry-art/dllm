import dllm
import torch
from Levenshtein import distance



def getLogProbs(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory) -> list[torch.Tensor]:
    nb_examples = outputs.histories_logprobs[0].shape[0]
    res_logProbs =[]
    for i in range(nb_examples):
        res_logProbs.append([e[i, outputs.start_idx_history[i]:outputs.start_idx_history[i]+outputs.max_new_tokens].detach().float().cpu().numpy() for e in outputs.histories_logprobs])
    return res_logProbs


def getUnmaskLogProbs(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory) -> list[torch.Tensor]:
    nb_examples = outputs.histories_unmask_logprobs[0].shape[0]
    res_unmaskLogProbs =[]
    for i in range(nb_examples):
        res_unmaskLogProbs.append([e[i, outputs.start_idx_history[i]:outputs.start_idx_history[i]+outputs.max_new_tokens].detach().float().cpu().numpy() for e in outputs.histories_unmask_logprobs])
    return res_unmaskLogProbs



def getEachStepGeneratedSequence(outputs: dllm.core.samplers.BaseSamplerOutputCompleteHistory, tokenizer) -> list[list[str]]:
    nb_examples = outputs.histories_x[0].shape[0]
    res = []
    
    for i in range(nb_examples):
        start_idx = outputs.start_idx_history[i]
        end_idx = start_idx + outputs.max_new_tokens
        example_history_tokens = []
        for e in outputs.histories_x:
            token_ids = e[i, start_idx:end_idx]
            tokens = [tokenizer.decode(t) for t in token_ids]
            example_history_tokens.append(tokens)
            
        res.append(example_history_tokens)
        
    return res

def getEachStepProposedSequence(outputs: dllm.core.samplers.BaseSamplerOutputCompleteHistory, tokenizer) -> list[list[str]]:
    nb_examples = outputs.histories_x0[0].shape[0]
    res = []
    
    for i in range(nb_examples):
        start_idx = outputs.start_idx_history[i]
        end_idx = start_idx + outputs.max_new_tokens
        example_history_tokens = []
        for e in outputs.histories_x0:
            token_ids = e[i, start_idx:end_idx]
            tokens = [tokenizer.decode(t) for t in token_ids]
            example_history_tokens.append(tokens)
            
        res.append(example_history_tokens)
        
    return res


def getEachStepMask(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory, remask = False) -> list[torch.Tensor]:
    nb_examples = outputs.histories_mask[0].shape[0]
    res_masks =[]
    for i in range(nb_examples):
        m = outputs.histories_remasking if remask else outputs.histories_mask
        res_masks.append([e[i, outputs.start_idx_history[i]:outputs.start_idx_history[i]+outputs.max_new_tokens].detach().float().cpu().numpy() for e in m]) 
    return res_masks



def getEachStepChange(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    stacked_all = torch.stack(outputs.histories_x0)
    diff_mask = (stacked_all[1:] != stacked_all[:-1])
    diff_mask = diff_mask.permute(1, 0, 2)
    res_changes = []
    for i in range(diff_mask.shape[0]):
        start_idx = outputs.start_idx_history[i]

        example_diff = diff_mask[i, :, start_idx:start_idx+outputs.max_new_tokens]
        zero_row = torch.zeros(
            (1, example_diff.shape[1]), dtype=example_diff.dtype, device=example_diff.device
        )
        example_diff = torch.cat([zero_row, example_diff], dim=0)

        res_changes.append(example_diff.detach().cpu().numpy())
        
    return res_changes


from Levenshtein import distance
import numpy as np

def getLevenshtein(Proposed_sequences):
    # Proposed_sequences structure: [Batch, Steps, 128 tokens]
    res_levenshtein = []
    
    for example_history in Proposed_sequences:
        # example_history: [Steps, 128 tokens]
        nb_steps = len(example_history)
        example_changes = []
        
        for k in range(1, nb_steps):
            # On compare chaque mot à l'index 'm' entre l'étape k et k-1
            # On obtient une liste de 128 distances pour cette transition
            step_distances = [
                distance(example_history[k-1][m], example_history[k][m])
                for m in range(len(example_history[k]))
            ]
            example_changes.append(step_distances)
            
        # On convertit en array numpy pour faciliter le plot (Shape: [Steps-1, 128])
        res_levenshtein.append(np.array(example_changes))
        
    return res_levenshtein

def getH(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_H[0].shape[0]
    res_H =[]
    for i in range(nb_examples):
        res_H.append([e[i, outputs.start_idx_history[i]:outputs.start_idx_history[i]+outputs.max_new_tokens].detach().float().cpu().numpy() for e in outputs.histories_H]) 
    return res_H


def getNumTransferTokens(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_num_transfer_tokens[0].shape[0]
    res_num_transfer_tokens =[]
    for i in range(nb_examples):
        res_num_transfer_tokens.append([e[i].detach().float().cpu().numpy() for e in outputs.histories_num_transfer_tokens]) 
    return res_num_transfer_tokens



