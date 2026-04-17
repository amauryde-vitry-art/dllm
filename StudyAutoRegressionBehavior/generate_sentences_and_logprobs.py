import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Levenshtein import distance


def semantic_similarity(model, id1, id2):
    embeddings = model.get_input_embeddings()
    
    vec1 = embeddings(torch.tensor([id1]).to(model.device))
    vec2 = embeddings(torch.tensor([id2]).to(model.device))
    
    return F.cosine_similarity(vec1, vec2).item()


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def get_generated_sentence_and_logprobs(model, tokenizer, prompt, prompt_lens, pad_id, steps=256, max_new_tokens=256, block_size=64):
    mask_id = tokenizer.mask_token_id # type: ignore

    # If model doesn't have mask_token (e.g., causal models), use pad_token as mask
    if mask_id is None:
        mask_id = 126336 
    
    batch_size = prompt.size(0)
    
    total_length = int(prompt_lens.max().item() + max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=model.device)
    
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt[i, :length]
        x[i, length : length + max_new_tokens] = mask_id

    positions = torch.arange(total_length, device=x.device)
    
    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    res_list = [defaultdict(dict) for _ in range(batch_size)]

    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size
        
        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        
        num_transfer_tokens = get_num_transfer_tokens(init_block_mask, steps_per_block)
        
        history_generated = [x.clone()]

        for i in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )


            with torch.no_grad(): 
                logits = model(x).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            

            changes = torch.zeros_like(x, dtype=torch.bool)
            changes[block_mask] = history_generated[-1][block_mask] != x0[block_mask]


            
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            confidence = torch.full_like(x0_p, -np.inf)
            confidence = torch.where(block_mask, x0_p, confidence)

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for j in range(batch_size):
                k = int(num_transfer_tokens[j, i].item())
                if k == 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            
            
            x[transfer_index] = x0[transfer_index]

            block_mask_history = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )
            
           
            for j in range(batch_size):               
                current_block_probs = x0_p[j][init_block_mask[j]].detach().cpu().to(torch.float32)
                semantic_distance_j = [semantic_similarity(model, history_generated[-1][j][idx], x0[j, idx].item()) for idx in range(block_start[j].item(), block_end[j].item())]
                tokens_block = x0[j][init_block_mask[j]] 
                token_bloc_history = history_generated[-1][j][init_block_mask[j]]
        
                words_list = [tokenizer.decode(t).strip() for t in tokens_block] # type: ignore
                words_list_history = [tokenizer.decode(t).strip() for t in token_bloc_history] # type: ignore
                current_words_list = [tokenizer.decode(t).strip() for t in x[j][init_block_mask[j]]] # type: ignore

                levenshtein_distances = [distance(word1, word2) for word1, word2 in zip(words_list, words_list_history)]

                start_idx = block_start[j].item()
                end_idx = block_end[j].item()
                res_list[j][num_block][i] = (current_words_list, current_block_probs, block_mask_history[j][start_idx:end_idx], changes[j][start_idx:end_idx], semantic_distance_j, levenshtein_distances)
            
            history_generated.append(x0.clone())

    return res_list



def concat_lp(res):    
    final_logprobs = []
    nb_iterations = len(res[0].keys())
    for i in range(nb_iterations):
        lp_list = []
        for key in res.keys():
            logprobs = res[key][i][1].detach().cpu().to(torch.float32).numpy()
            lp_list.append(logprobs)
        final_logprobs.append(np.concatenate(lp_list, axis=0))
    return (final_logprobs)


def concat_text(res):
    final_text = []
    nb_iterations = len(res[0].keys())
    for i in range(nb_iterations):
        text_list = []
        for key in res.keys():
            text = res[key][i][0]
            text_list.append(text)
        final_text.append(np.concatenate(text_list, axis=0))
    return (final_text)


def concat_mask(res):    
    final_mask = []
    nb_iterations = len(res[0].keys())
    for i in range(nb_iterations):
        mask_list = []
        for key in res.keys():
            masks = res[key][i][2].detach().cpu().to(torch.float32).numpy()
            mask_list.append(masks)
        final_mask.append(np.concatenate(mask_list, axis=0))
    return (final_mask)

def concat_changes(res):
    final_changes = []
    nb_iterations = len(res[0].keys())
    for i in range(nb_iterations):
        changes_list = []
        for key in res.keys():
            changes = res[key][i][3].detach().cpu().to(torch.float32).numpy()
            changes_list.append(changes)
        final_changes.append(np.concatenate(changes_list, axis=0))
    return (final_changes)

def concat_semantic(res):
    final_semantic = []
    nb_iterations = len(res[0].keys())
    for i in range(nb_iterations):
        semantic_list = []
        for key in res.keys():
            semantic = np.array(res[key][i][4])
            semantic_list.append(semantic)
        final_semantic.append(np.concatenate(semantic_list, axis=0))
    return final_semantic

def concat_levenshtein(res):
    final_levenshtein = []
    nb_iterations = len(res[0].keys())
    for i in range(nb_iterations):
        levenshtein_list = []
        for key in res.keys():
            levenshtein = np.array(res[key][i][5])
            levenshtein_list.append(levenshtein)
        final_levenshtein.append(np.concatenate(levenshtein_list, axis=0))
    return final_levenshtein