import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from generate_sentences_and_logprobs import concat_changes, concat_levenshtein, concat_semantic, get_generated_sentence_and_logprobs, concat_lp, concat_mask, concat_text
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from dataclasses import dataclass
import transformers

os.makedirs('LLaDa_figures', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "GSAI-ML/LLaDA-8B-Instruct"
model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)



if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.mask_token_id


messages = [
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Implement a BFS traversal in Python with clear inline comments."},
    ],
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a concise pytest that checks a Fibonacci implementation. "},
    ],
]

encoded = [
    tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True) 
    for m in messages
]
prompt_lens = torch.tensor([len(e) for e in encoded], dtype=torch.long)
max_prompt_len = prompt_lens.max().item()

prompt_tensor = torch.full((len(encoded), max_prompt_len), pad_id, dtype=torch.long) # type: ignore

for i, ids in enumerate(encoded):
    prompt_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)



prompt_tensor = prompt_tensor.to(device)
prompt_lens = prompt_lens.to(device)
max_new_tokens = 256

results = get_generated_sentence_and_logprobs(
    model,tokenizer, prompt_tensor, prompt_lens, pad_id=pad_id, steps=256, max_new_tokens=max_new_tokens, block_size=64, 
)



for j in range(len(results)):
    res = concat_lp(results[j])
    res_text = concat_text(results[j])
    res_mask = concat_mask(results[j])
    res_changes = concat_changes(results[j])[1:]
    res_changes_cumulative = np.cumsum(np.array(res_changes), axis=0)
    res_semantic_distance = concat_semantic(results[j])
    res_levenshtein_distance = concat_levenshtein(results[j])

    print(len(res_semantic_distance), len(res_semantic_distance[0]))
    print(np.sum([res_changes[j][50] for j in range(len(res_changes))]))


    plt.figure(figsize=(15, 6))
    sns.heatmap(res, cmap = 'viridis')
    plt.ylabel('Diffusion iterations')
    plt.xlabel('Word index')
    plt.title('Convergence of LogProbs: Diff. iterations VS Index (LLaDA)')
    for i in range(5):
        if i == 0:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--', label='Block size')
        else:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--')

    plt.legend()
    print('Final sentence:', " ".join(res_text[-1]))
    plt.savefig(f'LLaDa_figures/logprobs_convergence_LLaDa_{j}.png')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.imshow(res_mask, vmin=0, vmax=1, cmap='gray')
    for i in range(5):
        if i == 0:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--', label='Block size')
        else:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--')
    plt.ylabel('Diffusion iterations')
    plt.xlabel('Word index')
    plt.title('Evolution of masks across diffusion (LLaDA)')
    mask_handles = [
        Patch(facecolor='black', edgecolor='black', label='mask True'),
        Patch(facecolor='white', edgecolor='black', label='mask False'),
    ]
    plt.legend(handles=mask_handles + plt.gca().get_legend_handles_labels()[0])
    plt.savefig(f'LLaDa_figures/mask_evolution_LLaDa_{j}.png')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.imshow(res_changes, vmin=0, vmax=1, cmap='gray')
    plt.ylabel('Diffusion iterations')
    plt.xlabel('Word index')
    plt.title('Changes of words between two steps of diffusion (LLaDA)')
    for i in range(5):
        if i == 0:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--', label='Block size')
        else:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--')
    mask_handles = [
        Patch(facecolor='white', edgecolor='black', label='change True'),
        Patch(facecolor='black', edgecolor='black', label='change False'),
    ]
    plt.legend(handles=mask_handles + plt.gca().get_legend_handles_labels()[0])
    plt.savefig(f'LLaDa_figures/logprobs_changes_LLaDa_{j}.png')
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.heatmap(res_changes_cumulative, cmap='viridis', vmin=0, vmax=np.max(res_changes_cumulative))
    plt.ylabel('Diffusion iterations')
    plt.xlabel('Word index')
    plt.title('Cumulative changes in logprobs across diffusion (LLaDA)')
    for i in range(5):
        if i == 0:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--', label='Block size')
        else:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--')
    plt.legend()
    plt.savefig(f'LLaDa_figures/logprobs_cumulative_changes_LLaDa_{j}.png')
    plt.show()


    plt.figure(figsize=(15, 6))
    for i in range(len(res_semantic_distance)):
        if i%5==0:
            L = [res_semantic_distance[idx][i] for idx in range(len(res_semantic_distance))]
            plt.plot(L, label=f'Word index {i}')
    plt.ylabel('Semantic Distance')
    plt.xlabel('Diffusion iterations')
    plt.title('Semantic distance across diffusion (LLaDA, by non decreasing word index order)')
    plt.grid()
    plt.legend()
    plt.savefig(f'LLaDa_figures/semantic_distance_LLaDa_{j}.png')
    plt.show()


    plt.figure(figsize=(15, 6))
    for i in range(len(res_levenshtein_distance)):
        if i%5==0:
            L = [res_levenshtein_distance[idx][i] for idx in range(len(res_levenshtein_distance))]
            plt.plot(L, label=f'Word index {i}')
    plt.ylabel('Levenshtein Distance')
    plt.xlabel('Diffusion iterations')
    plt.title('Levenshtein distance across diffusion (LLaDA)')
    plt.grid()
    plt.legend()
    plt.savefig(f'LLaDa_figures/levenshtein_distance_LLaDa_{j}.png')
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.heatmap(res_levenshtein_distance, cmap='viridis', vmin=0, vmax=np.max(res_levenshtein_distance))
    for i in range(5):
        if i == 0:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--', label='Block size')
        else:
            plt.axvline(64*i, linewidth=4, color='red', linestyle='--')
    plt.ylabel('Diffusion iterations')
    plt.xlabel('Word index order')
    plt.title('Levenshtein distance: Diffusion VS word index order (LLaDA)')
    plt.grid()
    plt.legend()
    plt.savefig(f'LLaDa_figures/heatmap_levenshtein_distance_LLaDa_{j}.png')
    plt.show()






# for i in range(len(results)):
#     print(f'==================== MESSAGE {i} ====================')
#     for j in results[i].keys():
#         print(f'------------- BLOCK {j} -------------')
#         for k in (results[i][j].keys()):
#             print(k, '.', results[i][j][k][0])
