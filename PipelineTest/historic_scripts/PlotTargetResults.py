
from random import random
import sys
import os
import json

from transformers import AutoTokenizer
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getNumTransferTokens, getSemanticEntropy, getUnmaskLogProbs, getSemanticBestLogprobsLabel
from GenerateBaseSamplerOutputsAndExtractInfo.PlotResults import PlotlyLogProbs

import torch
from sklearn.model_selection import train_test_split
from PipelineTest.historic_scripts.AnalyseResults import PlotDistributionTau, PlotPipeline, PlotMeanMaskedEntropyAcrossTokens, PlotScatterVarMaskedEntropyAcrossTokensVsAlphaEntropyAvg, PlotMeanVarMaskedAcrossTokens, mergeOutputsList, plotEntropyForSteps

OUTPUTS_PATH  = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt'
TOKENIZER_PATH = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/tokenizer_DREAM_64steps_64tokens_maskgit.pt'
EVAL_JSON     = 'PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json'
SAVE_PATH     = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/TestVar'
N_PER_CAT     = 10

tokenizer = torch.load(TOKENIZER_PATH, map_location="cpu", weights_only=False)


with open(EVAL_JSON, encoding="utf-8") as f:
    data = json.load(f)


def filter_by_answer_length(indices, min_words=5):
        filtered = []
        for idx in indices:
            entry = by_index[idx]
            answer = entry.get("answer", "")
            if len(answer.split()) >= min_words:
                filtered.append(idx)
        return filtered



by_index = {d["index"]: d for d in data}

correct = [d["index"] for d in data if d["is_hallucination"] == 'no']
hallucinations = [d["index"] for d in data if d["is_hallucination"] == 'yes']

# --- Filter to logreg half (same split as AR1_SamplePlots/AR1_TrainTest) ---
outputs = mergeOutputsList(OUTPUTS_PATH)
hmap = {}
for i in range(len(outputs.sample_indices)):
    j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
    hmap[j] = i

samples_all = []
for d in data:
    idx = d["index"]
    if idx in hmap:
        label = 0 if d["is_hallucination"] == 'no' else 1
        samples_all.append((hmap[idx], label, idx))

positions_all = np.array([s[0] for s in samples_all])
labels_all = np.array([s[1] for s in samples_all])
_, logreg_idx = train_test_split(np.arange(len(samples_all)), test_size=0.5, stratify=labels_all, random_state=42)
logreg_positions_set = set(positions_all[logreg_idx])
logreg_orig_indices = set(s[2] for s in samples_all if s[0] in logreg_positions_set)

correct = [idx for idx in correct if idx in logreg_orig_indices]
hallucinations = [idx for idx in hallucinations if idx in logreg_orig_indices]
print(f"Filtered to logreg half: {len(correct)} correct, {len(hallucinations)} halluc")
# --- End filter ---

correct_filtered = filter_by_answer_length(correct, min_words=5)
hallucinations_filtered = filter_by_answer_length(hallucinations, min_words=5)
print('nb correct filtered', len(correct) - len(correct_filtered))
print('nb hallucinations filtered', len(hallucinations) - len(hallucinations_filtered))
# shuffle 
np.random.seed(42)
np.random.shuffle(correct)
np.random.shuffle(hallucinations)

print('nb correct: ', len(correct))
print('nb hallucinations: ', len(hallucinations))

def _short_title(entry, cat_label):
    q = entry["question"]
    q = q if len(q) <= 55 else q[:52] + "..."
    return f"[{cat_label}] {q}"

categories = [
    ("correct", correct, "OK"),
    ("hallucinations",  hallucinations,  "Hallucinations"),
]

for folder_name, indices, cat_label in categories:

    sample_indices = indices[:N_PER_CAT]
    if not sample_indices:
        print(f"[{cat_label}] no examples, skipping.")
        continue

    title_list = [_short_title(by_index[i], cat_label) for i in sample_indices]
    save_path  = os.path.join(SAVE_PATH, folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n=== {cat_label} ({len(sample_indices)} examples) ===")
    for i, t in zip(sample_indices, title_list):
        print(f"  idx {i}: {t}")
    
    # outputs = mergeOutputsList(OUTPUTS_PATH)
    # hmap = {}
    # for i in range(len(outputs.sample_indices)):
    #     j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
    #     hmap[j] = i
    # l_index = [hmap[i] for i in sample_indices]
    # logprobs = getLogProbs(outputs)
    # block_size = outputs.block_size
    # entropies = getEntropy(outputs)
    # tokenizer = AutoTokenizer.from_pretrained( "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)

    # Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    # masks = getEachStepMask(outputs)

    # masks = [masks[i] for i in l_index]
    # entropies = [entropies[i] for i in l_index]
    # proposed_sequences = [Proposed_sequences[i] for i in l_index]
    # logprobs = [logprobs[i] for i in l_index]
    # PlotlyLogProbs(logprobs, title_list, save_path, block_size, proposed_sequences, masks,  entropies=entropies)

    PlotPipeline(
        sample_indices,
        tokenizer,
        title_list,
        remasking=False,
        ouputpath=OUTPUTS_PATH,
        save_path=save_path,
    )

# PlotScatterVarMaskedEntropyAcrossTokensVsAlphaEntropyAvg(
#     correct,
#     hallucinations,
#     SAVE_PATH,
#     OUTPUTS_PATH
# )

# PlotMeanVarMaskedAcrossTokens(correct,
#     hallucinations,
#     SAVE_PATH,
#     OUTPUTS_PATH
# )

# PlotMeanVarMaskedAcrossTokens(correct_filtered,
#     hallucinations_filtered,
#     SAVE_PATH + "/filtered",
#     OUTPUTS_PATH
# )

# PlotMeanMaskedEntropyAcrossTokens(correct,
#     hallucinations,
#     SAVE_PATH,
#     OUTPUTS_PATH
# )

# PlotDistributionTau(correct,
#     hallucinations,
#     SAVE_PATH,
#     OUTPUTS_PATH
# )

# plotEntropyForSteps(correct,
#     hallucinations,
#     start_id=15,
#     end_id=30,
#     save_path=SAVE_PATH,
#     outputpath=OUTPUTS_PATH
# )