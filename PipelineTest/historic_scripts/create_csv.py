

import sys

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getLogProbs, getEntropy, getEachStepMask
import pandas as pd
import json
import numpy as np
from PipelineTest.historic_scripts.AnalyseResults import mergeOutputsList

OUTPUTS_PATH  = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt'
EVAL_JSON     = 'PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json'


with open(EVAL_JSON, encoding="utf-8") as f:
    data = json.load(f)

by_index = {d["index"]: d for d in data}

correct = [d["index"] for d in data if d["is_hallucination"] == 'no']
hallucinations = [d["index"] for d in data if d["is_hallucination"] == 'yes']
print('nb correct: ', len(correct))
print('nb hallucinations: ', len(hallucinations))

outputs = mergeOutputsList(OUTPUTS_PATH)
entropies = getEntropy(outputs)
mask_list = getEachStepMask(outputs)
d = {"Diffusion Step": [], "VarEntropy": [], "SampleIndex": [], "isHallucination": [], "mean_entropy": []}
for i in range(len(outputs.sample_indices)):
    j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
    sample_entropies = entropies[j]
    sample_entropies = np.asarray(sample_entropies)
    masks = np.asarray(mask_list[j], dtype=bool)
    for step in range(len(sample_entropies)):
        masked_entropies = []
        for k in range(len(sample_entropies[step])):
            if masks[step, k]:
                masked_entropies.append(sample_entropies[step, k])
        if len(masked_entropies) > 0:
            d["Diffusion Step"].append(step)
            d["VarEntropy"].append(np.var(masked_entropies))
            d["mean_entropy"].append(np.mean(masked_entropies))
            d["SampleIndex"].append(j)
            d["isHallucination"].append(by_index[j]["is_hallucination"])

df = pd.DataFrame(d)

# save the DataFrame to a CSV file
csv_save_path = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/VarAndMeanEntropy.csv'
df.to_csv(csv_save_path, index=False)
print(f"CSV file saved to: {csv_save_path}")

df_groupby = df.groupby(["Diffusion Step", "isHallucination"])[["VarEntropy", "mean_entropy"]].mean().reset_index()
# save the grouped DataFrame to a CSV file
csv_groupby_save_path = 'PipelineTest/res/DREAM_64steps_64tokens_maskgit/AnalyseResults/VarAndMeanEntropy_grouped.csv'
df_groupby.to_csv(csv_groupby_save_path, index=False)
print(f"Grouped CSV file saved to: {csv_groupby_save_path}")
