
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from PipelineTest.AnalyseResults import PlotPipeline, getScatterPlotVarEntropiesVSVarLogProbs

OUTPUTS_PATH  = 'PipelineTest/results_64_step/outputs_64_step.pt'
TOKENIZER_PATH = 'PipelineTest/results_64_step/tokenizer_64_step.pt'
EVAL_JSON     = 'PipelineTest/eval/results_triviaqa_64_step.json'
N_PER_CAT     = 10

tokenizer = torch.load(TOKENIZER_PATH, map_location="cpu", weights_only=False)

with open(EVAL_JSON, encoding="utf-8") as f:
    data = json.load(f)

by_index = {d["index"]: d for d in data}

correct = [4, 9, 10, 13, 15, 17, 18, 19, 20, 21, 24, 26, 28, 29, 31, 33, 34, 36, 45, 48, 49, 54, 61, 62, 65, 66, 67, 68, 69, 70, 75, 76, 78, 79, 80, 82, 84, 88, 90, 91, 93, 94, 103, 104, 105, 106, 107, 109, 115, 116, 119, 120, 122, 123, 124, 130, 132, 133, 134, 135, 137, 146, 149, 151, 152, 155, 156, 157, 158, 160, 163, 164, 166, 168, 169, 170, 171, 172, 173, 175, 177, 178, 179, 181, 182, 183, 184, 185, 188, 190, 193, 202, 203, 204, 205, 211, 214, 215, 216, 219, 220, 221, 227, 229, 230, 232, 233, 236, 237, 240, 242, 245, 246, 247, 248, 252, 253, 255]
partial = [0, 6, 25, 30, 35, 42, 47, 53, 55, 56, 71, 72, 73, 95, 96, 97, 136, 145, 148, 159, 165, 174, 176, 187, 191, 194, 196, 198, 213, 218, 223, 224, 238, 243, 254]
fabricated = [1, 2, 3, 5, 7, 8, 11, 12, 14, 16, 22, 23, 27, 32, 37, 38, 39, 40, 41, 43, 44, 46, 50, 51, 52, 57, 58, 59, 60, 63, 64, 74, 77, 81, 83, 85, 86, 87, 89, 92, 98, 99, 100, 101, 102, 108, 110, 111, 112, 113, 114, 117, 118, 121, 125, 126, 127, 128, 129, 131, 138, 139, 140, 141, 142, 143, 144, 147, 150, 153, 154, 161, 162, 167, 180, 186, 189, 192, 195, 197, 199, 200, 201, 206, 207, 209, 210, 212, 217, 222, 225, 226, 228, 231, 234, 235, 239, 241, 244, 249, 250, 251]
empty_garbled = [208]

def _short_title(entry, cat_label):
    q = entry["question"]
    q = q if len(q) <= 55 else q[:52] + "..."
    return f"[{cat_label}] {q}"

categories = [
    ("correct",        correct,       "OK"),
    ("partial_hallu",  partial,       "PARTIAL"),
    ("fabricated",     fabricated,    "FABRICATED"),
    ("empty_garbled",  empty_garbled, "EMPTY"),
]

# for folder_name, indices, cat_label in categories:
#     sample_indices = indices[:N_PER_CAT]
#     if not sample_indices:
#         print(f"[{cat_label}] no examples, skipping.")
#         continue

#     title_list = [_short_title(by_index[i], cat_label) for i in sample_indices]
#     save_path  = f"PipelineTest/results_no_remasking/Plots_hallucination_entropic_cost/{folder_name}"

#     print(f"\n=== {cat_label} ({len(sample_indices)} examples) ===")
#     for i, t in zip(sample_indices, title_list):
#         print(f"  idx {i}: {t}")

#     PlotPipeline(
#         sample_indices,
#         tokenizer,
#         title_list,
#         remasking=False,
#         ouputpath=OUTPUTS_PATH,
#         save_path=save_path,
#     )

def category(i):
    if i in correct:
        return 0, 'Correct'
    elif i in partial:
        return 1, 'Partial'
    elif i in fabricated:
        return 2, 'Fabricated'
    elif i in empty_garbled:
        return 3, 'Empty/Garbled'
    else:
        raise ValueError(f"Index {i} not found in any category")
    

category_colors =[category(i)[0] for i in range(len(by_index))] # crée une liste de 0 1 2 3 correspondant à chaque catégorie
category_labels = [category(i)[1] for i in range(len(by_index))]
save_path = 'PipelineTest/results_64_step'
getScatterPlotVarEntropiesVSVarLogProbs(tokenizer, save_path, category_colors, category_labels, ouputpath=OUTPUTS_PATH)


