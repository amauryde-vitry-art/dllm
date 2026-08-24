"""Inspect the 10 samples with the highest VarEntropy: show text and label."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from transformers import AutoTokenizer
from PipelineTest.historic_scripts.AnalyseResults import mergeOutputsList
from PipelineTest.historic_scripts.CreateMetrics import getFromOutputsVarEntropy, getLabels

RESULTS_DIR = "PipelineTest/res/results_semantic_dispersion_and_semantic_entropy_3k_stc_2048_samples"
outputpath = f"{RESULTS_DIR}/outputs_semantic_dispersion_and_semantic_entropy_3k_stc_2048_samples.pt"

# Load outputs and compute VarEntropy
outputs = mergeOutputsList(outputpath)
var_entropy = getFromOutputsVarEntropy(outputs)

# Load all labels indexed by global triviaqa index
all_labels = getLabels()

# Tokenizer for decoding
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

# Use sample_indices if available for correct label alignment
if outputs.sample_indices is not None:
    sample_indices = outputs.sample_indices.numpy()
    labels = all_labels[sample_indices]
    print(f"Using sample_indices for label alignment (found {len(sample_indices)} indices)")
else:
    labels = all_labels
    sample_indices = np.arange(len(labels))
    print("WARNING: No sample_indices in outputs, assuming positional alignment")

# Top 10 indices by VarEntropy (descending)
top10_idx = np.argsort(var_entropy)[-10:][::-1]

print("=" * 80)
print("TOP 10 INDICES BY VarEntropy")
print("=" * 80)
for rank, idx in enumerate(top10_idx, 1):
    label_str = "HALLUCINATION" if labels[idx] == 1 else "CORRECT"
    global_idx = sample_indices[idx]
    # sequences has shape [n_samples, seq_len]
    text = tokenizer.decode(outputs.sequences[idx], skip_special_tokens=True)
    print(f"\n{'─' * 80}")
    print(f"Rank {rank} | Position {idx} | Global Index {global_idx} | VarEntropy = {var_entropy[idx]:.6f} | Label: {label_str}")
    print(f"{'─' * 80}")
    print(text[:2000])  # Limit display length
    print()
