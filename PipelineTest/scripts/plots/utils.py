"""
plots.utils
============
Shared utilities for all plot modes.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sklearn.model_selection import train_test_split

from PipelineTest.features.utils import (
    load_outputs, match_samples, get_pad_token_id, compute_padding_mask,
)
from PipelineTest.features.ar1 import fit_ar1_models
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import (
    getEntropy, getEachStepMask,
)


# =========================================================================
# CONFIGS
# =========================================================================



CONFIGS = {
    "llada": {
        "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence/outputs_LLaDa_64steps_64tokens_low_confidence.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence/tokenizer_LLaDa_64steps_64tokens_low_confidence.pt",
        "name": "LLaDA_64steps_64tokens",
    },
    "dream": {
        "outputs_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json",
        "tokenizer_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/tokenizer_DREAM_64steps_64tokens_maskgit.pt",
        "name": "DREAM_64steps_64tokens",
    },
    "dream16": {
        "outputs_path": "PipelineTest/res/results_DREAM_16steps_32tokens_maskgit/outputs_DREAM_16steps_32tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_16steps_32tokens_maskgit.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_16steps_32tokens_maskgit/tokenizer_DREAM_16steps_32tokens_maskgit.pt",
        "name": "DREAM_16steps_32tokens",
    },
    "dream128": {
        "outputs_path": "PipelineTest/res/results_DREAM_128steps_128tokens_maskgit/outputs_DREAM_128steps_128tokens_maskgit.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_128steps_128tokens_maskgit.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_128steps_128tokens_maskgit/tokenizer_DREAM_128steps_128tokens_maskgit.pt",
        "name": "DREAM_128steps_128tokens",
    },
    "dream64_800samples": {
        "outputs_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_800samples/outputs_DREAM_64steps_64tokens_maskgit_800samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit_800samples.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_800samples/tokenizer_DREAM_64steps_64tokens_maskgit_800samples.pt",
        "name": "DREAM_64steps_64tokens_800samples",
    },
    "llada64_800samples": {
        "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_800samples/outputs_LLaDa_64steps_64tokens_low_confidence_800samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence_800samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_800samples/tokenizer_LLaDa_64steps_64tokens_low_confidence_800samples.pt",
        "name": "LLaDA_64steps_64tokens_800samples",
    },
    "dream64_400samples": {
        "outputs_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_400samples/outputs_DREAM_64steps_64tokens_maskgit_400samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit_400samples.json",
        "tokenizer_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_400samples/tokenizer_DREAM_64steps_64tokens_maskgit_400samples.pt",
        "name": "DREAM_64steps_64tokens_400samples",
    },
    "llada64_400samples": {
        "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_400samples/outputs_LLaDa_64steps_64tokens_low_confidence_400samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence_400samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_400samples/tokenizer_LLaDa_64steps_64tokens_low_confidence_400samples.pt",
        "name": "LLaDA_64steps_64tokens_400samples",
    },
    "llada128_400samples": {
        "outputs_path": "PipelineTest/res/results_LLaDa_128steps_128tokens_low_confidence_400samples/outputs_LLaDa_128steps_128tokens_low_confidence_400samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_128steps_128tokens_low_confidence_400samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_128steps_128tokens_low_confidence_400samples/tokenizer_LLaDa_128steps_128tokens_low_confidence_400samples.pt",
        "name": "LLaDA_128steps_128tokens_400samples",
    },
    "llada16_32tokens": {
        "outputs_path": "PipelineTest/res/results_LLaDa_16steps_32tokens_low_confidence_samples/outputs_LLaDa_16steps_32tokens_low_confidence_samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_16steps_32tokens_low_confidence_samples.json",
        "tokenizer_path": "PipelineTest/res/results_LLaDa_16steps_32tokens_low_confidence_samples/tokenizer_LLaDa_16steps_32tokens_low_confidence_samples.pt",
        "name": "LLaDA_16steps_32tokens",
    },
    "dream16_32tokens_2100samples": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples/outputs_dream_16steps_32tokens_triviaqa_2100samples.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples/tokenizer_dream_16steps_32tokens_triviaqa_2100samples.pt",
        "name": "dream_16steps_32tokens_2100samples",
    },
    "dream16_32tokens_2100samples_naturalquestion": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2100samples/outputs_dream_16steps_32tokens_naturalquestion_2100samples.pt",
        "eval_json": "PipelineTest/res/eval/results_naturalquestion_dream_16steps_32tokens_naturalquestion_2100samples.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2100samples/tokenizer_dream_16steps_32tokens_naturalquestion_2100samples.pt",
        "name": "dream_16steps_32tokens_naturalquestion_2100samples",
    },
    "llada16_32tokens_2100samples_naturalquestion": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2100samples/outputs_llada_16steps_32tokens_naturalquestion_2100samples.pt",
        "eval_json": "PipelineTest/res/eval/results_naturalquestion_llada_16steps_32tokens_naturalquestion_2100samples.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2100samples/tokenizer_llada_16steps_32tokens_naturalquestion_2100samples.pt",
        "name": "llada_16steps_32tokens_naturalquestion_2100samples",
},
    "llada16_32tokens_2100samples_hotpotqa": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2100samples/outputs_llada_16steps_32tokens_hotpotqa_2100samples.pt",
        "eval_json": "PipelineTest/res/eval/results_hotpotqa_llada_16steps_32tokens_hotpotqa_2100samples.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2100samples/tokenizer_llada_16steps_32tokens_hotpotqa_2100samples.pt",
        "name": "llada_16steps_32tokens_hotpotqa_2100samples",
},
    "dream16_32tokens_2100samples_hotpotqa": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2100samples/outputs_dream_16steps_32tokens_hotpotqa_2100samples.pt",
        "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_16steps_32tokens_hotpotqa_2100samples.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2100samples/tokenizer_dream_16steps_32tokens_hotpotqa_2100samples.pt",
        "name": "dream_16steps_32tokens_hotpotqa_2100samples",
},
    "llada16_32tokens_2100samples_triviaqa_without_collapse": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60/outputs_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60/tokenizer_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.pt",
        "name": "llada16_32tokens_2100samples_triviaqa_without_collapse",
},      
    "dream16_32tokens_2100samples_triviaqa_without_collapse": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60/outputs_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60/tokenizer_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.pt",
        "name": "dream16_32tokens_2100samples_triviaqa_without_collapse",
}


}



# =========================================================================
# DATA LOADING CONTEXT
# =========================================================================

class PlotContext:
    """Holds all data needed for plotting, loaded once and shared across modes."""

    def __init__(self, config_name="llada", use_padding=True):
        cfg = CONFIGS[config_name]
        self.cfg = cfg
        self.config_name = config_name

        # Load outputs
        self.outputs = load_outputs(cfg["outputs_path"])
        self.positions, self.labels, self.data = match_samples(cfg["eval_json"], self.outputs)

        # Entropies and masks
        self.entropies = getEntropy(self.outputs)
        self.masks = getEachStepMask(self.outputs)

        # Pad token
        self.pad_token_id = None
        self.use_padding = use_padding
        if use_padding:
            try:
                self.pad_token_id = get_pad_token_id(cfg["outputs_path"])
            except Exception:
                self.use_padding = False

        # Padding mask (N_all, D) for matched samples
        self.padding_2d = None
        if self.use_padding and self.pad_token_id is not None:
            padding_full = compute_padding_mask(self.outputs, self.positions, self.pad_token_id)
            self.padding_2d = padding_full[:, 0, :]  # (N, D)

        # Split 50/50 for AR1
        self.ar1_idx, self.eval_idx = train_test_split(
            np.arange(len(self.positions)), test_size=0.5,
            stratify=self.labels, random_state=42,
        )

        # AR1 models (lazy)
        self._ar1_models = None
        self._ar1_models_no_pad = None

    @property
    def ar1_positions(self):
        return self.positions[self.ar1_idx]

    @property
    def ar1_labels(self):
        return self.labels[self.ar1_idx]

    @property
    def eval_positions(self):
        return self.positions[self.eval_idx]

    @property
    def eval_labels(self):
        return self.labels[self.eval_idx]

    def get_ar1_models(self, no_padding=True):
        """Fit AR1 models (cached)."""
        if no_padding and self.padding_2d is not None:
            if self._ar1_models_no_pad is None:
                pad_ar1 = self.padding_2d[self.ar1_idx]
                self._ar1_models_no_pad = fit_ar1_models(
                    self.outputs, self.ar1_positions, self.ar1_labels,
                    padding_2d=pad_ar1,
                )
            return self._ar1_models_no_pad
        else:
            if self._ar1_models is None:
                self._ar1_models = fit_ar1_models(
                    self.outputs, self.ar1_positions, self.ar1_labels,
                    padding_2d=None,
                )
            return self._ar1_models

    def get_save_dir(self, mode_name):
        """Get save directory for a given mode."""
        base = os.path.join(os.path.dirname(self.cfg["outputs_path"]), "plots", mode_name)
        os.makedirs(base, exist_ok=True)
        return base

    def select_samples(self, n_samples=20, from_eval_half=True):
        """Select balanced samples for plotting.
        
        Returns: indices (into positions/labels), corresponding positions, labels
        """
        if from_eval_half:
            idx_pool = self.eval_idx
        else:
            idx_pool = np.arange(len(self.positions))

        pool_labels = self.labels[idx_pool]
        correct_in_pool = idx_pool[pool_labels == 0]
        halluc_in_pool = idx_pool[pool_labels == 1]

        np.random.seed(42)
        n_per = n_samples // 2
        sel_c = np.random.choice(correct_in_pool, min(n_per, len(correct_in_pool)), replace=False)
        sel_h = np.random.choice(halluc_in_pool, min(n_per, len(halluc_in_pool)), replace=False)
        selected = np.concatenate([sel_c, sel_h])

        return selected, self.positions[selected], self.labels[selected]
