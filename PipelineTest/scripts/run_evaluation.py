"""
scripts/run_evaluation.py
=========================
Centralized evaluation script.

Computes all features (baseline + markovian + AR1 + semantic), trains classifiers
(Logistic Regression with GridSearchCV), and reports metrics (ROC-AUC, PR-AUC, accuracy).

Usage:
    python -m PipelineTest.scripts.run_evaluation [--config CONFIG]
"""

import sys
import os
import json
import argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from PipelineTest.features.utils import load_outputs, match_samples, get_pad_token_id, compute_padding_mask
from PipelineTest.features.baseline import get_baseline_features
from PipelineTest.features.markovian import get_markovian_features
from PipelineTest.features.ar1 import fit_ar1_models, get_ar1_features
from PipelineTest.features.semantic import get_semantic_features
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask


# =========================================================================
# CONFIG
# =========================================================================


# CONFIGS = {
#     "llada": {
#         "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence/outputs_LLaDa_64steps_64tokens_low_confidence.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence.json",
#         "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence/tokenizer_LLaDa_64steps_64tokens_low_confidence.pt",
#         "name": "LLaDA_64steps_64tokens",
#     },
#     "dream": {
#         "outputs_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/outputs_DREAM_64steps_64tokens_maskgit.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit.json",
#         "tokenizer_path": "PipelineTest/res/DREAM_64steps_64tokens_maskgit/tokenizer_DREAM_64steps_64tokens_maskgit.pt",
#         "name": "DREAM_64steps_64tokens",
#     },
#     "dream16": {
#         "outputs_path": "PipelineTest/res/results_DREAM_16steps_32tokens_maskgit/outputs_DREAM_16steps_32tokens_maskgit.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_16steps_32tokens_maskgit.json",
#         "tokenizer_path": "PipelineTest/res/results_DREAM_16steps_32tokens_maskgit/tokenizer_DREAM_16steps_32tokens_maskgit.pt",
#         "name": "DREAM_16steps_32tokens",
#     },
#     "dream128": {
#         "outputs_path": "PipelineTest/res/results_DREAM_128steps_128tokens_maskgit/outputs_DREAM_128steps_128tokens_maskgit.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_128steps_128tokens_maskgit.json",
#         "tokenizer_path": "PipelineTest/res/results_DREAM_128steps_128tokens_maskgit/tokenizer_DREAM_128steps_128tokens_maskgit.pt",
#         "name": "DREAM_128steps_128tokens",
#     },
#     "dream64_800samples": {
#         "outputs_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_800samples/outputs_DREAM_64steps_64tokens_maskgit_800samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit_800samples.json",
#         "tokenizer_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_800samples/tokenizer_DREAM_64steps_64tokens_maskgit_800samples.pt",
#         "name": "DREAM_64steps_64tokens_800samples",
#     },
#     "llada64_800samples": {
#         "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_800samples/outputs_LLaDa_64steps_64tokens_low_confidence_800samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence_800samples.json",
#         "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_800samples/tokenizer_LLaDa_64steps_64tokens_low_confidence_800samples.pt",
#         "name": "LLaDA_64steps_64tokens_800samples",
#     },
#     "dream64_400samples": {
#         "outputs_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_400samples/outputs_DREAM_64steps_64tokens_maskgit_400samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_DREAM_64steps_64tokens_maskgit_400samples.json",
#         "tokenizer_path": "PipelineTest/res/results_DREAM_64steps_64tokens_maskgit_400samples/tokenizer_DREAM_64steps_64tokens_maskgit_400samples.pt",
#         "name": "DREAM_64steps_64tokens_400samples",
#     },
#     "llada64_400samples": {
#         "outputs_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_400samples/outputs_LLaDa_64steps_64tokens_low_confidence_400samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_low_confidence_400samples.json",
#         "tokenizer_path": "PipelineTest/res/results_LLaDa_64steps_64tokens_low_confidence_400samples/tokenizer_LLaDa_64steps_64tokens_low_confidence_400samples.pt",
#         "name": "LLaDA_64steps_64tokens_400samples",
#     },
#     "llada128_400samples": {
#         "outputs_path": "PipelineTest/res/results_LLaDa_128steps_128tokens_low_confidence_400samples/outputs_LLaDa_128steps_128tokens_low_confidence_400samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_128steps_128tokens_low_confidence_400samples.json",
#         "tokenizer_path": "PipelineTest/res/results_LLaDa_128steps_128tokens_low_confidence_400samples/tokenizer_LLaDa_128steps_128tokens_low_confidence_400samples.pt",
#         "name": "LLaDA_128steps_128tokens_400samples",
#     },
#     "llada16_32tokens": {
#         "outputs_path": "PipelineTest/res/results_LLaDa_16steps_32tokens_low_confidence_samples/outputs_LLaDa_16steps_32tokens_low_confidence_samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_LLaDa_16steps_32tokens_low_confidence_samples.json",
#         "tokenizer_path": "PipelineTest/res/results_LLaDa_16steps_32tokens_low_confidence_samples/tokenizer_LLaDa_16steps_32tokens_low_confidence_samples.pt",
#         "name": "LLaDA_16steps_32tokens",
#     },
#     "dream16_32tokens_2100samples": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples/outputs_dream_16steps_32tokens_triviaqa_2100samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples/tokenizer_dream_16steps_32tokens_triviaqa_2100samples.pt",
#         "name": "dream_16steps_32tokens_2100samples",
#     },
#     "dream16_32tokens_2100samples_naturalquestion": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2100samples/outputs_dream_16steps_32tokens_naturalquestion_2100samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_naturalquestion_dream_16steps_32tokens_naturalquestion_2100samples.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2100samples/tokenizer_dream_16steps_32tokens_naturalquestion_2100samples.pt",
#         "name": "dream_16steps_32tokens_naturalquestion_2100samples",
#     },
#     "llada16_32tokens_2100samples_naturalquestion": {
#         "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2100samples/outputs_llada_16steps_32tokens_naturalquestion_2100samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_naturalquestion_llada_16steps_32tokens_naturalquestion_2100samples.json",
#         "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2100samples/tokenizer_llada_16steps_32tokens_naturalquestion_2100samples.pt",
#         "name": "llada_16steps_32tokens_naturalquestion_2100samples",
# },
#     "llada16_32tokens_2100samples_hotpotqa": {
#         "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2100samples/outputs_llada_16steps_32tokens_hotpotqa_2100samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_hotpotqa_llada_16steps_32tokens_hotpotqa_2100samples.json",
#         "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2100samples/tokenizer_llada_16steps_32tokens_hotpotqa_2100samples.pt",
#         "name": "llada_16steps_32tokens_hotpotqa_2100samples",
# },
#     "dream16_32tokens_2100samples_hotpotqa": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2100samples/outputs_dream_16steps_32tokens_hotpotqa_2100samples.pt",
#         "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_16steps_32tokens_hotpotqa_2100samples.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2100samples/tokenizer_dream_16steps_32tokens_hotpotqa_2100samples.pt",
#         "name": "dream_16steps_32tokens_hotpotqa_2100samples",
# },
#     "llada16_32tokens_2100samples_triviaqa_without_collapse": {
#         "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2100samples_temp0/outputs_llada_16steps_32tokens_triviaqa_2100samples_temp0.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2100samples_temp0.json",
#         "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2100samples_temp0/tokenizer_llada_16steps_32tokens_triviaqa_2100samples_temp0.pt",
#         "name": "llada16_32tokens_2100samples_triviaqa_without_collapse",
# },      
#    "llada16_32tokens_2100samples_triviaqa_without_collapse_exact_matches": {
#         "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp100-0_evalexact_match/outputs_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp100-0_evalexact_match.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp100-0_evalexact_match.json",
#         "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp100-0_evalexact_match/tokenizer_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp100-0_evalexact_match.pt",
#         "name": "llada16_32tokens_2100samples_triviaqa_without_collapse_exact_matches",
# },      
#     "dream16_32tokens_2100samples_triviaqa_without_collapse": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60/outputs_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60/tokenizer_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.pt",
#         "name": "dream16_32tokens_2100samples_triviaqa_without_collapse",
# },
#  "dream16_32tokens_2100samples_triviaqa_without_collapse_exact_matches": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_evalexact_match/outputs_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_evalexact_match.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_evalexact_match.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_evalexact_match/tokenizer_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_evalexact_match.pt",
#         "name": "dream16_32tokens_2100samples_triviaqa_without_collapse_exact_matches",
# },      
# "dream16_32tokens_2100samples_hotpotqa_without_collapse": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2100samples_mixedtemp40-60/outputs_dream_16steps_32tokens_hotpotqa_2100samples_mixedtemp40-60.pt",
#         "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_16steps_32tokens_hotpotqa_2100samples_mixedtemp40-60.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2100samples_mixedtemp40-60/tokenizer_dream_16steps_32tokens_hotpotqa_2100samples_mixedtemp40-60.pt",
#         "name": "dream16_32tokens_2100samples_hotpotqa_without_collapse",
# },
# "llada16_32tokens_2500samples_triviaqa_without_collapse_mixedtemp100-0_evalexact_match_seed42": {
#         "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalexact_match_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalexact_match_seed42.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalexact_match_seed42.json",
#         "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalexact_match_seed42/tokenizer_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalexact_match_seed42.pt",
#         "name": "llada16_32tokens_2500samples_triviaqa_without_collapse_mixedtemp100-0_evalexact_match_seed42",
# },
# "dream16_32tokens_2500samples_triviaqa_without_collapse_mixedtemp40-60_evalexact_match_seed42": {
#         "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp40-60_evalexact_match_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp40-60_evalexact_match_seed42.pt",
#         "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp40-60_evalexact_match_seed42.json",
#         "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp40-60_evalexact_match_seed42/tokenizer_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp40-60_evalexact_match_seed42.pt",
#         "name": "dream16_32tokens_2500samples_triviaqa_without_collapse_mixedtemp40-60_evalexact_match_seed42",
# },





# }
CONFIGS = {
    # =========================================================================
    # CONFIGURATIONS : 16 STEPS / 32 TOKENS (Existantes)
    # =========================================================================
    "dream16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "dream_16steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "llada16_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_naturalquestion_llada_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "llada_16steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "diffgemma16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_diffgemma_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_diffgemma_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_triviaqa_diffgemma_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_diffgemma_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_diffgemma_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "diffgemma_16steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
        },
    "llada16_32tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_16steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "llada_16steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "dream16_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_naturalquestion_dream_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_16steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "dream_16steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada16_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_hotpotqa_llada_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "llada_16steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
    "dream16_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_16steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "dream_16steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
    },

    # =========================================================================
    # CONFIGURATIONS : 32 STEPS / 64 TOKENS (Nouvelles)
    # =========================================================================
    "dream32_64tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "dream_32steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "llada32_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_naturalquestion_llada_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "llada_32steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada32_64tokens_2500samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_32steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "llada_32steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
    },
    "dream32_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_naturalquestion_dream_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_32steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "dream_32steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
    },
    "llada32_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_llada_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_hotpotqa_llada_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_llada_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "llada_32steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
    "dream32_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_32steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
        "name": "dream_32steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
    },
     # =========================================================================
        # CONFIGURATIONS : 32 STEPS / 32 TOKENS 
        # =========================================================================
        "dream32_32tokens_2500samples_triviaqa_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_dream_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_dream_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "dream_32steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
        },
        "llada32_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_llada_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_naturalquestion_llada_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_llada_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "llada_32steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
        },
        "llada32_32tokens_2500samples_triviaqa_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_llada_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_llada_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_32steps_32tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "llada_32steps_32tokens_triviaqa_2500samples_evalqwen_seed42",
        },
        "dream32_32tokens_2500samples_naturalquestion_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_dream_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_naturalquestion_dream_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_dream_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_32steps_32tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "dream_32steps_32tokens_naturalquestion_2500samples_evalqwen_seed42",
        },
        "llada32_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_llada_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_hotpotqa_llada_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_llada_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "llada_32steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
        },
        "dream32_32tokens_2500samples_hotpotqa_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_dream_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_dream_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_32steps_32tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
            "name": "dream_32steps_32tokens_hotpotqa_2500samples_evalqwen_seed42",
        },

    # =========================================================================
    # CONFIGURATIONS : 64 STEPS / 64 TOKENS 
    # =========================================================================
            "dream64_64tokens_2500samples_triviaqa_evalqwen_seed42": {
                "outputs_path": "PipelineTest/res/results_dream_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "eval_json": "PipelineTest/res/eval/results_triviaqa_dream_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
                "tokenizer_path": "PipelineTest/res/results_dream_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "name": "dream_64steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
            },
            "llada64_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
                "outputs_path": "PipelineTest/res/results_llada_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "eval_json": "PipelineTest/res/eval/results_naturalquestion_llada_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
                "tokenizer_path": "PipelineTest/res/results_llada_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "name": "llada_64steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
            },
            "llada64_64tokens_2500samples_triviaqa_evalqwen_seed42": {
                "outputs_path": "PipelineTest/res/results_llada_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "eval_json": "PipelineTest/res/eval/results_triviaqa_llada_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
                "tokenizer_path": "PipelineTest/res/results_llada_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_64steps_64tokens_triviaqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "name": "llada_64steps_64tokens_triviaqa_2500samples_evalqwen_seed42",
            },
            "dream64_64tokens_2500samples_naturalquestion_evalqwen_seed42": {
                "outputs_path": "PipelineTest/res/results_dream_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "eval_json": "PipelineTest/res/eval/results_naturalquestion_dream_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.json",
                "tokenizer_path": "PipelineTest/res/results_dream_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_64steps_64tokens_naturalquestion_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "name": "dream_64steps_64tokens_naturalquestion_2500samples_evalqwen_seed42",
            },
            "llada64_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
                "outputs_path": "PipelineTest/res/results_llada_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_llada_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "eval_json": "PipelineTest/res/eval/results_hotpotqa_llada_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
                "tokenizer_path": "PipelineTest/res/results_llada_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_llada_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "name": "llada_64steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
            },
            "dream64_64tokens_2500samples_hotpotqa_evalqwen_seed42": {
                "outputs_path": "PipelineTest/res/results_dream_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/outputs_dream_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "eval_json": "PipelineTest/res/eval/results_hotpotqa_dream_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.json",
                "tokenizer_path": "PipelineTest/res/results_dream_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42/tokenizer_dream_64steps_64tokens_hotpotqa_2500samples_mixedtemp100-0_evalqwen_seed42.pt",
                "name": "dream_64steps_64tokens_hotpotqa_2500samples_evalqwen_seed42",
            },
}


# =========================================================================
# LOGISTIC REGRESSION PIPELINE
# =========================================================================

def run_logreg(X, y, feature_names, name):
    """Run GridSearchCV logistic regression, return results dict."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42)),
    ])
    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logreg__penalty": ["l1", "l2"],
        "logreg__solver": ["saga"],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
        )
    print("TRAINING SAMPLES:", len(X_train), "TESTING SAMPLES:", len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_scores = best.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)
    pr = average_precision_score(y_test, y_scores)

    thresholds = np.linspace(0, 1, 201)
    accs = [accuracy_score(y_test, (y_scores >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    coefs = best.named_steps["logreg"].coef_[0]

    result = {
        "name": name,
        "features": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "best_cv_roc_auc": float(grid.best_score_),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(best_acc),
        "test_best_threshold": float(best_thresh),
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "test_pos_rate": float(np.mean(y_test)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "coefficients": {fname: float(c) for fname, c in zip(feature_names, coefs)},
    }

    print(f"  [{name}] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  Acc={acc:.4f}  BestAcc={best_acc:.4f} (thresh={best_thresh:.2f})")
    print(f"           Best params: {grid.best_params_}")
    return result


# =========================================================================
# MAIN
# =========================================================================

def main(config_name="llada"):
    cfg = CONFIGS[config_name]
    output_path = cfg["outputs_path"]
    eval_json = cfg["eval_json"]
    print(f"\n{'='*70}")
    print(f"  EVALUATION: {cfg['name']}")
    print(f"{'='*70}")

    # Load data
    print("\n[1] Loading outputs...")
    outputs = load_outputs(output_path)
    positions, labels, data = match_samples(eval_json, outputs)
    print(f"    Matched samples: {len(positions)} (correct={np.sum(labels==0)}, halluc={np.sum(labels==1)})")

    # Pad token
    try:
        pad_token_id = get_pad_token_id(output_path)
        print(f"    Pad token id: {pad_token_id}")
    except Exception:
        pad_token_id = None
        print("    No tokenizer found, skipping no-padding features.")

    # --- BASELINE FEATURES ---
    print("\n[2] Computing baseline features...")
    feat_baseline, names_baseline = get_baseline_features(outputs, pad_token_id=pad_token_id)
    feat_baseline = feat_baseline[positions]  # align with matched samples

    # --- MARKOVIAN FEATURES ---
    print("\n[3] Computing Markovian dynamic features...")
    feat_markov, names_markov = get_markovian_features(outputs, k_tokens=64)
    feat_markov = feat_markov[positions]

    # --- AR(1) FEATURES ---
    print("\n[4] Fitting AR(1) models and computing features...")
    # Split 50/50 for AR1 fitting vs evaluation
    ar1_idx, eval_idx = train_test_split(
        np.arange(len(positions)), test_size=0.5, stratify=labels, random_state=42
    )

    ar1_positions = positions[ar1_idx]
    ar1_labels = labels[ar1_idx]
    eval_positions = positions[eval_idx]
    eval_labels = labels[eval_idx]

    # Padding for AR1
    if pad_token_id is not None:
        padding_all = compute_padding_mask(outputs, positions, pad_token_id)
        padding_ar1 = padding_all[ar1_idx, 0, :]  # (N_ar1, D)
    else:
        padding_ar1 = None

    ar1_models = fit_ar1_models(outputs, ar1_positions, ar1_labels, padding_2d=None)

    # Compute AR1 features on eval half
    entropies = getEntropy(outputs)
    masks = getEachStepMask(outputs)
    entropy_eval = np.array([entropies[i] for i in eval_positions])
    mask_eval = np.array([masks[i] for i in eval_positions], dtype=float)

    if pad_token_id is not None:
        padding_eval = padding_all[eval_idx]
        effective_mask_eval = mask_eval * (1 - padding_eval.astype(float))
    else:
        effective_mask_eval = mask_eval

    feat_ar1, names_ar1 = get_ar1_features(outputs, eval_positions, effective_mask_eval, ar1_models)

    # --- EVALUATION ---
    print("\n[5] Running classifiers...")

    # Baseline only (on eval half)
    X_baseline_eval = feat_baseline[eval_idx]
    print("\n  === Baseline Features ===")
    res_baseline = run_logreg(feat_baseline, labels, names_baseline, "Baseline")

    # Markovian only (on eval half)
    X_markov_eval = feat_markov[eval_idx]
    print("\n  === Markovian Features ===")
    res_markov = run_logreg(feat_markov, labels, names_markov, "Markovian")

    # AR1 only
    print("\n  === AR(1) Features ===")
    res_ar1 = run_logreg(feat_ar1, eval_labels, names_ar1, "AR1")

    # Baseline + Markovian
    X_base_markov = np.column_stack([feat_baseline, feat_markov])
    names_bm = names_baseline + names_markov
    print("\n  === Baseline + Markovian ===")
    res_bm = run_logreg(X_base_markov, labels, names_bm, "Baseline+Markov")

    # All combined
    X_all = np.column_stack([X_baseline_eval, X_markov_eval, feat_ar1])
    names_all = names_baseline + names_markov + names_ar1
    print("\n  === All Combined ===")
    res_all = run_logreg(X_all, eval_labels, names_all, "All")

    # --- SAVE ---
    save_dir = os.path.join(os.path.dirname(output_path), "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "config": cfg,
        "n_samples": len(positions),
        "n_correct": int(np.sum(labels == 0)),
        "n_halluc": int(np.sum(labels == 1)),
        "results": {
            "Baseline": res_baseline,
            "Markovian": res_markov,
            "AR1": res_ar1,
            "Baseline+Markov": res_bm,
            "All": res_all,
        },
    }

    out_path = os.path.join(save_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n    Results saved to: {out_path}")

    # --- SUMMARY TABLE ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {cfg['name']}")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'ROC-AUC':>10} {'PR-AUC':>10} {'Accuracy':>10}")
    print(f"  {'-'*52}")
    for r in [res_baseline, res_markov, res_ar1, res_bm, res_all]:
        print(f"  {r['name']:<22} {r['test_roc_auc']:>10.4f} {r['test_pr_auc']:>10.4f} {r['test_accuracy']:>10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hallucination detection evaluation")
    parser.add_argument("--config", type=str, default="llada", choices=list(CONFIGS.keys()),
                        help="Which model config to evaluate")
    args = parser.parse_args()
    main(args.config)
