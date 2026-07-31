"""
Centralized data loading and train/test splitting for all benchmark scripts.

Ensures every script uses the exact same samples and the exact same train/test partition.
"""

import json
import numpy as np


def load_eval_data(eval_json_path):
    """
    Load eval JSON and return aligned (labels, indices, idx_to_prompt).

    Filtering rules (applied uniformly to ALL scripts):
      - Skip items that have no ``is_hallucination`` field.
      - Skip items whose ``is_hallucination`` is neither "yes" nor "no".
      - Skip duplicate indices (keep first occurrence).

    Collapse samples are **never** dropped – they stay in the pool.
    """
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    indices = []
    idx_to_prompt = {}
    n_missing_label = 0
    seen_indices = set()

    for item in data:
        idx = item.get("index")

        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        if "is_hallucination" not in item:
            n_missing_label += 1
            continue

        raw_label = str(item["is_hallucination"]).lower()
        if raw_label not in ("yes", "no"):
            n_missing_label += 1
            continue

        labels.append(1 if raw_label == "yes" else 0)
        indices.append(idx)
        idx_to_prompt[int(idx)] = item.get("question", "")

    return (
        np.array(labels),
        np.array(indices),
        n_missing_label,
        idx_to_prompt,
    )


def get_train_test_split(n_samples, train_ratio=0.8):
    """
    Return (train_idx, test_idx) as sequential 80/20 split.

    The split is purely positional: first ``train_ratio`` of the array is
    training, the rest is test.  This guarantees every script that calls
    ``load_eval_data`` followed by ``get_train_test_split`` produces
    identical partitions.
    """
    all_idx = np.arange(n_samples)
    split_point = int(n_samples * train_ratio)
    return all_idx[:split_point], all_idx[split_point:]
