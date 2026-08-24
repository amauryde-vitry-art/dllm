"""
Semantic Token Clustering (STC) for LLaDA.

Based on: "Semantic Token Clustering for Efficient Uncertainty Quantification
in Large Language Models" (EACL 2026).

Steps:
  1. Extract input embeddings (token embedding layer) and output embeddings
     (language modeling head) from LLaDA.
  2. Concatenate them to form unified semantic representations per token.
  3. Filter out stopwords (NLTK) and Arabic numerals.
  4. Cluster with Agglomerative Clustering (cosine distance, n_clusters=16000).
  5. Save token_id -> cluster_id mapping as JSON.
"""

import json
import os
import re
import time

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"
N_CLUSTERS = 16_000
SAVE_DIR = "PipelineTest/embeddings"
SAVE_PATH = os.path.join(SAVE_DIR, "DREAM_token_to_cluster_stc_16k.json")


def get_stopwords() -> set[str]:
    """Return NLTK English stopwords. Downloads if needed."""
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


def is_arabic_numeral(token_str: str) -> bool:
    """Check if a decoded token string represents an Arabic numeral."""
    cleaned = token_str.strip().replace("Ġ", "").replace("▁", "").replace(" ", "")
    if not cleaned:
        return False
    return bool(re.fullmatch(r"\d+", cleaned))

 
def is_stopword(token_str: str, stopwords_set: set[str]) -> bool:
    """Check if a decoded token matches an English stopword."""
    cleaned = token_str.strip().replace("Ġ", "").replace("▁", "").lower().strip()
    return cleaned in stopwords_set


def extract_embeddings(model_name: str = MODEL_NAME):
    """
    Extract input embeddings and output embeddings (lm_head weights) from LLaDA.

    Returns:
        input_emb: np.ndarray [V, D_in]
        output_emb: np.ndarray [V, D_out]
        tokenizer: the tokenizer
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.eval()

    # Input embeddings: token embedding layer
    input_emb = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    print(f"  Input embeddings shape: {input_emb.shape}")

    # Output embeddings: lm_head (language modeling head)
    if hasattr(model, "lm_head"):
        output_emb = model.lm_head.weight.detach().cpu().float().numpy()
    elif hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        output_emb = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    else:
        # If tied embeddings, input == output
        print("  Warning: no separate lm_head found, using input embeddings as output embeddings")
        output_emb = input_emb.copy()
    print(f"  Output embeddings shape: {output_emb.shape}")

    del model
    torch.cuda.empty_cache()

    return input_emb, output_emb, tokenizer


def build_stc_clusters(
    model_name: str = MODEL_NAME,
    n_clusters: int = N_CLUSTERS,
    save_path: str = SAVE_PATH,
):
    """
    Full STC pipeline: extract embeddings, filter, cluster, save.

    Args:
        model_name: HuggingFace model name or path.
        n_clusters: Number of agglomerative clusters.
        save_path: Where to save the JSON mapping.
    """
    # 1. Extract embeddings
    input_emb, output_emb, tokenizer = extract_embeddings(model_name)
    vocab_size = input_emb.shape[0]

    # 2. Concatenate input + output embeddings → unified representation
    unified_emb = np.concatenate([input_emb, output_emb], axis=1)  # [V, D_in + D_out]
    print(f"Unified embeddings shape: {unified_emb.shape}")

    del input_emb, output_emb

    # 3. Filter: exclude stopwords and Arabic numerals
    stopwords_set = get_stopwords()
    keep_mask = np.ones(vocab_size, dtype=bool)
    excluded_stopword = 0
    excluded_numeral = 0

    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
        except Exception:
            keep_mask[token_id] = False
            continue
        if is_stopword(token_str, stopwords_set):
            keep_mask[token_id] = False
            excluded_stopword += 1
        elif is_arabic_numeral(token_str):
            keep_mask[token_id] = False
            excluded_numeral += 1

    kept_ids = np.where(keep_mask)[0]
    excluded_ids = np.where(~keep_mask)[0]
    print(f"Vocab size: {vocab_size}")
    print(f"  Excluded stopwords: {excluded_stopword}")
    print(f"  Excluded numerals:  {excluded_numeral}")
    print(f"  Tokens to cluster:  {len(kept_ids)}")

    # 4. L2-normalize for cosine distance in AgglomerativeClustering
    emb_to_cluster = normalize(unified_emb[kept_ids], norm="l2")

    # 5. Agglomerative Clustering with cosine distance
    n_clusters_actual = min(n_clusters, len(kept_ids))
    print(f"Running AgglomerativeClustering (n_clusters={n_clusters_actual}, metric=cosine)...")
    t0 = time.time()

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters_actual,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(emb_to_cluster)
    print(f"  Clustering done in {time.time() - t0:.1f}s")

    # 7. Silhouette score (sampled for speed — full dataset is too large)
    print("Computing silhouette score (on a 50k sample)...")
    sample_size = min(50_000, len(emb_to_cluster))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(emb_to_cluster), size=sample_size, replace=False)
    sil = silhouette_score(
        emb_to_cluster[sample_idx], labels[sample_idx], metric="cosine", sample_size=None
    )
    print(f"  Silhouette score (cosine, {sample_size} samples): {sil:.4f}")

    # 8. Build token_id -> cluster_id mapping
    #    - Clustered tokens get their assigned cluster label
    #    - Excluded tokens (stopwords/numerals) each get their own unique cluster
    token_to_cluster = {}
    next_cluster_id = int(labels.max()) + 1

    for i, token_id in enumerate(kept_ids):
        token_to_cluster[int(token_id)] = int(labels[i])

    for token_id in excluded_ids:
        token_to_cluster[int(token_id)] = next_cluster_id
        next_cluster_id += 1

    total_clusters = next_cluster_id
    print(f"Total clusters (incl. singletons for excluded tokens): {total_clusters}")

    # 9. Save token_to_cluster mapping
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(token_to_cluster, f)
    print(f"Saved token_to_cluster mapping to {save_path}")

    # 10. Save readable cluster contents JSON
    cluster_contents = {}
    for token_id_str, cluster_id in token_to_cluster.items():
        cid = str(cluster_id)
        if cid not in cluster_contents:
            cluster_contents[cid] = []
        try:
            token_str = tokenizer.decode([int(token_id_str)])
        except Exception:
            token_str = f"<token_{token_id_str}>"
        cluster_contents[cid].append({"id": int(token_id_str), "token": token_str})

    # Sort clusters by size (largest first) for readability
    cluster_contents = dict(
        sorted(cluster_contents.items(), key=lambda x: len(x[1]), reverse=True)
    )

    clusters_path = save_path.replace(".json", "_contents.json")
    with open(clusters_path, "w") as f:
        json.dump(
            {"silhouette_score": sil, "n_clusters": total_clusters, "clusters": cluster_contents},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved cluster contents to {clusters_path}")

    return token_to_cluster


if __name__ == "__main__":
    token_to_cluster = build_stc_clusters()

    # Print the largest cluster
    from collections import Counter
    cluster_counts = Counter(token_to_cluster.values())
    largest_cluster_id = cluster_counts.most_common(1)[0][0]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokens_in_cluster = [
        tokenizer.decode([tid]) for tid, cid in token_to_cluster.items() if cid == largest_cluster_id
    ]
    print(f"\nLargest cluster (id={largest_cluster_id}, size={len(tokens_in_cluster)}):")
    print(tokens_in_cluster)