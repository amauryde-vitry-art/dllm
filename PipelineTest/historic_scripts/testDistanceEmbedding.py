import numpy as np
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import hdbscan
import umap
import torch
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
import faiss


def get_vocab_embeddings(model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"):
    """
    Récupère la matrice d'embeddings du vocabulaire du modèle.

    Args:
        model_name_or_path: chemin ou nom HuggingFace du modèle

    Returns:
        embeddings: np.ndarray shape (vocab_size, hidden_dim)
        tokenizer: le tokenizer associé
    """


    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Récupérer la couche d'embedding (input embeddings)
    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer.weight.detach().cpu().numpy()

    return embeddings

embeddings = get_vocab_embeddings()


# ============================================================
# Top-K Cluster Entropy : entropie sémantique locale et dynamique
# ============================================================
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

# Normaliser les embeddings pour cosine similarity
embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def topk_cluster_entropy(logits, embedding_matrix_normed, topk=20, cos_threshold=0.7):
    """
    Calcule une entropie sémantique locale à partir des top-K tokens.
    
    Args:
        logits: np.ndarray shape (V,) — logits bruts pour UNE position
        embedding_matrix_normed: np.ndarray shape (V, D) — embeddings L2-normalisés
        topk: nombre de tokens candidats
        cos_threshold: seuil de cosine similarity pour merger en cluster
        
    Returns:
        cluster_entropy: float — Shannon entropy sur les probas agrégées par cluster
        n_clusters: int — nombre de clusters trouvés
        cluster_info: list[dict] — détail de chaque cluster (tokens, probs)
    """
    # Top-K par logits
    topk_indices = np.argpartition(logits, -topk)[-topk:]
    topk_logits = logits[topk_indices]
    
    # Softmax sur le top-K seulement (renormalisé)
    topk_logits -= topk_logits.max()
    topk_probs = np.exp(topk_logits) / np.exp(topk_logits).sum()
    
    # Embeddings des top-K tokens
    topk_embs = embedding_matrix_normed[topk_indices]  # (K, D)
    
    # Cosine distance matrix (1 - cosine_sim)
    cos_dists = 1.0 - topk_embs @ topk_embs.T
    np.fill_diagonal(cos_dists, 0.0)
    cos_dists = np.clip(cos_dists, 0, 2)  # numerical stability
    
    # Agglomerative clustering avec seuil de distance
    if topk <= 1:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - cos_threshold,  # cos_dist = 1 - cos_sim
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(cos_dists)
    
    # Agréger les probas par cluster
    unique_labels = np.unique(labels)
    cluster_probs = []
    cluster_info = []
    for label in unique_labels:
        mask = labels == label
        p = topk_probs[mask].sum()
        cluster_probs.append(p)
        cluster_info.append({
            "tokens": [tokenizer.decode(topk_indices[i]) for i in np.where(mask)[0]],
            "token_ids": topk_indices[mask].tolist(),
            "total_prob": float(p),
        })
    
    cluster_probs = np.array(cluster_probs)
    
    # Shannon entropy sur les clusters
    cluster_probs = cluster_probs[cluster_probs > 0]
    entropy = -np.sum(cluster_probs * np.log(cluster_probs))
    
    return entropy, len(unique_labels), cluster_info



# Distribution des cosine sims entre tokens aléatoires
sample_idx = np.random.choice(len(embeddings_normed), 1000, replace=False)
sample_embs = embeddings_normed[sample_idx]
sims = sample_embs @ sample_embs.T
upper_tri = sims[np.triu_indices(1000, k=1)]
print(f"cosine sim stats: mean={upper_tri.mean():.4f}, std={upper_tri.std():.4f}, "
      f"median={np.median(upper_tri):.4f}, p95={np.percentile(upper_tri, 95):.4f}")

# # ============================================================
# # DEMO : simuler des logits et voir la différence
# # ============================================================
# V = embeddings.shape[0]

# print("\n" + "="*60)
# print("DEMO: Top-K Cluster Entropy")
# print("="*60)

# # Cas 1 : le modèle hésite entre tokens de whitespace
# space_id = tokenizer.encode(" ", add_special_tokens=False)[0]   # 220
# newline_id = tokenizer.encode("\n", add_special_tokens=False)[0] # 198
# tab_id = tokenizer.encode("\t", add_special_tokens=False)[0]

# fake_logits_whitespace = np.full(V, -10.0)
# fake_logits_whitespace[space_id] = 2.0
# fake_logits_whitespace[newline_id] = 1.8
# fake_logits_whitespace[tab_id] = 1.5

# ent, n_cl, info = topk_cluster_entropy(fake_logits_whitespace, embeddings_normed, topk=20, cos_threshold=0.7)
# print(f"\nCas 1 - Hésitation whitespace:")
# print(f"  Entropy classique (top-K): {-np.sum(np.array([0.4, 0.35, 0.25]) * np.log(np.array([0.4, 0.35, 0.25]))):.4f}")
# print(f"  Cluster entropy:           {ent:.4f}  ({n_cl} clusters)")
# for c in info[:5]:
#     print(f"    cluster: {c['tokens'][:5]} → p={c['total_prob']:.3f}")

# # Cas 2 : le modèle hésite entre mots sémantiquement éloignés
# dog_id = tokenizer.encode("dog", add_special_tokens=False)[0]
# eq_id = tokenizer.encode("equation", add_special_tokens=False)[0]

# fake_logits_divergent = np.full(V, -10.0)
# fake_logits_divergent[dog_id] = 2.0
# fake_logits_divergent[eq_id] = 1.8

# ent2, n_cl2, info2 = topk_cluster_entropy(fake_logits_divergent, embeddings_normed, topk=20, cos_threshold=0.7)
# print(f"\nCas 2 - Hésitation dog/equation:")
# print(f"  Cluster entropy:           {ent2:.4f}  ({n_cl2} clusters)")
# for c in info2[:5]:
#     print(f"    cluster: {c['tokens'][:5]} → p={c['total_prob']:.3f}")

# # Cas 3 : le modèle est très sûr
# fake_logits_certain = np.full(V, -10.0)
# fake_logits_certain[dog_id] = 10.0

# ent3, n_cl3, info3 = topk_cluster_entropy(fake_logits_certain, embeddings_normed, topk=20, cos_threshold=0.7)
# print(f"\nCas 3 - Modèle sûr (dog):")
# print(f"  Cluster entropy:           {ent3:.4f}  ({n_cl3} clusters)")
# for c in info3[:3]:
#     print(f"    cluster: {c['tokens'][:5]} → p={c['total_prob']:.3f}")