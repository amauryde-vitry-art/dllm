import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import hdbscan
import umap
import torch
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


def reduce_embeddings_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", save_path=None):
    """
    Réduit la dimension des embeddings avec UMAP.

    Args:
        embeddings: np.ndarray shape (vocab_size, hidden_dim)
        n_components: dimension cible (2 ou 3 pour visualisation)
        n_neighbors: nombre de voisins (contrôle local vs global)
        min_dist: distance minimale entre points dans l'espace réduit
        metric: métrique de distance ("cosine", "euclidean", etc.)
        save_path: chemin pour sauvegarder le résultat (.npy). None = pas de sauvegarde.

    Returns:
        embeddings_reduced: np.ndarray shape (vocab_size, n_components)
    """

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    embeddings_reduced = reducer.fit_transform(embeddings)

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.save(save_path, embeddings_reduced)
        print(f"Embeddings réduits sauvegardés: {save_path}")

    return embeddings_reduced


def cluster_hdbscan(embeddings, min_cluster_size=15, min_samples=5, metric="euclidean"):
    """
    Clusterise des embeddings avec HDBSCAN.

    Args:
        embeddings: np.ndarray shape (n, d) — embeddings (éventuellement réduits par UMAP)
        min_cluster_size: taille minimale d'un cluster
        min_samples: nombre min de voisins pour qu'un point soit core
        metric: métrique de distance

    Returns:
        labels: np.ndarray (n,) — label de cluster par point (-1 = bruit)
        clusterer: l'objet HDBSCAN fitté
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} points bruit ({n_noise/len(labels)*100:.1f}%)")
    return labels, clusterer


def build_token_to_cluster(reduced_embedding_path, model_name_or_path="GSAI-ML/LLaDA-8B-Instruct", min_cluster_size=15, min_samples=5, metric="euclidean"):
    """
    Charge les embeddings UMAP sauvegardés, clusterise avec HDBSCAN,
    et retourne un dictionnaire token_id (int) -> cluster_label (int).
    Les token_id correspondent exactement aux indices du tokenizer.

    Args:
        reduced_embedding_path: chemin vers le fichier .npy des embeddings réduits
        model_name_or_path: modèle dont le tokenizer définit les token_id
        min_cluster_size: paramètre HDBSCAN
        min_samples: paramètre HDBSCAN
        metric: métrique de distance pour HDBSCAN

    Returns:
        token_to_cluster: dict {int: int} — token_id -> label cluster (-1 = bruit)
    """
    embeddings_reduced = np.load(reduced_embedding_path)

    
    cluster_labels, _ = cluster_hdbscan(
        embeddings_reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )

    # Les indices de la matrice d'embedding correspondent aux token_id du tokenizer
    token_to_cluster = {
        int(token_id): int(cluster_labels[token_id])
        for token_id in range(len(cluster_labels))
    }
    # save json
    save_path = "PipelineTest/embeddings/token_to_cluster.json"
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(token_to_cluster, f, indent=2)
    print(f"Token to cluster mapping saved: {save_path}")



# embeddings, tokenizer = get_vocab_embeddings()
# embeddings_reduced = reduce_embeddings_umap(embeddings, n_components=50, n_neighbors=15, min_dist=0.1, metric="cosine", save_path="PipelineTest/embeddings/embeddings_umap_dim_50.npy")
# build_token_to_cluster("PipelineTest/embeddings/embeddings_umap_dim_50.npy", min_cluster_size=15, min_samples=1, metric="euclidean")

def reduce_embedding_dim_pca(embeddings, n_components=50, save_path=None):
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(normalized_embeddings)

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.save(save_path, embeddings_reduced)
        print(f"Embeddings réduits par PCA sauvegardés: {save_path}")

    return embeddings_reduced





def build_token_to_cluster_kmeans(reduced_embedding_path, n_clusters=1024, save_path="PipelineTest/embeddings/token_to_cluster_kmeans.json"):
    """
    Clusterise les embeddings réduits avec KMeans (aucun bruit, tous les tokens sont assignés).
    """
    from sklearn.cluster import MiniBatchKMeans

    embeddings_reduced = np.load(reduced_embedding_path)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=10)
    labels = kmeans.fit_predict(embeddings_reduced)

    print(f"KMeans: {n_clusters} clusters, 0 points bruit")

    token_to_cluster = {int(i): int(labels[i]) for i in range(len(labels))}

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(token_to_cluster, f, indent=2)
    print(f"Token to cluster mapping saved: {save_path}")

    return token_to_cluster


def find_optimal_k(reduced_embedding_path, k_range=range(256, 4097, 256), save_path="PipelineTest/results_64_step/metricsResults/Plots/silhouette_scores.png"):
    """
    Teste plusieurs valeurs de k et trace le silhouette score pour choisir le meilleur.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import MiniBatchKMeans
    import matplotlib.pyplot as plt

    embeddings_reduced = np.load(reduced_embedding_path)
    scores = []

    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3)
        labels = kmeans.fit_predict(embeddings_reduced)
        score = silhouette_score(embeddings_reduced, labels, sample_size=10000, random_state=42)
        scores.append(score)
        print(f"k={k}: silhouette={score:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.grid(True)
    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.close()

    best_k = list(k_range)[np.argmax(scores)]
    print(f"Best k: {best_k} (silhouette={max(scores):.4f})")
    return best_k

def visualize_embeddings_tsne(embeddings, labels, perplexity=30,  save_path=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    tsne = TSNE(n_components=2, perplexity=perplexity,random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    print('embeddings_2d shape:', embeddings_2d.shape)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette='tab10', legend='full', s=50)
    plt.title("t-SNE Visualization of Token Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path)
        print(f"t-SNE plot saved: {save_path}")

    plt.show()

def build_token_to_cluster_rq(embeddings_path, M=24, nbits=8, save_path="PipelineTest/embeddings/token_to_codes_rq.npy"):
    """
    Residual Quantization avec FAISS.
    Chaque token est encodé en M indices de codebook (chacun dans [0, 2^nbits)).

    Args:
        embeddings_path: chemin vers les embeddings bruts (.npy), shape (vocab_size, d)
        M: nombre de codebooks (étapes de quantification résiduelle)
        nbits: bits par codebook (256 centroids = 8 bits)
        save_path: où sauvegarder les codes

    Returns:
        codes: np.ndarray shape (vocab_size, M) — chaque ligne = tuple de M indices de codebook
    """
    embeddings = np.load(embeddings_path).astype(np.float32)

    # Normaliser L2
    faiss.normalize_L2(embeddings)
    # diminuer la dimension pour accélérer le RQ 
    pca = PCA(n_components=100, random_state=42)
    embeddings = pca.fit_transform(embeddings)
    
    n, d = embeddings.shape

    # Créer le ResidualQuantizer
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = True

    # Entrainer sur les embeddings
    print(f"Training RQ: d={d}, M={M}, nbits={nbits} ({2**nbits} centroids/codebook)")
    rq.train(embeddings)

    # Encoder tous les tokens
    codes = rq.compute_codes(embeddings)  # shape (n, code_size_bytes)

    # Décoder les codes compacts en indices par codebook
    # codes est un array de bytes, on doit extraire les indices
    if nbits == 8:
        # 1 byte = 1 indice de codebook
        codes_per_token = np.frombuffer(codes, dtype=np.uint8).reshape(n, M)
    elif nbits == 4:
        # 1 byte = 2 indices de codebook (4 bits chacun)
        raw = np.frombuffer(codes, dtype=np.uint8).reshape(n, M // 2)
        codes_per_token = np.zeros((n, M), dtype=np.int32)
        for j in range(M // 2):
            codes_per_token[:, 2 * j] = raw[:, j] & 0x0F
            codes_per_token[:, 2 * j + 1] = (raw[:, j] >> 4) & 0x0F
    else:
        raise ValueError(f"nbits={nbits} not supported (use 4 or 8)")

    print(f"RQ codes shape: {codes_per_token.shape}")
    print(f"Codebook utilization: {[len(np.unique(codes_per_token[:, j])) for j in range(min(M, 5))]} (first 5 codebooks)")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, codes_per_token)
    print(f"RQ codes saved: {save_path}")

    return codes_per_token


def get_semantic_cluster_from_rq(codes, level=1):
    """
    Retourne le cluster assignment pour un niveau donné du RQ.
    level=0 → cluster le plus grossier (premier codebook, 256 groupes)
    level plus élevé → clusters plus fins

    Pour la semantic entropy, utiliser level=0 donne 256 clusters sémantiques.
    """
    return codes[:, level]


def build_token_to_cluster_from_codes(codes_path, n_sub=4, save_path=None):
    """
    Construit un token_to_cluster à partir des codes RQ sauvegardés.

    Args:
        codes_path: chemin vers le .npy des codes RQ (shape: vocab_size x M)
        n_sub: subdivisions du 2e codebook (1=256, 4=1024, 16=4096, 256=65536 clusters)
        save_path: chemin JSON de sauvegarde (optionnel)

    Returns:
        token_to_cluster: dict {int: int}
    """
    codes = np.load(codes_path)
    token_to_cluster = {
        int(i): int(codes[i, 0]) * n_sub + int(codes[i, 1]) // (256 // n_sub)
        for i in range(len(codes))
    }
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(token_to_cluster, f)
    return token_to_cluster


def evaluate_clustering_granularities(
    codes_path,
    embeddings_path,
    n_sub_values=(1, 2, 4, 8, 16, 32, 64, 256),
    save_plot_path="PipelineTest/results_64_step/metricsResults/Plots/granularity_impact.png",
):
    """
    Teste différentes granularités de clustering RQ et mesure la qualité
    via le silhouette score (non supervisé).

    Args:
        codes_path: chemin vers les codes RQ (.npy), shape (vocab_size, M)
        embeddings_path: chemin vers les embeddings (.npy) pour calculer le silhouette
        n_sub_values: tuple de valeurs de n_sub à tester
        save_plot_path: chemin pour sauvegarder le plot

    Returns:
        results: dict {n_clusters: {"silhouette": float, "inertia_approx": float}}
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    codes = np.load(codes_path)
    embeddings = np.load(embeddings_path).astype(np.float32)

    # Sous-échantillonner pour le silhouette (trop lent sur 128k)
    sample_size = min(20000, len(embeddings))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(embeddings), size=sample_size, replace=False)
    embeddings_sample = embeddings[sample_idx]

    # Normaliser L2 pour le silhouette en cosine
    norms = np.linalg.norm(embeddings_sample, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_sample_normed = embeddings_sample / norms

    results = {}

    for n_sub in n_sub_values:
        n_clusters = 256 * n_sub
        print(f"\n{'='*50}")
        print(f"Testing n_sub={n_sub} → {n_clusters} clusters (~{len(codes) // n_clusters} tok/cluster)")

        # Construire les labels pour l'échantillon
        labels_sample = np.array([
            int(codes[i, 0]) * n_sub + int(codes[i, 1]) // (256 // n_sub)
            for i in sample_idx
        ])

        # Silhouette score (cosine)
        score = silhouette_score(embeddings_sample_normed, labels_sample, metric="cosine", sample_size=10000, random_state=42)

        results[n_clusters] = {
            "silhouette": float(score),
            "n_sub": int(n_sub),
            "tok_per_cluster": int(len(codes) // n_clusters),
        }
        print(f"  Silhouette (cosine): {score:.4f}")

    # Plot
    n_clusters_list = sorted(results.keys())
    silhouettes = [results[k]["silhouette"] for k in n_clusters_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_clusters_list, silhouettes, marker='o', linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette Score (cosine)")
    ax.set_title("Silhouette Score vs Clustering Granularity (RQ)")
    ax.grid(True)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_plot_path:
        dirname = os.path.dirname(save_plot_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {save_plot_path}")
    plt.close()

    # Save results JSON
    results_path = save_plot_path.replace('.png', '.json') if save_plot_path else "PipelineTest/results_64_step/metricsResults/granularity_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\nBEST: {best_k} clusters → silhouette = {results[best_k]['silhouette']:.4f}")

    return results


def main():

    mode = 'eval'
    if mode == "train":
        # Entrainer le RQ et sauvegarder les codes
        embeddings = get_vocab_embeddings()
        embed_path = "PipelineTest/embeddings/embeddings_raw.npy"
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        np.save(embed_path, embeddings)
        codes = build_token_to_cluster_rq(embed_path, M=24, nbits=4, save_path="PipelineTest/embeddings/token_to_codes_rq_pca_dim_100_4bits.npy")

    elif mode == "eval":
        # Evaluer l'impact de la granularité sur la classification
        results = evaluate_clustering_granularities(
            codes_path="PipelineTest/embeddings/token_to_codes_rq_pca_dim_100_4bits.npy",
            embeddings_path="PipelineTest/embeddings/embeddings_raw.npy",
            n_sub_values=(1, 2, 4, 8, 16, 32, 64),
            save_plot_path="PipelineTest/results_64_step/metricsResults/Plots/granularity_impact_rq_4bits.png",
        )
        embeddings = get_vocab_embeddings()
        
    else:
        print(f"Usage: python SemanticEntropy.py [train|eval]")

if __name__ == "__main__":
    main()
    token_id_to_cluster =build_token_to_cluster_from_codes("PipelineTest/embeddings/token_to_codes_rq_pca_dim_100_4bits.npy", n_sub=1, save_path="PipelineTest/embeddings/token_to_cluster_rq_128_pca_dim_100.json")
    embeddings = get_vocab_embeddings()
    labels = [token_id_to_cluster[i] for i in range(len(embeddings))]
    indices = np.random.choice(len(embeddings), size=10000, replace=False)
    embeddings_sample = embeddings[indices]
    labels_sample = [labels[i] for i in indices]
    visualize_embeddings_tsne(embeddings_sample, labels_sample, perplexity=30, save_path="PipelineTest/res/results_64_step/metricsResults/Plots/tsne_rq_128_pca_dim_100.png")

