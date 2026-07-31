import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
import transformers
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import dllm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from PipelineTest.scripts.run_evaluation import CONFIGS
from PipelineTest.Benchmark.data_split import load_eval_data, get_train_test_split

# =========================================================================
# CONFIGURATIONS DES SAMPLERS
# =========================================================================
@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    block_size: int = 32
    temperature: float = 0.5
    remasking: str = "low_confidence"

@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    alg: str = "maskgit_plus"
    alg_temp: float = 0.0
    top_p: float = 1.0
    temperature: float = 0.5

@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = False


# =========================================================================
# DISTRIBUTED HELPERS
# =========================================================================
def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# =========================================================================
# GENERATION ET CALCUL NLI DISTRIBUÉ (OPTIMISÉ MULTI-GPU)
# =========================================================================
def generate_and_evaluate_nli_parallel(model_type: str, generation_steps: int, max_new_tokens: int, 
                                      temperature: float, messages: list, N: int, seed: int = 42, 
                                      batch_size: int = 8, nli_batch_size: int = 64):
    """
    Génère N variantes stochastiques ET calcule l'entropie sémantique par NLI en parallèle.
    Chaque GPU s'occupe de son propre Shard du début à la fin (Génération -> NLI) sur sa propre VRAM.
    """
    rank, world_size, local_rank = _init_distributed()

    if seed is not None:
        transformers.set_seed(seed + rank)

    if model_type.lower() == "llada":
        model_path = "GSAI-ML/LLaDA-8B-Instruct"
        sampler_config = SamplerConfig(steps=generation_steps, max_new_tokens=max_new_tokens, temperature=temperature)
        sampler_cls = dllm.core.samplers.MDLMSampler
    elif model_type.lower() == "dream":
        model_path = "Dream-org/Dream-v0-Instruct-7B"
        sampler_config = DreamSamplerConfig(steps=generation_steps, max_new_tokens=max_new_tokens, temperature=temperature)
        sampler_cls = dllm.pipelines.dream.sampler.DreamSampler
    else:
        raise ValueError("model_type doit être 'llada' ou 'dream'")

    resolved_path = dllm.utils.resolve_with_base_env(model_path, "BASE_MODELS_DIR")
    print(f"[rank {rank}] Loading generation model: {resolved_path}...", flush=True)
    model = dllm.utils.get_model(
        model_name_or_path=resolved_path,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    ).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=resolved_path)
    sampler = sampler_cls(model=model, tokenizer=tokenizer)

    shard_indices = list(range(rank, len(messages), world_size))
    shard_messages = [messages[i] for i in shard_indices]
    print(f"[rank {rank}/{world_size}] Assigned {len(shard_messages)} prompts for Gen + NLI", flush=True)

    shard_results = {msg[0]['content']: [] for msg in shard_messages}

    # 1. Génération des variantes
    for v in range(N):
        for start_idx in range(0, len(shard_messages), batch_size):
            end_idx = min(start_idx + batch_size, len(shard_messages))
            batch_messages = shard_messages[start_idx:end_idx]

            inputs = tokenizer.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=True)
            outputs = sampler.sample(inputs, sampler_config, return_dict=True)
            flat_sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

            for idx, msg in enumerate(batch_messages):
                shard_results[msg[0]['content']].append(flat_sequences[idx])

    # Nettoyage de la mémoire du LLM pour libérer de la place pour DeBERTa
    del model, sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Pipeline NLI sur le GPU local
    print(f"[rank {rank}] Initializing NLI pipeline (DeBERTa) on GPU {local_rank}...", flush=True)
    nli_pipeline = _build_nli_pipeline(local_rank)

    # 3. Calcul de l'entropie sémantique en local sur le shard
    shard_detailed = calculate_semantic_entropy_nli(shard_results, nli_pipeline, nli_batch_size=nli_batch_size, rank=rank)

    # Nettoyage final du pipeline NLI
    del nli_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Sauvegarde temporaire du shard calculé
    tmp_dir = "/tmp/semantic_entropy_shards"
    os.makedirs(tmp_dir, exist_ok=True)
    shard_path = os.path.join(tmp_dir, f"detailed_shard_rank{rank}.json")
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(shard_detailed, f)
    print(f"[rank {rank}] Saved detailed NLI shard ({len(shard_detailed)} prompts)", flush=True)

    if world_size > 1:
        # Correction du warning NCCL en spécifiant explicitement le device_id local
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    # 5. Fusion finale sur le rank 0
    if rank == 0:
        grouped_detailed = {}
        for r in range(world_size):
            p = os.path.join(tmp_dir, f"detailed_shard_rank{r}.json")
            with open(p, "r", encoding="utf-8") as f:
                shard = json.load(f)
            grouped_detailed.update(shard)
            if os.path.exists(p):
                os.remove(p)
        print(f"[rank 0] Merged {len(grouped_detailed)} processed prompts from {world_size} GPUs", flush=True)
        return grouped_detailed
    else:
        return None

def _build_nli_pipeline(local_rank):
    """Initialise proprement le pipeline NLI DeBERTa-v2 sur le GPU ciblé via local_rank."""
    if torch.cuda.is_available() and local_rank != -1:
        device_str = f"cuda:{local_rank}"
        pipeline_device = local_rank
        dtype = torch.float16  # Parfaitement stable pour DeBERTa-v2-xlarge sur RTX 6000 Ada
    else:
        device_str = "cpu"
        pipeline_device = -1  # -1 indique à Hugging Face d'utiliser le CPU
        dtype = torch.float32

    # Utilisation de la version v2 de DeBERTa (XLarge est la version standard pour MNLI en v2)
    model_id = "microsoft/deberta-v2-xlarge-mnli"

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=dtype
    )

    # TRUC CLEF : Envoyer explicitement les poids du modèle sur le GPU alloué à ce Rank
    if device_str != "cpu":
        print(f"[rank {local_rank}] Moving DeBERTa weights to {device_str}...", flush=True)
        model = model.to(device_str)

    pipeline_nli = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=pipeline_device,  # Aligne le pipeline sur le même appareil
        truncation=True,
        max_length=256,
    )
    return pipeline_nli


def _batched_entailment_flags(pairs, pipeline_nli, batch_size=64):
    if not pairs:
        return []
    inputs = [{"text": p, "text_pair": h} for p, h in pairs]
    outputs = pipeline_nli(inputs, batch_size=batch_size)
    flags = []
    for out in outputs:
        res = out[0] if isinstance(out, list) else out
        flags.append(res["label"].lower() == "entailment" and res["score"] > 0.5)
    return flags


def _partition_into_nli_clusters_batched(variants, pipeline_nli, batch_size=64):
    clusters = []          
    rep_index_of_cluster = []  
    seen_text_to_cluster = {}  

    for idx, text in enumerate(variants):
        stripped = text.strip()
        if not stripped:
            clusters.append([idx])
            rep_index_of_cluster.append(idx)
            continue

        if stripped in seen_text_to_cluster:
            clusters[seen_text_to_cluster[stripped]].append(idx)
            continue

        if not clusters:
            clusters.append([idx])
            rep_index_of_cluster.append(idx)
            seen_text_to_cluster[stripped] = 0
            continue

        rep_texts = [variants[r] for r in rep_index_of_cluster]
        forward_pairs = [(text, rep) for rep in rep_texts]
        backward_pairs = [(rep, text) for rep in rep_texts]
        all_pairs = forward_pairs + backward_pairs

        flags = _batched_entailment_flags(all_pairs, pipeline_nli, batch_size=batch_size)
        n = len(rep_texts)
        forward_flags = flags[:n]
        backward_flags = flags[n:]

        assigned = False
        for c_i in range(n):
            if forward_flags[c_i] and backward_flags[c_i]:
                clusters[c_i].append(idx)
                assigned = True
                break

        if not assigned:
            clusters.append([idx])
            rep_index_of_cluster.append(idx)
            seen_text_to_cluster[stripped] = len(clusters) - 1

    return clusters


def calculate_semantic_entropy_nli(dict_results, pipeline_nli, nli_batch_size=64, rank=0):
    detailed_results = {}
    total_questions = len(dict_results)

    for q_idx, (question, variantes) in enumerate(dict_results.items(), 1):
        N = len(variantes)
        if N == 0:
            detailed_results[question] = {"variants": [], "clusters": [], "semantic_entropy": 0.0}
            continue

        clusters_indices = _partition_into_nli_clusters_batched(variantes, pipeline_nli, batch_size=nli_batch_size)
        clusters_text = [[variantes[idx] for idx in cluster] for cluster in clusters_indices]

        probabilities = [len(c) / N for c in clusters_indices]
        shannon_entropy = -sum(p * np.log(p + 1e-12) for p in probabilities)

        detailed_results[question] = {
            "variants": variantes,
            "clusters": clusters_text,
            "semantic_entropy": float(shannon_entropy)
        }

        if q_idx % 20 == 0 or q_idx == total_questions:
            print(f"  [rank {rank}] -> Avancement NLI : {q_idx}/{total_questions} questions filtrées.", flush=True)

    return detailed_results


# =========================================================================
# METRIQUES STANDARDISÉES
# =========================================================================
def evaluate(scores, labels, name="SemanticEntropy"):
    valid = ~np.isnan(scores)
    scores = scores[valid]
    labels = labels[valid]

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    thresholds = np.linspace(scores.min(), scores.max(), 201)
    accs = [accuracy_score(labels, (scores >= t).astype(int)) for t in thresholds]
    best_acc = max(accs)

    y_pred = (scores >= np.median(scores)).astype(int)
    acc = accuracy_score(labels, y_pred)

    return {
        "name": name,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(best_acc),
        "n_samples": len(labels),
    }


# =========================================================================
# EXECUTION
# =========================================================================
def run_config(config_name, n_variants=10, balance_seed=42, nli_batch_size=64):
    cfg = CONFIGS[config_name]
    eval_json = cfg["eval_json"]
    rank, world_size, local_rank = _init_distributed()

    is_dream = "dream" in config_name.lower()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  SEMANTIC ENTROPY NLI PARALLEL: {cfg['name']} ({config_name})")
        print(f"{'='*70}")

    if not os.path.exists(eval_json):
        if rank == 0:
            print(f"  [SKIP] eval_json introuvable: {eval_json}")
        return None

    labels_raw, indices_raw, n_missing_label, idx_to_prompt = load_eval_data(eval_json)
    if len(labels_raw) == 0:
        return None

    labels_balanced = labels_raw
    indices_balanced = indices_raw
    balance_stats = {
        "n_pos_before": int(np.sum(labels_raw == 1)),
        "n_neg_before": int(np.sum(labels_raw == 0)),
        "n_kept_per_class": "N/A (Sequential Split)",
    }

    train_idx, test_idx = get_train_test_split(len(labels_balanced))

    balanced_prompts = [[{"role": "user", "content": idx_to_prompt[int(idx)]}] for idx in indices_balanced]

    model_type = "dream" if "dream" in config_name.lower() else "llada"
    steps = 16 if "16" in config_name else (128 if "128" in config_name else 16)
    tokens = 32 if "32" in config_name else (128 if "128" in config_name else 32)

    # Appel de la fonction fusionnée (Génération + NLI répartis sur tous les GPUs)
    dict_detailed = generate_and_evaluate_nli_parallel(
        model_type=model_type,
        generation_steps=steps,
        max_new_tokens=tokens,
        temperature=0.5,
        messages=balanced_prompts,
        N=n_variants,
        seed=42,
        batch_size=8,
        nli_batch_size=nli_batch_size
    )

    if rank != 0:
        return None

    # À partir d'ici, seul le rank 0 traite les métriques agrégées globales
    scores_vector = np.array([dict_detailed[idx_to_prompt[int(idx)]]["semantic_entropy"] for idx in indices_balanced])

    test_scores = scores_vector[test_idx]
    test_labels = labels_balanced[test_idx]

    results = evaluate(test_scores, test_labels, name=f"SemanticEntropy_{config_name}")

    individual_samples_log = {}
    for i, idx in enumerate(indices_balanced):
        q_text = idx_to_prompt[int(idx)]
        info = dict_detailed[q_text]

        individual_samples_log[str(int(idx))] = {
            "question": q_text,
            "label_hallucination": int(labels_balanced[i]),
            "variants": info["variants"],
            "clusters": info["clusters"],
            "semantic_entropy": info["semantic_entropy"],
            "is_in_test_split": bool(i in test_idx)
        }

    results.update({
        "config": config_name,
        "n_collapse_dropped": 0,
        "n_balanced_pool": len(labels_balanced),
        "samples": individual_samples_log
    })

    print(f"  ROC-AUC: {results['test_roc_auc']:.4f}  |  PR-AUC: {results['test_pr_auc']:.4f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Entropy Multi-GPU Parallelized pipeline (Gen + NLI).")
    parser.add_argument("--config", type=str, default="all", choices=list(CONFIGS.keys()) + ["all"])
    parser.add_argument("--n_variants", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--nli_batch_size", type=int, default=64,
                         help="Taille de batch pour DeBERTa NLI.")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = _init_distributed()

    output_dir = args.output_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "eval"))
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]

    all_results = {}
    for config_name in configs_to_run:
        result = run_config(config_name, n_variants=args.n_variants, balance_seed=args.balance_seed,
                             nli_batch_size=args.nli_batch_size)
        if rank == 0 and result is not None:
            all_results[config_name] = result

    if rank == 0 and all_results:
        print(f"\n\n{'='*70}\n  SUMMARY TABLE : SEMANTIC ENTROPY (NLI INTERN PARALLEL)\n{'='*70}")
        for name, r in all_results.items():
            print(f"  {name:<45} ROC-AUC: {r['test_roc_auc']:.4f} | PR-AUC: {r['test_pr_auc']:.4f}")

        suffix = CONFIGS[configs_to_run[0]]["name"] if len(configs_to_run) == 1 else "all"
        out_path = os.path.join(output_dir, f"semantic_entropy_results_{suffix}_debertav1.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[SUCCESS] Saved to: {out_path}")

    _cleanup_distributed(world_size)


if __name__ == "__main__":
    main()