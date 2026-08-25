import sys
import os
import json
import argparse
import numpy as np
import re
import torch
import torch.distributed as dist
import transformers
from dataclasses import dataclass
import time


def _gpu_mem_str(local_rank):
    """Résumé mémoire GPU courant (alloué/réservé/total) pour vérifier la charge réelle."""
    if not torch.cuda.is_available() or local_rank < 0:
        return "cpu"
    alloc = torch.cuda.memory_allocated(local_rank) / 1024**3
    reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
    return f"{alloc:.1f}GiB alloc / {reserved:.1f}GiB reserved / {total:.1f}GiB total"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import dllm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline

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
# IMPLEMENTATION LOGICIELLE ROUGE-L (Lin et al., 2003)
# =========================================================================
def _tokenize(text):
    """Tokenization en minuscules: découpage sur les caractères non-alphanumériques."""
    return [t for t in re.split(r'[^a-z0-9]+', text.lower()) if t]


def _lcs_length(x, y):
    """Calcule la longueur de la plus longue sous-séquence commune (LCS)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(candidate, reference):
    """Calcule le score F1 ROUGE-L entre un candidat et une référence."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_lexical_features(dict_detailed, original_prompts):
    """
    Calcule les caractéristiques basées sur ROUGE-L entre la réponse principale
    (la première génération) et les autres variantes stochastiques d'auto-cohérence.
    """
    features = []
    for q_text in original_prompts:
        variantes = dict_detailed.get(q_text, [])
        if len(variantes) < 2:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        
        main_answer = variantes[0]
        alternative_answers = variantes[1:]

        precisions, recalls, f1s = [], [], []
        for alt in alternative_answers:
            p, r, f = rouge_l_f1(main_answer, alt)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        if len(f1s) == 0:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            f1s = np.array(f1s)
            precisions = np.array(precisions)
            recalls = np.array(recalls)
            best_idx = np.argmax(f1s)
            features.append([
                float(f1s.max()),            # MaxRougeLF1
                float(f1s.mean()),           # MeanRougeLF1
                float(precisions[best_idx]), # MaxRougeLPrec
                float(recalls[best_idx]),    # MaxRougeLRec
                float(f1s.var()),            # RougeLVar
            ])
            
    return np.array(features)


# =========================================================================
# GENERATION DES VARIANTES PARALLELISÉE MULTI-GPU
# =========================================================================
def generate_variants_parallel(model_type: str, generation_steps: int, max_new_tokens: int, 
                               temperature: float, messages: list, N: int, seed: int = 42, 
                               batch_size: int = 8):
    """
    Génère N variantes stochastiques parallèles sur l'infrastructure distribuée.
    Chaque GPU traite son propre shard de données de manière autonome.
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
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(local_rank)
        print(f"[rank {rank}] Generation model ({model_type}) resident on cuda:{local_rank} "
              f"({dev_name}) | {_gpu_mem_str(local_rank)}", flush=True)

    shard_indices = list(range(rank, len(messages), world_size))
    shard_messages = [messages[i] for i in shard_indices]
    t_shard_start = time.time()
    print(f"[rank {rank}/{world_size}] Assigned {len(shard_messages)} prompts for Stochastic Generation "
          f"(t={t_shard_start:.0f})", flush=True)

    shard_results = {msg[0]['content']: [] for msg in shard_messages}

    n_batches_per_variant = (len(shard_messages) + batch_size - 1) // max(batch_size, 1)
    total_batches = n_batches_per_variant * N
    batch_counter = 0

    # Échantillonnage de N variantes stochastiques indépendantes
    for v in range(N):
        for start_idx in range(0, len(shard_messages), batch_size):
            end_idx = min(start_idx + batch_size, len(shard_messages))
            batch_messages = shard_messages[start_idx:end_idx]

            inputs = tokenizer.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=True)
            outputs = sampler.sample(inputs, sampler_config, return_dict=True)
            flat_sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

            for idx, msg in enumerate(batch_messages):
                shard_results[msg[0]['content']].append(flat_sequences[idx])

            del outputs, flat_sequences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_counter += 1
            if batch_counter % 10 == 0 or batch_counter == total_batches:
                elapsed = time.time() - t_shard_start
                rate = batch_counter / elapsed if elapsed > 0 else 0.0
                eta = (total_batches - batch_counter) / rate if rate > 0 else float("inf")
                print(f"  [rank {rank}] gen batch {batch_counter}/{total_batches} "
                      f"(variant {v + 1}/{N}, {elapsed:.1f}s elapsed, {rate:.2f} batch/s, ETA {eta:.0f}s) | "
                      f"{_gpu_mem_str(local_rank)}", flush=True)

    print(f"[rank {rank}] Gen phase done at t={time.time():.0f} "
          f"(+{time.time() - t_shard_start:.1f}s since shard start)", flush=True)

    del model, sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Sauvegarde temporaire du shard généré localement
    tmp_dir = "/tmp/lexical_similarity_shards"
    os.makedirs(tmp_dir, exist_ok=True)
    shard_path = os.path.join(tmp_dir, f"gen_shard_rank{rank}.json")
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(shard_results, f)

    if world_size > 1:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Fusion globale sur le Rank 0
    if rank == 0:
        grouped_results = {}
        for r in range(world_size):
            p = os.path.join(tmp_dir, f"gen_shard_rank{r}.json")
            with open(p, "r", encoding="utf-8") as f:
                shard = json.load(f)
            grouped_results.update(shard)
            if os.path.exists(p):
                os.remove(p)
        print(f"[rank 0] Merged {len(grouped_results)} generated vectors from {world_size} GPUs", flush=True)
        return grouped_results
    else:
        return None


# =========================================================================
# CLASSIFIEUR LOGISTIC REGRESSION (GridSearchCV)
# =========================================================================
def evaluate_classifier(features, labels, train_idx, test_idx, name="LexicalSimilarity"):
    """Entraîne une régression logistique sur le train_idx et évalue sur le test_idx."""
    pipe = SkPipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42)),
    ])
    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logreg__penalty": ["l1", "l2"],
        "logreg__solver": ["saga"],
    }

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_scores = best.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    acc = accuracy_score(y_test, y_pred)

    thresholds = np.linspace(y_scores.min(), y_scores.max(), 201)
    accs = [accuracy_score(y_test, (y_scores >= t).astype(int)) for t in thresholds]
    best_acc = max(accs)
    best_thresh = thresholds[np.argmax(accs)]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "name": name,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_accuracy": float(acc),
        "test_best_accuracy": float(best_acc),
        "test_best_threshold": float(best_thresh),
        "n_samples": len(y_test),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }


# =========================================================================
# EXECUTION DE LA CONFIGURATION
# =========================================================================
def run_config(config_name, n_variants=5, balance_seed=42):
    cfg = CONFIGS[config_name]
    eval_json = cfg["eval_json"]
    rank, world_size, local_rank = _init_distributed()

    is_dream = "dream" in config_name.lower()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  LEXICAL SIMILARITY PARALLEL PIPELINE: {cfg['name']} ({config_name})")
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
    train_idx, test_idx = get_train_test_split(len(labels_balanced))

    balanced_prompts = [[{"role": "user", "content": idx_to_prompt[int(idx)]}] for idx in indices_balanced]

    model_type = "dream" if "dream" in config_name.lower() else "llada"
    steps = 16 if "16" in config_name else (128 if "128" in config_name else 16)
    tokens = 32 if "32" in config_name else (128 if "128" in config_name else 32)

    # Lancement du processus de génération parallèle multi-GPU
    dict_detailed = generate_variants_parallel(
        model_type=model_type,
        generation_steps=steps,
        max_new_tokens=tokens,
        temperature=0.5, # Forcer une température > 0 pour générer des variantes divergentes
        messages=balanced_prompts,
        N=n_variants,
        seed=42,
        batch_size=8
    )

    if rank != 0:
        return None

    # Étape locale au Rank 0 : Extraction des caractéristiques ROUGE-L et Classification
    original_prompts = [idx_to_prompt[int(idx)] for idx in indices_balanced]
    features = compute_lexical_features(dict_detailed, original_prompts)

    print(f"  [INFO] Extracted lexical features matrix: {features.shape}")
    print(f"  Mean MaxRougeLF1: {features[:, 0].mean():.4f} "
          f"(halluc: {features[labels_balanced == 1, 0].mean():.4f} | correct: {features[labels_balanced == 0, 0].mean():.4f})")

    results = evaluate_classifier(features, labels_balanced, train_idx, test_idx, name=f"LexicalSimilarity_{config_name}")

    # Logs de traçabilité des index pour vérification
    question_ids_train = indices_balanced[train_idx].tolist()
    question_ids_test = indices_balanced[test_idx].tolist()
    print("\n" + "="*50)
    print(f"  INDEX TRIVIAQA SÉLECTIONNÉS POUR LE TEST SPLIT (N={len(question_ids_test)})")
    print("="*50)
    print(question_ids_test[:20], "... (truncated)" if len(question_ids_test) > 20 else "")

    results.update({
        "config": config_name,
        "n_collapse_dropped": 0,
        "n_balanced_pool": len(labels_balanced),
    })

    print(f"  ROC-AUC: {results['test_roc_auc']:.4f}  |  PR-AUC: {results['test_pr_auc']:.4f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Lexical Similarity Multi-GPU Parallelized online pipeline (Gen + ROUGE-L).")
    parser.add_argument("--config", type=str, default="all", choices=list(CONFIGS.keys()) + ["all"])
    parser.add_argument("--n_variants", type=int, default=5, help="Nombre de générations stochastiques parallèles")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--balance_seed", type=int, default=42)
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
        result = run_config(config_name, n_variants=args.n_variants, balance_seed=args.balance_seed)
        if rank == 0 and result is not None:
            all_results[config_name] = result

    if rank == 0 and all_results:
        print(f"\n\n{'='*70}\n  SUMMARY TABLE : LEXICAL SIMILARITY (ONLINE CROSS-GENERATION)\n{'='*70}")
        for name, r in all_results.items():
            print(f"  {name:<45} ROC-AUC: {r['test_roc_auc']:.4f} | PR-AUC: {r['test_pr_auc']:.4f}")

        suffix = CONFIGS[configs_to_run[0]]["name"] if len(configs_to_run) == 1 else "all"
        out_path = os.path.join(output_dir, f"lexical_similarity_results_{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[SUCCESS] Saved online results to: {out_path}")

    _cleanup_distributed(world_size)


if __name__ == "__main__":
    main()