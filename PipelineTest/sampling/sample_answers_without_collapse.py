import sys
import os
import time
import argparse
import random


# Add both the sampling dir (for load_data, metric_qwen, AnalyseResults) and project root (for PipelineTest, dllm)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEachStepGeneratedSequence

import torch
import json
import transformers
import torch.distributed as dist
import dllm
from dataclasses import dataclass
from load_data import load_triviaqa, load_naturalquestion, load_hotpotqa
from dllm.pipelines.diffusiongemma.sampler import DiffusionGemmaSamplerWithCompleteHistory, DiffusionGemmaSamplerConfig


DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "naturalquestion": load_naturalquestion,
    "hotpotqa": load_hotpotqa,
}



@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"

@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 16
    max_new_tokens: int = 32
    alg: str = "maskgit_plus"
    alg_temp: float = 0.0
    top_p: float = 1.0
    temperature: float = 0.0

@dataclass
class GemmaSamplerConfig(DiffusionGemmaSamplerConfig):
    max_new_tokens: int = 64
    steps: int = 64
    max_temperature: float = None
    min_temperature: float = None
    return_dict: bool = True
    canvas_length: int = 64


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


MODEL_FOR_SAMPLER = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "diffgemma": "google/diffusiongemma-26B-A4B-it",
}


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


def _cleanup_distributed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _split_low_high_temp(num_items, low_temp_frac, seed):
    """
    Deterministically split `num_items` indices [0..num_items) into a
    low-temperature group and a high-temperature group.

    Shuffling (with a fixed seed) avoids any bias from dataset ordering
    (e.g. questions sorted by difficulty/topic) leaking into which
    temperature regime a sample gets assigned to.

    Returns two sorted lists of indices: (low_temp_indices, high_temp_indices).
    """
    all_idx = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(all_idx)

    n_low = int(round(num_items * low_temp_frac))
    low_idx = sorted(all_idx[:n_low])
    high_idx = sorted(all_idx[n_low:])
    return low_idx, high_idx


def _run_batches_for_group(
    sampler_obj,
    sampler_config,
    tokenizer,
    group_indices,
    messages,
    labels,
    batch_size,
    rank,
    temp_label,
    group_name,
    processor=None,
):
    """
    Runs sampling over a subset of the (already-sharded) dataset using a
    single sampler_config (i.e. a fixed temperature regime).

    `group_indices` are positions into the shard-local `messages`/`labels`
    lists (NOT global dataset indices) that belong to this temperature group.
    """
    local_results = []
    local_outputs = []

    group_messages = [messages[i] for i in group_indices]
    group_labels = [labels[i] for i in group_indices]
    total_local = len(group_messages)
    num_local_batches = (total_local + batch_size - 1) // batch_size

    print(
        f"[rank {rank}] [{group_name}] Assigned {total_local} samples over "
        f"{num_local_batches} batches (temperature={temp_label}).",
        flush=True,
    )

    for start_idx in range(0, len(group_messages), batch_size):
        batch_t0 = time.time()
        end_idx = min(start_idx + batch_size, len(group_messages))
        batch_messages = group_messages[start_idx:end_idx]
        # Map back to shard-local indices, then these get remapped to global
        # indices by the caller before being used as sample_indices.
        batch_local_indices = group_indices[start_idx:end_idx]
        batch_labels = group_labels[start_idx:end_idx]
        batch_id = start_idx // batch_size + 1

        print(
            f"[rank {rank}] [{group_name}] Batch {batch_id}/{num_local_batches} "
            f"| local range [{start_idx}:{end_idx})",
            flush=True,
        )

        # Encodage propre séquence par séquence puis conversion en liste de tenseurs 1D PyTorch
        if processor is not None:
          # Pour DiffGemma
          formatted_prompts = [
              processor.apply_chat_template(
                  msg, add_generation_prompt=True, tokenize=False, 
              )
              for msg in batch_messages
          ]
          inputs = [
              tokenizer.encode(p, return_tensors="pt").squeeze(0)
              for p in formatted_prompts
          ]
        else:
          # Pour LLaDA / Dream / Samplers standards
          formatted_prompts = [
              tokenizer.apply_chat_template(
                  msg, add_generation_prompt=True, tokenize=False
              )
              for msg in batch_messages
          ]
          inputs = [
              tokenizer.encode(p, return_tensors="pt").squeeze(0)
              for p in formatted_prompts
          ]


        outputs = sampler_obj.sample(inputs, sampler_config, return_dict=True)
        # sample_indices are set by the caller once shard-local -> global
        # mapping is known; stash the shard-local indices for now.
        outputs.sample_indices = torch.tensor(batch_local_indices, dtype=torch.long)

        batch_answers = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
        for local_idx, label_entry, answer in zip(batch_local_indices, batch_labels, batch_answers):
            local_results.append({
                "question": label_entry["question"],
                "label": label_entry["label"],
                "answer": answer,
                "index": local_idx,  # shard-local for now; remapped below
                "temperature": temp_label,
                # "proposed_sequences": proposed_sequences,
            })
        # results_json_path = f"results_{group_name}_batch{batch_id}.json"
        # with open(results_json_path, "w", encoding="utf-8") as f:
        #     json.dump(local_results, f, indent=2, ensure_ascii=False)

        local_outputs.append(outputs)

        torch.cuda.empty_cache()
        print(
            f"[rank {rank}] [{group_name}] Batch {batch_id}/{num_local_batches} done in "
            f"{time.time() - batch_t0:.1f}s | cumulative={len(local_results)}/{total_local}",
            flush=True,
        )

    return local_results, local_outputs


def SampleAnswers(
    sampler,
    num_sample,
    batch_size,
    suffix,
    dataset="triviaqa",
    sampler_config_cls=SamplerConfig,
    sampler_name="llada",
    low_temp_frac=0.4,
    temp_low=0.0,
    temp_high=0.8,
    temp_split_seed=1234,
    eval_mode="qwen",
    data_seed=42,
):
    rank, world_size, local_rank = _init_distributed()
    run_t0 = time.time()
    print(
        f"[rank {rank}/{world_size}] Starting run | dataset={dataset} | num_sample={num_sample} "
        f"| batch_size={batch_size} | suffix={suffix} | low_temp_frac={low_temp_frac} "
        f"| temp_low={temp_low} | temp_high={temp_high}",
        flush=True,
    )

    script_args = ScriptArguments(model_name_or_path=MODEL_FOR_SAMPLER[sampler_name])
    if script_args.seed is not None:
        transformers.set_seed(script_args.seed + rank)

    print(f"[rank {rank}] Loading model/tokenizer: {script_args.model_name_or_path}", flush=True)

    if sampler_name == "diffgemma":
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        model_path = MODEL_FOR_SAMPLER["diffgemma"]
        processor = AutoProcessor.from_pretrained(model_path)
        device_map = {
            "model.encoder.language_model.embed_tokens": 0,
            "model.decoder.embed_tokens": 0,
            "model.encoder.vision_tower": 0,
            "model.encoder.embed_vision": 0,
            "model.decoder.self_conditioning": 0,
            "lm_head": 0,
            "model.encoder.language_model.norm": 1,
            "model.decoder.norm": 1,
        }
        for i in range(30):
            gpu = 0 if i < 15 else 1
            device_map[f"model.encoder.language_model.layers.{i}"] = gpu
            device_map[f"model.decoder.layers.{i}"] = gpu
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map,
        ).eval()
        tokenizer = processor.tokenizer
        sampler_obj = sampler(model=model, tokenizer=tokenizer)
    else:
        processor = None
        model = dllm.utils.get_model(
            model_name_or_path=script_args.model_name_or_path,
            device_map={"": local_rank} if torch.cuda.is_available() else None,
        ).eval()
        
        # --- PATCH POUR LLADA / DREAM : Ajouter use_cache s'il manque ---
        if not hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        tokenizer = dllm.utils.get_tokenizer(
            model_name_or_path=script_args.model_name_or_path
        )
        sampler_obj = sampler(model=model, tokenizer=tokenizer)

    # Build two configs sharing all fields except temperature.
    config_low = sampler_config_cls()
    config_low.temperature = temp_low
    config_high = sampler_config_cls()
    config_high.temperature = temp_high
    print(
        f"[rank {rank}] Model/tokenizer ready. "
        f"config_low.temperature={config_low.temperature}, "
        f"config_high.temperature={config_high.temperature}",
        flush=True,
    )

    load_fn = DATASET_LOADERS[dataset]
    messages, labels = load_fn(num_samples=num_sample, seed=data_seed)
    print(f"[rank {rank}] Loaded dataset with {len(messages)} samples (data_seed={data_seed}).", flush=True)

    # --- Global (pre-shard) low/high temperature split ---
    # Splitting BEFORE sharding, with a fixed seed shared across ranks,
    # guarantees every rank agrees on which global index belongs to which
    # temperature regime, regardless of world_size.
    global_low_idx, global_high_idx = _split_low_high_temp(
        len(messages), low_temp_frac, temp_split_seed
    )
    global_temp_map = {}
    for i in global_low_idx:
        global_temp_map[i] = temp_low
    for i in global_high_idx:
        global_temp_map[i] = temp_high

    print(
        f"[rank {rank}] Global split: {len(global_low_idx)} low-temp "
        f"({len(global_low_idx)/len(messages):.1%}) / {len(global_high_idx)} high-temp "
        f"({len(global_high_idx)/len(messages):.1%})",
        flush=True,
    )

    # Shard work across ranks: rank r processes indices r, r+world_size, ...
    shard_indices = list(range(rank, len(messages), world_size))
    shard_messages = [messages[i] for i in shard_indices]
    shard_labels = [labels[i] for i in shard_indices]
    total_local = len(shard_messages)
    print(f"[rank {rank}] Assigned {total_local} samples (shard-local).", flush=True)

    # Within this shard, split shard-local positions into low/high groups
    # based on the GLOBAL split decided above.
    shard_low_local_positions = [
        pos for pos, global_idx in enumerate(shard_indices)
        if global_temp_map[global_idx] == temp_low
    ]
    shard_high_local_positions = [
        pos for pos, global_idx in enumerate(shard_indices)
        if global_temp_map[global_idx] == temp_high
    ]

    # --- Run low-temperature batches (factual-leaning samples) ---
    low_results, low_outputs = _run_batches_for_group(
        sampler_obj=sampler_obj,
        sampler_config=config_low,
        tokenizer=tokenizer,
        group_indices=shard_low_local_positions,
        messages=shard_messages,
        labels=shard_labels,
        batch_size=batch_size,
        rank=rank,
        temp_label=temp_low,
        group_name="LOW-TEMP",
        processor=processor,
    )

    # --- Run high-temperature batches (hallucination-leaning samples) ---
    high_results, high_outputs = _run_batches_for_group(
        sampler_obj=sampler_obj,
        sampler_config=config_high,
        tokenizer=tokenizer,
        group_indices=shard_high_local_positions,
        messages=shard_messages,
        labels=shard_labels,
        batch_size=batch_size,
        rank=rank,
        temp_label=temp_high,
        group_name="HIGH-TEMP",
        processor=processor,
    )

    # Remap shard-local indices -> global dataset indices for both results
    # and their corresponding outputs.sample_indices tensors.
    for r in low_results:
        r["index"] = shard_indices[r["index"]]
    for r in high_results:
        r["index"] = shard_indices[r["index"]]
    for o in low_outputs:
        o.sample_indices = torch.tensor(
            [shard_indices[i] for i in o.sample_indices.tolist()], dtype=torch.long
        )
    for o in high_outputs:
        o.sample_indices = torch.tensor(
            [shard_indices[i] for i in o.sample_indices.tolist()], dtype=torch.long
        )

    local_results = low_results + high_results
    all_outputs = low_outputs + high_outputs

    # Save artifacts for later post-processing (all via disk to avoid NCCL hangs).
    artifacts_dir = os.path.join(os.path.dirname(__file__), f"res/results_{suffix}")
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- Save outputs to disk FIRST to free GPU memory ---
    outputs_shard_path = os.path.join(artifacts_dir, f"_outputs_rank{rank}.pt")
    torch.save(all_outputs, outputs_shard_path)
    del all_outputs
    print(f"[rank {rank}] Saved outputs shard to disk.", flush=True)

    # --- Free sampling model from VRAM before loading Qwen ---
    del model, sampler_obj
    torch.cuda.empty_cache()
    print(f"[rank {rank}] Freed sampling model from GPU.", flush=True)

    # --- Each rank evaluates its own shard ---
    local_results_path = os.path.join(artifacts_dir, f"_results_rank{rank}.json")
    with open(local_results_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)

    # Both modes load Qwen for collapse detection; only "qwen" mode uses it for factuality too
    from metric_qwen_without_collapse import load_qwen, compute_correctness_truthfulqa, compute_correctness_exact_match

    # Dans sample_answers_without_collapse.py autour de la ligne 390 :
    eval_device = rank if world_size > 1 else (1 if torch.cuda.device_count() > 1 else 0)
    print(f"[rank {rank}] Loading Qwen evaluator on cuda:{eval_device}...", flush=True)
    tokenizer_qwen, model_qwen = load_qwen(device=eval_device)

    if eval_mode == "qwen":
        print(f"[rank {rank}] Running Qwen eval (collapse + factuality) on {len(local_results)} samples...", flush=True)
        local_correctness = compute_correctness_truthfulqa(local_results_path, model_qwen, tokenizer_qwen)
    else:
        print(f"[rank {rank}] Running exact-match eval (Qwen collapse + token overlap) on {len(local_results)} samples...", flush=True)
        local_correctness = compute_correctness_exact_match(local_results_path, model=model_qwen, tokenizer=tokenizer_qwen)

    del model_qwen, tokenizer_qwen
    torch.cuda.empty_cache()

    local_accuracy = sum(local_correctness) / len(local_correctness) if local_correctness else 0.0
    print(f"[rank {rank}] Local accuracy: {local_accuracy:.2%} ({sum(local_correctness)}/{len(local_correctness)})", flush=True)

    # Read back the eval-enriched JSON (contains is_hallucination field)
    eval_dir = os.path.abspath(os.path.join(artifacts_dir, "..", "eval"))
    eval_rank_path = os.path.join(eval_dir, f"_results_rank{rank}.json")
    with open(eval_rank_path, "r", encoding="utf-8") as f:
        local_eval_results = json.load(f)
    os.remove(eval_rank_path)

    # --- Each rank saves its shard (results + eval) to disk ---
    shard_path = os.path.join(artifacts_dir, f"_shard_rank{rank}.pt")
    torch.save({
        "indices": shard_indices,
        "results": local_results,
        "eval_results": local_eval_results,
        "correctness": local_correctness,
    }, shard_path)
    print(f"[rank {rank}] Saved shard to disk.", flush=True)

    # Clean up temporary per-rank JSON
    os.remove(local_results_path)

    # Barrier so rank 0 waits for all shards to be written
    # Synchronisation via le disque dur (évite d'appeler NCCL après déchargement VRAM)
    if world_size > 1:
      if rank != 0:
        flag_file = os.path.join(artifacts_dir, f"_done_rank{rank}.flag")
        with open(flag_file, "w") as f:
          f.write("done")
      else:
        for r in range(1, world_size):
          flag_file = os.path.join(artifacts_dir, f"_done_rank{r}.flag")
          while not os.path.exists(flag_file):
            time.sleep(0.5)
          if os.path.exists(flag_file):
            os.remove(flag_file)

    # Rank 0 merges everything into final artifacts
    if rank == 0:
        from AnalyseResults import mergeOutputsList
        from PipelineTest.features.io import _pad_and_cat_tensors
        from dllm.core.samplers.base import BaseSamplerOutputCompleteHistory
        from dataclasses import fields as dc_fields

        results = []
        eval_results = []
        all_correctness = []
        # Collect (global_start_idx, batch_output) pairs for ordering
        ordered_batches = []

        for r in range(world_size):
            # Load outputs shard
            op = os.path.join(artifacts_dir, f"_outputs_rank{r}.pt")
            rank_outputs = torch.load(op, map_location="cpu", weights_only=False)
            # Load metadata shard
            p = os.path.join(artifacts_dir, f"_shard_rank{r}.pt")
            shard = torch.load(p, map_location="cpu", weights_only=False)

            # Each batch_output now carries GLOBAL sample_indices directly
            # (remapped above), so we can order using its own first index
            # instead of relying on shard["indices"] positional slicing.
            for batch_output in rank_outputs:
                first_global_idx = int(batch_output.sample_indices[0].item())
                ordered_batches.append((first_global_idx, batch_output))

            results.extend(shard["results"])
            eval_results.extend(shard["eval_results"])
            all_correctness.extend(shard["correctness"])
            os.remove(op)
            os.remove(p)

        # Sort batches by their first global index so final merge is in order
        ordered_batches.sort(key=lambda x: x[0])
        outputs_list = [batch_output for _, batch_output in ordered_batches]

        # Merge all batch outputs into a single BaseSamplerOutputCompleteHistory
        merged = outputs_list[0]
        if len(outputs_list) > 1:
            merged = BaseSamplerOutputCompleteHistory()
            for f in dc_fields(BaseSamplerOutputCompleteHistory):
                name = f.name
                values = [getattr(o, name) for o in outputs_list]
                non_none_values = [v for v in values if v is not None]

                if len(non_none_values) == 0:
                    setattr(merged, name, None)
                    continue

                sample = non_none_values[0]

                if isinstance(sample, torch.Tensor):
                    setattr(merged, name, _pad_and_cat_tensors(non_none_values))
                    continue

                if isinstance(sample, list):
                    if len(sample) == 0:
                        setattr(merged, name, [])
                        continue
                    if all(isinstance(v, list) for v in non_none_values):
                        if all(len(v) == len(non_none_values[0]) for v in non_none_values):
                            if all(len(v) > 0 and isinstance(v[0], torch.Tensor) for v in non_none_values):
                                merged_history = []
                                for step_idx in range(len(non_none_values[0])):
                                    merged_history.append(
                                        _pad_and_cat_tensors([v[step_idx] for v in non_none_values])
                                    )
                                setattr(merged, name, merged_history)
                            else:
                                flat = []
                                for v in non_none_values:
                                    flat.extend(v)
                                setattr(merged, name, flat)
                        else:
                            flat = []
                            for v in non_none_values:
                                flat.extend(v)
                            setattr(merged, name, flat)
                        continue

                setattr(merged, name, sample)

        # Sort results
        results.sort(key=lambda x: x["index"])
        eval_results.sort(key=lambda x: x["index"])

        # Save final merged BaseSamplerOutputCompleteHistory (2048 samples)
        torch.save(merged, os.path.join(artifacts_dir, f"outputs_{suffix}.pt"))
        torch.save(tokenizer, os.path.join(artifacts_dir, f"tokenizer_{suffix}.pt"))

        # Save results JSON (without is_hallucination) in PipelineTest/res/
        res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "res"))
        os.makedirs(res_dir, exist_ok=True)
        results_json_path = os.path.join(res_dir, f"results_{suffix}.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save eval JSON (with is_hallucination) in PipelineTest/res/eval/
        eval_dir = os.path.join(res_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        eval_json_path = os.path.join(eval_dir, f"results_{dataset}_{suffix}.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        accuracy = sum(all_correctness) / len(all_correctness) if all_correctness else 0.0

        # Report accuracy broken down by temperature regime, so you can
        # sanity-check that low-temp really skews correct and high-temp
        # really skews hallucinated.
        low_correct = [c for r, c in zip(results, all_correctness) if r.get("temperature") == temp_low]
        high_correct = [c for r, c in zip(results, all_correctness) if r.get("temperature") == temp_high]
        low_acc = sum(low_correct) / len(low_correct) if low_correct else float("nan")
        high_acc = sum(high_correct) / len(high_correct) if high_correct else float("nan")

        print(f"[rank 0] Merged {merged.sequences.shape[0]} samples.", flush=True)
        print(f"[rank 0] Saved results_{dataset}_{suffix}.json ({len(results)} entries)", flush=True)
        print(f"[rank 0] Saved eval/results_{dataset}_{suffix}.json ({len(eval_results)} entries)", flush=True)
        print(f"Overall accuracy: {accuracy:.2%} ({sum(all_correctness)}/{len(all_correctness)})")
        print(f"  Low-temp  (T={temp_low}) accuracy: {low_acc:.2%} ({len(low_correct)} samples)")
        print(f"  High-temp (T={temp_high}) accuracy: {high_acc:.2%} ({len(high_correct)} samples)")
        print(f"[rank 0] All done in {time.time() - run_t0:.1f}s", flush=True)

    print(f"[rank {rank}] Finished in {time.time() - run_t0:.1f}s", flush=True)

    _cleanup_distributed(world_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample answers from a diffusion LLM with a mixed low/high temperature split")
    parser.add_argument("--dataset", type=str, choices=["triviaqa", "naturalquestion", "hotpotqa"], default="triviaqa",
                        help="Dataset to sample from (default: triviaqa)")
    parser.add_argument("--num_sample", type=int, default=2100, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sampler", type=str, choices=["llada", "dream", "diffgemma"], default="llada",
                        help="Sampler to use (default: llada)")
    parser.add_argument("--steps", type=int, default=16, help="Number of diffusion steps")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--low_temp_frac", type=float, default=1,
                        help="Fraction of the dataset sampled at temp_low (default: 0.4 -> 40%%)")
    parser.add_argument("--temp_low", type=float, default=0.0,
                        help="Low temperature value, used to harvest factual-leaning samples (default: 0.0)")
    parser.add_argument("--temp_high", type=float, default=0.8,
                        help="High temperature value, used to harvest hallucination-leaning samples "
                             "(recommended range: 0.7-1.0, default: 0.8)")
    parser.add_argument("--temp_split_seed", type=int, default=1234,
                        help="Seed controlling which global indices go to low vs high temp (default: 1234)")
    parser.add_argument("--eval", type=str, choices=["qwen", "exact_match"], default="qwen",
                        help="Evaluation method: 'qwen' (LLM judge) or 'exact_match' (token overlap, TDGNet-style)")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed for dataset shuffling/selection (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sampler == "llada":
        sampler_cls = dllm.core.samplers.MDLMSamplerWithCompleteHistory
        config_cls = SamplerConfig
    elif args.sampler == "diffgemma":
        sampler_cls = DiffusionGemmaSamplerWithCompleteHistory
        config_cls = GemmaSamplerConfig
    else:
        sampler_cls = dllm.pipelines.dream.sampler.DreamSamplerWithCompleteHistory
        config_cls = DreamSamplerConfig

    # Override steps/max_new_tokens from CLI
    overrides = {
        "__annotations__": {"steps": int, "max_new_tokens": int},
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.sampler == "diffgemma":
        overrides["__annotations__"]["canvas_length"] = int
        overrides["canvas_length"] = args.max_new_tokens
    elif args.sampler == "llada":
        overrides["__annotations__"]["block_size"] = int
        overrides["block_size"] = args.max_new_tokens
    config_cls = dataclass(type(f"CLI_{config_cls.__name__}", (config_cls,), overrides))

    suffix = (
        f"{args.sampler}_{args.steps}steps_{args.max_new_tokens}tokens_{args.dataset}_"
        f"{args.num_sample}samples_mixedtemp{int(args.low_temp_frac*100)}-{int((1-args.low_temp_frac)*100)}_eval{args.eval}_seed{args.data_seed}"
    )

    SampleAnswers(
        sampler=sampler_cls,
        num_sample=args.num_sample,
        batch_size=args.batch_size,
        suffix=suffix,
        dataset=args.dataset,
        sampler_config_cls=config_cls,
        sampler_name=args.sampler,
        low_temp_frac=args.low_temp_frac,
        temp_low=args.temp_low,
        temp_high=args.temp_high,
        temp_split_seed=args.temp_split_seed,
        eval_mode=args.eval,
        data_seed=args.data_seed,
    )