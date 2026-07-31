import sys
import os
import time


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import json
import transformers
import torch.distributed as dist
import dllm
from dllm.pipelines.dream import DreamSamplerWithCompleteHistory
from dataclasses import dataclass
from load_data import load_triviaqa



@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 64
    max_new_tokens: int = 64
    block_size: int = 64
    temperature: float = 0.0
    remasking: str = "entropy"

@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 64
    max_new_tokens: int = 64
    alg: str = "entropy"
    alg_temp: float = 0.0
    top_p: float = 1.0

@dataclass
class ScriptArguments:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    seed: int = 42
    visualize: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


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


def SampleAnswerTrivaQA(sampler, num_sample, batch_size, suffix, sampler_config_cls=SamplerConfig):
    rank, world_size, _ = _init_distributed()
    run_t0 = time.time()
    print(f"[rank {rank}/{world_size}] Starting run | num_sample={num_sample} | batch_size={batch_size} | suffix={suffix}", flush=True)

    script_args = ScriptArguments()
    if script_args.seed is not None:
        transformers.set_seed(script_args.seed + rank)

    print(f"[rank {rank}] Loading model/tokenizer: {script_args.model_name_or_path}", flush=True)
    model = dllm.utils.get_model(model_name_or_path=script_args.model_name_or_path).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=script_args.model_name_or_path)
    sampler_config = sampler_config_cls()
    sampler_obj = sampler(model=model, tokenizer=tokenizer)
    print(f"[rank {rank}] Model/tokenizer ready.", flush=True)

    messages, labels = load_triviaqa(num_samples=num_sample)
    print(f"[rank {rank}] Loaded dataset with {len(messages)} samples.", flush=True)

    # Shard work across ranks: rank r processes indices r, r+world_size, ...
    shard_indices = list(range(rank, len(messages), world_size))
    shard_messages = [messages[i] for i in shard_indices]
    shard_labels = [labels[i] for i in shard_indices]
    total_local = len(shard_messages)
    num_local_batches = (total_local + batch_size - 1) // batch_size
    print(
        f"[rank {rank}] Assigned {total_local} samples over {num_local_batches} batches.",
        flush=True,
    )

    local_results = []
    all_outputs = []

    for start_idx in range(0, len(shard_messages), batch_size):
        batch_t0 = time.time()
        end_idx = min(start_idx + batch_size, len(shard_messages))
        batch_messages = shard_messages[start_idx:end_idx]
        batch_indices = shard_indices[start_idx:end_idx]
        batch_labels = shard_labels[start_idx:end_idx]
        batch_id = start_idx // batch_size + 1

        print(
            f"[rank {rank}] Batch {batch_id}/{num_local_batches} | local range [{start_idx}:{end_idx})",
            flush=True,
        )

        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        outputs = sampler_obj.sample(inputs, sampler_config, return_dict=True)
        outputs.sample_indices = torch.tensor(batch_indices, dtype=torch.long)

        
        batch_answers = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

        for global_idx, label_entry, answer in zip(batch_indices, batch_labels, batch_answers):
            local_results.append({
                "question": label_entry["question"],
                "label": label_entry["label"],
                "answer": answer,
                "index": global_idx,
            })

        all_outputs.append(outputs)

        torch.cuda.empty_cache()
        print(
            f"[rank {rank}] Batch {batch_id}/{num_local_batches} done in {time.time() - batch_t0:.1f}s | cumulative={len(local_results)}/{total_local}",
            flush=True,
        )

    # Save artifacts for later post-processing (all via disk to avoid NCCL hangs).
    artifacts_dir = os.path.join(os.path.dirname(__file__), f"res/results_{suffix}")
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- Save outputs to disk FIRST to free GPU memory ---
    outputs_shard_path = os.path.join(artifacts_dir, f"_outputs_rank{rank}.pt")
    torch.save(all_outputs, outputs_shard_path)
    del all_outputs
    print(f"[rank {rank}] Saved outputs shard to disk.", flush=True)

    # --- Free LLaDA from VRAM before loading Qwen ---
    del model, sampler_obj
    torch.cuda.empty_cache()
    print(f"[rank {rank}] Freed sampling model from GPU.", flush=True)

    # --- Each rank evaluates its own shard with Qwen ---
    local_results_path = os.path.join(artifacts_dir, f"_results_rank{rank}.json")
    with open(local_results_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)

    from metric_qwen import load_qwen, compute_correctness_truthfulqa

    print(f"[rank {rank}] Loading Qwen evaluator on cuda:{rank}...", flush=True)
    tokenizer_qwen, model_qwen = load_qwen(device=rank)
    print(f"[rank {rank}] Running correctness evaluation on {len(local_results)} samples...", flush=True)
    local_correctness = compute_correctness_truthfulqa(local_results_path, model_qwen, tokenizer_qwen)
    local_accuracy = sum(local_correctness) / len(local_correctness) if local_correctness else 0.0
    print(f"[rank {rank}] Local accuracy: {local_accuracy:.2%} ({sum(local_correctness)}/{len(local_correctness)})", flush=True)

    # Read back the eval-enriched JSON (contains is_hallucination field)
    eval_dir = os.path.abspath(os.path.join(artifacts_dir, "..", "eval"))
    eval_rank_path = os.path.join(eval_dir, f"_results_rank{rank}.json")
    with open(eval_rank_path, "r", encoding="utf-8") as f:
        local_eval_results = json.load(f)
    os.remove(eval_rank_path)

    del model_qwen, tokenizer_qwen
    torch.cuda.empty_cache()

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
    if world_size > 1:
        dist.barrier()

    # Rank 0 merges everything into final artifacts
    if rank == 0:
        from AnalyseResults import mergeOutputsList, _pad_and_cat_tensors
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

            # Each batch_output corresponds to a slice of shard["indices"]
            idx_cursor = 0
            for batch_output in rank_outputs:
                bs = batch_output.sequences.shape[0]
                first_global_idx = shard["indices"][idx_cursor]
                ordered_batches.append((first_global_idx, batch_output))
                idx_cursor += bs

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

        # Save results JSON (without is_hallucination)
        results_json_path = os.path.join(artifacts_dir, f"results_triviaqa_{suffix}.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save eval JSON (with is_hallucination) in eval/ directory
        eval_dir = os.path.join(os.path.dirname(__file__), "res", "eval")
        os.makedirs(eval_dir, exist_ok=True)
        eval_json_path = os.path.join(eval_dir, f"results_triviaqa_{suffix}.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        accuracy = sum(all_correctness) / len(all_correctness) if all_correctness else 0.0
        print(f"[rank 0] Merged {merged.sequences.shape[0]} samples.", flush=True)
        print(f"[rank 0] Saved results_triviaqa_{suffix}.json ({len(results)} entries)", flush=True)
        print(f"[rank 0] Saved eval/results_triviaqa_{suffix}.json ({len(eval_results)} entries)", flush=True)
        print(f"Accuracy: {accuracy:.2%} ({sum(all_correctness)}/{len(all_correctness)})")
        print(f"[rank 0] All done in {time.time() - run_t0:.1f}s", flush=True)

    print(f"[rank {rank}] Finished in {time.time() - run_t0:.1f}s", flush=True)

    _cleanup_distributed(world_size)



SampleAnswerTrivaQA(dllm.core.samplers.MDLMSamplerWithCompleteHistory, num_sample=4, batch_size=32, suffix="LLaDa_64steps_64tokens_entropy", sampler_config_cls=SamplerConfig)

# SampleAnswerTrivaQA(DreamSamplerWithCompleteHistory, num_sample=2048, batch_size=32, suffix="DREAM_64steps_64tokens_entropy", sampler_config_cls=DreamSamplerConfig)
