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

BRIEF_SUFFIX = " please answer briefly"

def _apply_brief_suffix(messages):
    """Ajoute BRIEF_SUFFIX à la fin de chaque message utilisateur."""
    new_messages = []
    for convo in messages:
        new_convo = [
            {**msg, "content": msg["content"] + BRIEF_SUFFIX} if msg.get("role") == "user" else msg
            for msg in convo
        ]
        new_messages.append(new_convo)
    return new_messages

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
    # Libère la mémoire GPU dans tous les cas, même en single-GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()




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
    sampler_name="llada",
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
        f"[rank {rank}] Assigned {total_local} samples over "
        f"{num_local_batches} batches (temperature={temp_label}).",
        flush=True,
    )

    for start_idx in range(0, len(group_messages), batch_size):
        batch_t0 = time.time()
        end_idx = min(start_idx + batch_size, len(group_messages))
        batch_messages = group_messages[start_idx:end_idx]
        batch_local_indices = group_indices[start_idx:end_idx]
        batch_labels = group_labels[start_idx:end_idx]
        batch_id = start_idx // batch_size + 1

        print(
            f"[rank {rank}] Batch {batch_id}/{num_local_batches} "
            f"| local range [{start_idx}:{end_idx})",
            flush=True,
        )

        if sampler_name == "diffgemma":
            # Le tokenizer diffgemma (issu de l'AutoProcessor) ne renvoie pas des
            # ids avec apply_chat_template(..., tokenize=True) en mode batch --
            # on formate en texte puis on tokenize nous-mêmes par échantillon,
            # comme dans sample_answers_without_collapse.py et dllm/pipelines/diffusiongemma/eval.py.
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False,
                )
                for msg in batch_messages
            ]
            inputs = [
                tokenizer.encode(p, return_tensors="pt").squeeze(0)
                for p in formatted_prompts
            ]
        else:
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
            )

        outputs = sampler_obj.sample(inputs, sampler_config, return_dict=True)
        # sample_indices are set by the caller once shard-local -> global
        # mapping is known; stash the shard-local indices for now.
        outputs.sample_indices = torch.tensor(batch_local_indices, dtype=torch.long)

        batch_answers = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
        # print(getEachStepGeneratedSequence(outputs, tokenizer))
        for local_idx, label_entry, answer in zip(batch_local_indices, batch_labels, batch_answers):
            print(answer)
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
            f"[rank {rank}] Batch {batch_id}/{num_local_batches} done in "
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
    temp=0.0,
    data_seed=42,
):
    rank, world_size, local_rank = _init_distributed()
    model = None
    sampler_obj = None
    try:
        run_t0 = time.time()
        print(
            f"[rank {rank}/{world_size}] Starting run | dataset={dataset} | num_sample={num_sample} "
            f"| batch_size={batch_size} | suffix={suffix} | temp={temp}",
            flush=True,
        )

        script_args = ScriptArguments(model_name_or_path=MODEL_FOR_SAMPLER[sampler_name])
        if script_args.seed is not None:
            transformers.set_seed(script_args.seed + rank)

        print(f"[rank {rank}] Loading model/tokenizer: {script_args.model_name_or_path}", flush=True)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        if sampler_name == "diffgemma":
            # DiffusionGemma (26B-A4B) ne tient pas sur un seul GPU : on répartit
            # manuellement l'encodeur/décodeur sur les GPU 0 et 1, comme dans
            # sample_answers_without_collapse.py.
            from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
            model_path = MODEL_FOR_SAMPLER["diffgemma"]
            processor = AutoProcessor.from_pretrained(model_path)
            gemma_device_map = {
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
                gemma_device_map[f"model.encoder.language_model.layers.{i}"] = gpu
                gemma_device_map[f"model.decoder.layers.{i}"] = gpu
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=gemma_device_map,
            ).eval()
            tokenizer = processor.tokenizer
        else:
            model = dllm.utils.get_model(
                model_name_or_path=script_args.model_name_or_path,
            ).to(device).eval()
            tokenizer = dllm.utils.get_tokenizer(model_name_or_path=script_args.model_name_or_path)

        sampler_obj = sampler(model=model, tokenizer=tokenizer)

        # Build two configs sharing all fields except temperature.
        config = sampler_config_cls()
        config.temperature = temp

        load_fn = DATASET_LOADERS[dataset]
        messages, labels = load_fn(num_samples=num_sample, seed=data_seed)

        messages = _apply_brief_suffix(messages)
        print(f"[rank {rank}] Loaded dataset with {len(messages)} samples (data_seed={data_seed}).", flush=True)
        print(messages[0], flush=True)
        # --- Global (pre-shard) low/high temperature split ---
        # Splitting BEFORE sharding, with a fixed seed shared across ranks,
        # guarantees every rank agrees on which global index belongs to which
        # temperature regime, regardless of world_size.
        global_temp_map = {i: temp for i in range(len(messages))}

        print(
            f"[rank {rank}] Global split: all samples at temp={temp}",
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
        shard_local_positions = [
            pos for pos, global_idx in enumerate(shard_indices)
            if global_temp_map[global_idx] == temp
        ]
        

        # --- Run low-temperature batches (factual-leaning samples) ---
        BatchResults, BatchOutputs = _run_batches_for_group(
            sampler_obj=sampler_obj,
            sampler_config=config,
            tokenizer=tokenizer,
            group_indices=shard_local_positions,
            messages=shard_messages,
            labels=shard_labels,
            batch_size=batch_size,
            rank=rank,
            temp_label=temp,
            sampler_name=sampler_name,
        )

        # Remap shard-local indices -> global dataset indices for both results
        # and their corresponding outputs.sample_indices tensors.
        for r in BatchResults:
            r["index"] = shard_indices[r["index"]]
        local_results = BatchResults
        all_outputs = BatchOutputs

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



        # --- Each rank saves its shard (results + eval) to disk ---
        shard_path = os.path.join(artifacts_dir, f"_shard_rank{rank}.pt")
        torch.save({
            "indices": shard_indices,
            "results": local_results,

        }, shard_path)
        print(f"[rank {rank}] Saved shard to disk.", flush=True)

        # Clean up temporary per-rank JSON
        os.remove(local_results_path)

        # Barrier so rank 0 waits for all shards to be written
        if world_size > 1:
            dist.barrier()

        # Rank 0 merges everything into final artifacts
        if rank == 0:
            from PipelineTest.features.io import _pad_and_cat_tensors
            from dllm.core.samplers.base import BaseSamplerOutputCompleteHistory
            from dataclasses import fields as dc_fields

            results = []
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

            # Save final merged BaseSamplerOutputCompleteHistory (2048 samples)
            torch.save(merged, os.path.join(artifacts_dir, f"outputs_{suffix}.pt"))
            torch.save(tokenizer, os.path.join(artifacts_dir, f"tokenizer_{suffix}.pt"))

            # Save results JSON (without is_hallucination) in PipelineTest/res/
            res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "res/to_eval"))
            os.makedirs(res_dir, exist_ok=True)
            results_json_path = os.path.join(res_dir, f"results_{suffix}.json")
            with open(results_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Save eval JSON (with is_hallucination) in PipelineTest/res/eval/
            

            # Report accuracy broken down by temperature regime, so you can
            # sanity-check that low-temp really skews correct and high-temp
            # really skews hallucinated.
            
        

            print(f"[rank 0] Merged {merged.sequences.shape[0]} samples.", flush=True)
            print(f"[rank 0] Saved results_{dataset}_{suffix}.json ({len(results)} entries)", flush=True)
            print(f"[rank 0] All done in {time.time() - run_t0:.1f}s", flush=True)

        print(f"[rank {rank}] Finished in {time.time() - run_t0:.1f}s", flush=True)
    finally:
        try:
            del model, sampler_obj
            torch.cuda.empty_cache()
        except NameError:
            pass
        try:
            del sampler_obj
        except NameError:
            pass

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
    parser.add_argument("--canvas_length", type=int, default=None,
                        help="Canvas length for diffgemma sampler (default: same as --max_new_tokens)")
    parser.add_argument("--temp", type=float, default=0.0,
                        help=" temperature value, used to harvest factual-leaning samples (default: 0.0)")
    
   
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

    if args.sampler == "llada":
        overrides["__annotations__"]["block_size"] = int
        overrides["block_size"] = args.max_new_tokens
    elif args.sampler == "diffgemma":
        overrides["__annotations__"]["canvas_length"] = int
        overrides["canvas_length"] = args.canvas_length if args.canvas_length is not None else args.max_new_tokens
    config_cls = dataclass(type(f"CLI_{config_cls.__name__}", (config_cls,), overrides))

    suffix = (
        f"{args.sampler}_{args.steps}steps_{args.max_new_tokens}tokens_{args.dataset}_"
        f"{args.num_sample}samples_seed{args.data_seed}"
    )

    SampleAnswers(
        sampler=sampler_cls,
        num_sample=args.num_sample,
        batch_size=args.batch_size,
        suffix=suffix,
        dataset=args.dataset,
        sampler_config_cls=config_cls,
        sampler_name=args.sampler,
        temp=0,
        
        data_seed=args.data_seed,
    )
    os._exit(0) 