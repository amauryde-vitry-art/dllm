"""
python -u examples/diffusiongemma/sample.py \
    --model_name_or_path /mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it
"""

from dataclasses import dataclass
import importlib.util
import sys
import types
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getEachStepMask, getLogProbs


def load_diffusiongemma_sampler():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from dllm.pipelines.diffusiongemma.sampler import DiffusionGemmaSampler, DiffusionGemmaSamplerConfig,DiffusionGemmaSamplerWithCompleteHistory
    return DiffusionGemmaSampler, DiffusionGemmaSamplerConfig, DiffusionGemmaSamplerWithCompleteHistory


DiffusionGemmaSampler, DiffusionGemmaSamplerConfig, DiffusionGemmaSamplerWithCompleteHistory = load_diffusiongemma_sampler()


@dataclass
class ScriptArguments:
    model_name_or_path: str = "google/diffusiongemma-26B-A4B-it"
    prompt: str | None = None
    prompt_file: str | None = None
    seed: int = 42
    dtype: str = "bfloat16"  


@dataclass
class SamplerConfig(DiffusionGemmaSamplerConfig):
    max_new_tokens: int = 128
    steps: int = 48
    entropy_bound: float = 0.1
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
    max_temperature: float = 0.8
    min_temperature: float = 0.4
    eos_token_id: int | None = None
    return_dict: bool = True


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

torch_dtype = getattr(torch, script_args.dtype)
processor = AutoProcessor.from_pretrained(script_args.model_name_or_path, device = "cuda", torch_dtype=torch_dtype)
device_map = {
    "model.encoder.language_model.embed_tokens": 0,
    "model.decoder.embed_tokens": 0,
    "model.encoder.vision_tower": 0,
    "model.encoder.embed_vision": 0,
    "model.decoder.self_conditioning": 0,
    "lm_head": 0,
}

# Assigner 0 et 1 (les 2 GPU qui seront vus par PyTorch)
for i in range(30):
    target_gpu = 0 if i < 15 else 1
    device_map[f"model.encoder.language_model.layers.{i}"] = target_gpu
    device_map[f"model.decoder.layers.{i}"] = target_gpu

device_map["model.encoder.language_model.norm"] = 1
device_map["model.decoder.norm"] = 1

model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    script_args.model_name_or_path,
    dtype=torch_dtype,
    device_map=device_map,
).eval()

sampler = DiffusionGemmaSamplerWithCompleteHistory(
    model=model,
    tokenizer=processor.tokenizer,
)

prompts = []
if script_args.prompt is not None:
    prompts.append(script_args.prompt)
if script_args.prompt_file is not None:
    prompts.extend(
        line.strip()
        for line in Path(script_args.prompt_file).read_text().splitlines()
        if line.strip()
    )
if not prompts:
    prompts = [
        "Give a concise explanation of text diffusion models.",
        "Write one short haiku about compilers.",
    ]

messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

outputs = sampler.sample(inputs, sampler_config)
print('Entropy:', getEntropy(outputs))
print('Each Step Mask:', getEachStepMask(outputs))
print('Log Probs:', getLogProbs(outputs))
sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
decoded = processor.batch_decode(sequences, skip_special_tokens=True)

print("\n" + "=" * 80)
print("TEST: diffusiongemma.generate() via dllm sampler".center(80))
print("=" * 80)
for index, text in enumerate(decoded):
    print("\n" + "-" * 80)
    print(f"[Case {index}]")
    print("-" * 80)
    print(text.strip())
print("\n" + "=" * 80 + "\n")