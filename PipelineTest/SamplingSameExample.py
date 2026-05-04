import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import json
from StudyAutoRegressionBehavior.GetInfoFromBaseSamplerOutput import getEachStepGeneratedSequence, getEachStepProposedSequence
from StudyAutoRegressionBehavior.PlotResults import getTxt

from StudyAutoRegressionBehavior.GenerateWithMDLMSampler import CreateBaseSampleWithHistory
from PipelineTest.AnalyseResults import PlotPipeline, GetInfoFromIndex
import dllm
from dataclasses import dataclass
from load_data import load_triviaqa
import sys
sys.argv = [sys.argv[0]] # Cette ligne "vide" les arguments du terminal



@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 32
    max_new_tokens: int = 32
    block_size: int = 32
    temperature: float = 1
    remasking: str = "low_confidence"

@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = None
    visualize: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )





def SamplingExamples(num_example, samples):
    messages, labels = load_triviaqa(num_samples=256)
    messages = [messages[num_example] for i in range(samples)]
    labels = [labels[num_example] for i in range(samples)]
    all_answers = []
    all_outputs = []
    tokenizer = None
    for i in range(len(messages)):
        seed_i = int(torch.seed()) & ((1 << 32) - 1)
        message_i = [messages[i]]
        outputs, tokenizer = CreateBaseSampleWithHistory(
            dllm.core.samplers.MDLMSamplerWithCompleteHistory,
            message_i,
            SamplerConfig,
            ScriptArguments,
            seed_override=seed_i,
        )

        inputs = tokenizer.apply_chat_template(
            message_i,
            add_generation_prompt=True,
            tokenize=True,
        )
        
        answers = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
        print(f'answers [{i}', answers)
        all_answers.extend(answers)
        all_outputs.append(outputs)
        torch.cuda.empty_cache()

    answers = all_answers

    # Save full output object and lightweight artifacts for later post-processing.
    artifacts_dir = os.path.join(os.path.dirname(__file__), f"results_{num_example}")
    os.makedirs(artifacts_dir, exist_ok=True)

    torch.save(all_outputs, os.path.join(artifacts_dir, f"outputs_{num_example}.pt"))
    torch.save(tokenizer, os.path.join(artifacts_dir, f"tokenizer_{num_example}.pt"))

    # Build results JSON for metric_qwen.py evaluation
    results = []
    for i, (label_entry, answer) in enumerate(zip(labels, answers)):
        results.append({
            "question": label_entry["question"],
            "label": label_entry["label"],
            "answer": answer,
            "index": i
        })

    results_json_path = os.path.join(artifacts_dir, f"results_triviaqa_{num_example}.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    from metric_qwen import load_qwen, compute_correctness_truthfulqa

    tokenizer_qwen, model_qwen = load_qwen()
    correctness = compute_correctness_truthfulqa(results_json_path, model_qwen, tokenizer_qwen)
    print(f"Accuracy: {sum(correctness)/len(correctness):.2%}")

sampling = 3
SamplingExamples(11, sampling)
outputs, tokenizer = torch.load('PipelineTest/results_11/outputs_11.pt', map_location="cpu", weights_only=False), torch.load('PipelineTest/results_11/tokenizer_11.pt', map_location="cpu", weights_only=False)
listIndex = [i for i in range(sampling)]
title_list = [f'MDML - Sample {i}' for i in listIndex]

PlotPipeline(listIndex, tokenizer, title_list, remasking=False, ouputpath='PipelineTest/results_11/outputs_11.pt', save_path='PipelineTest/results_11/teeeeeest')