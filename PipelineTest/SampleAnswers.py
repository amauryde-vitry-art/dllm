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
    steps: int = 64
    max_new_tokens: int = 64
    block_size: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"

@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def SampleAnswerTrivaQA(sampler, num_sample, batch_size, suffix):
    messages, labels = load_triviaqa(num_samples=num_sample)
    all_answers = []
    all_outputs = []
    tokenizer = None

    for start_idx in range(0, len(messages), batch_size):
        end_idx = min(start_idx + batch_size, len(messages))
        batch_messages = messages[start_idx:end_idx]

        outputs, tokenizer = CreateBaseSampleWithHistory(
            sampler,
            batch_messages,
            SamplerConfig,
            ScriptArguments,
        )

        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        
        batch_answers = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
        all_answers.extend(batch_answers)
        all_outputs.append(outputs)

        torch.cuda.empty_cache()

    answers = all_answers


    # Save full output object and lightweight artifacts for later post-processing.
    artifacts_dir = os.path.join(os.path.dirname(__file__), f"results_{suffix}")
    os.makedirs(artifacts_dir, exist_ok=True)

    torch.save(all_outputs, os.path.join(artifacts_dir, f"outputs_{suffix}.pt"))
    torch.save(tokenizer, os.path.join(artifacts_dir, f"tokenizer_{suffix}.pt"))

    # Build results JSON for metric_qwen.py evaluation
    results = []
    for i, (label_entry, answer) in enumerate(zip(labels, answers)):
        results.append({
            "question": label_entry["question"],
            "label": label_entry["label"],
            "answer": answer,
            "index": i
        })

    results_json_path = os.path.join(artifacts_dir, f"results_triviaqa_{suffix}.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)



    from metric_qwen import load_qwen, compute_correctness_truthfulqa

    tokenizer_qwen, model_qwen = load_qwen()
    correctness = compute_correctness_truthfulqa(results_json_path, model_qwen, tokenizer_qwen)
    print(f"Accuracy: {sum(correctness)/len(correctness):.2%}")



SampleAnswerTrivaQA(dllm.core.samplers.MDLMSamplerWithCompleteHistory, num_sample=256, batch_size=128, suffix="64_step")

