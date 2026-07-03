# PipelineTest/sampling

## Sampling & Evaluation Pipeline

Scripts for generating diffusion model outputs and evaluating hallucinations.

### Files

| File | Description |
|------|-------------|
| `sample_answers.py` | Main sampling script: generates answers with LLaDA/Dream diffusion models on TriviaQA |
| `sample_same_example.py` | Generates multiple samples for the same question (for variability analysis) |
| `load_data.py` | Loads and preprocesses TriviaQA dataset |
| `metric_qwen.py` | Hallucination judge using Qwen3-8B (TraceDet protocol) |

### Usage

```bash
# Generate answers (requires GPU)
python -m PipelineTest.sampling.sample_answers

# Evaluate hallucinations
python -m PipelineTest.sampling.metric_qwen
```

### Outputs

- `res/<model_config>/outputs_<config>.pt` — sampler outputs (entropy histories, masks, proposed tokens)
- `res/<model_config>/tokenizer_<config>.pt` — saved tokenizer
- `res/eval/results_triviaqa_<config>.json` — hallucination labels from Qwen judge
