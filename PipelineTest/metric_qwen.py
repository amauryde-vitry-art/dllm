from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import os
import re
from tqdm import tqdm

def load_qwen(device=None):
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True, )
    if device is None:
        device_map = "auto"
    else:
        device_map = {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-8B',
        device_map=device_map,
        dtype=torch.float16,
        trust_remote_code=True,
    ).eval()
    return tokenizer, model

def get_prompt(sample):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': ''}
    ]
    messages[-1]['content'] = '''Your task is to determine whether the output contains hallucination. Follow these guidelines strictly:

Fluency Check: If the output is not fluent natural language (e.g., it contains garbled or unreadable text), it should be considered hallucinated.

Relevance Check: If the output contains many correct facts but does not directly answer the question, it should be considered hallucinated.

Support Check: If the output cannot be inferred from any of the reference answers, or contains information inconsistent with the reference answers, it should be considered hallucinated.

Exact Match Rule: If the output is supported by any one of the reference correct answers, it should be considered not hallucinated.

Semantic Match Rule: If the output is not directly supported by any reference answer, but is semantically similar (i.e., expresses the same meaning), it should be considered not hallucinated.

Unknown Answer Rule: If the reference answers include phrases like "This question cannot be answered", then an output like "I don't know" or "Cannot answer this question" should be considered not hallucinated.'''

    messages.append({'role': 'assistant',
                     'content': 'I understand. Please provide the question and the bot\'s answer.'})
    messages.append({'role': 'user', 'content': ''})


    user_input_for_judging = 'Question:{}\n\n'.format(sample['question'].strip())
    user_input_for_judging += 'The correct anwer example is as follow:\n'
    if isinstance(sample['label'], str):
        user_input_for_judging += '{}\n'.format(sample['label'].strip())
    else:
        for example_answer in sample['label']:
            if isinstance(example_answer, str):
                user_input_for_judging += '{}\n'.format(example_answer.strip())
            elif isinstance(example_answer, list):
                user_input_for_judging += ', '.join([example_answer[0].strip()]) + '\n'
            else:
                raise ValueError("Unsupported answer format: {}".format(type(example_answer)))


    user_input_for_judging += '\nThe bot replied as follow:\n'
    user_input_for_judging += '{}\n\n'.format(sample['answer'].strip())
    user_input_for_judging += 'Now please judge whether the bot\'s answer is hallucinated or not. If it is hallucinated, please answer "yes", otherwise answer "no".  Dont show thinking and put your answer in <answer> </answer>.\n'

    messages[-1]['content'] = user_input_for_judging

    return messages


def compute_correctness_truthfulqa(answer_path, model, tokenizer, batch_size=16):
    import time

    with open(answer_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError(f"Empty JSON file: {answer_path}")
    try:
        results = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {answer_path}: {e}") from e

    total = len(results)
    print(f"[eval] Starting evaluation of {total} samples (batch_size={batch_size})", flush=True)
    t0 = time.time()

    # Prepare all prompts upfront
    prompts = []
    for sample in results:
        messages = get_prompt(sample)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    # Ensure left-padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    correctness = [0] * total

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=64,
            )

        # Decode only the generated part (after input)
        input_len = inputs["input_ids"].shape[1]
        for i, idx in enumerate(range(batch_start, batch_end)):
            generated_tokens = output_ids[i][input_len:]
            output_text = extract_answer(
                tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
            )

            if output_text == "no":
                results[idx]['is_hallucination'] = "no"
            else:
                results[idx]['is_hallucination'] = "yes"

            if "yes" in output_text:
                correctness[idx] = 0  # hallucinated
            elif "no" in output_text:
                correctness[idx] = 1  # not hallucinated
            else:
                correctness[idx] = 0  # treat unclear answers as hallucinated

        # Progress log every batch
        elapsed = time.time() - t0
        done = batch_end
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        acc_so_far = sum(correctness[:done]) / done if done > 0 else 0
        print(
            f"[eval] {done}/{total} ({done*100//total}%) | "
            f"acc={acc_so_far:.2%} | {elapsed:.1f}s elapsed | ETA {eta:.0f}s",
            flush=True,
        )

    # Restore padding side
    tokenizer.padding_side = original_padding_side

    print(f"[eval] Done. Final accuracy: {sum(correctness)/total:.2%} in {time.time()-t0:.1f}s", flush=True)

    eval_dir = os.path.abspath(os.path.join(os.path.dirname(answer_path), "..", "eval"))
    os.makedirs(eval_dir, exist_ok=True)  

    eval_filename = os.path.basename(answer_path)
    eval_path = os.path.join(eval_dir, eval_filename)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return correctness

def compute_correctness_sciqa(answer_path):
    with open(answer_path, "r") as f:
        results = json.load(f)

    correctness = []
    for index, sample in enumerate(results):
        label_list = sample['label']
        answer = sample['answer']

        num_label = str(label_list[0])

        if num_label in answer or label_list[1].lower() in answer.lower():
            sample['is_hallucination'] = "no"
            correctness.append(1)
        else:
            sample['is_hallucination'] = "yes"
            correctness.append(0)

        results[index] = sample

    eval_dir = os.path.abspath(os.path.join(os.path.dirname(answer_path), "..", "eval"))
    os.makedirs(eval_dir, exist_ok=True)

    eval_filename = os.path.basename(answer_path)
    eval_path = os.path.join(eval_dir, eval_filename)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return correctness

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text

if __name__ == "__main__":
    tokenizer, model = load_qwen()

    input_dir = "./PipelineTest/results"
    output_dir = "./PipelineTest/correctness"
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for filename in tqdm(json_files):
        answer_path = os.path.join(input_dir, filename)
        output_file = filename.replace(".json", "_correctness.pt")
        output_path = os.path.join(output_dir, output_file)

        print(f"Evaluating {filename}...")

        if "sciqa" in filename or "commonsenseqa" in filename:
            correctness = compute_correctness_sciqa(answer_path)
        else:
            correctness = compute_correctness_truthfulqa(answer_path, model, tokenizer)

        correctness = torch.tensor(correctness)
        torch.save(correctness, output_path)
        print(f"  -> Accuracy: {correctness.float().mean().item():.2%} | Saved to {output_path}")