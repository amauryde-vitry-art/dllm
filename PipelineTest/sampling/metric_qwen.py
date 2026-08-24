from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import os
import re
from tqdm import tqdm
import time
import OpenAI



def load_qwen(device=None):
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
    
    tokenizer.padding_side = "left" 
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device is None:
        device_map = "auto"
    else:
        device_map = {"": device}
        
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-9B',
        device_map=device_map,
        dtype=torch.float16,
        trust_remote_code=True,
    ).eval()
    return tokenizer, model

from jinja2 import Template


def get_prompt(sample):
  # 1. Extraction et nettoyage des données de sample
  query = sample.get("question", "").strip()
  generated_answer = sample.get("answer", "").strip()

  # Sépare le premier label (expected_answer) et les suivants (answer_aliases)
  labels = sample.get("label", [])
  flat_labels = []

  if isinstance(labels, str):
    flat_labels = [labels.strip()]
  elif isinstance(labels, list):
    for entry in labels:
      if isinstance(entry, str):
        flat_labels.append(entry.strip())
      elif isinstance(entry, list) and len(entry) > 0:
        flat_labels.append(str(entry[0]).strip())

  expected_answer = flat_labels[0] if flat_labels else ""
  answer_aliases = flat_labels[1:] if len(flat_labels) > 1 else []

  # 2. Votre template TRUE_FALSE_PROMPT exact (syntaxe Jinja2)
  TRUE_FALSE_PROMPT = """
You are an expert evaluator tasked with determining if two answers convey compatible information. Your task is to make a binary True/False judgment on whether the answers are SEMANTICALLY COMPATIBLE.

Query:
{{ query }}

Expected Answer:
{{ expected_answer }}
{% if answer_aliases %}
Answer Aliases (Additional Correct Answers):
{% for alias in answer_aliases %}
- {{ alias }}
{% endfor %}
{% endif %}

Generated Answer:
{{ generated_answer }}

CRITICAL INSTRUCTIONS:
1. FIRST, perform a simple VERBATIM TEXT COMPARISON:
   - If the generated answer is IDENTICAL (exact same text) to EITHER the expected answer OR ANY of the answer aliases, your judgment MUST be TRUE
   - If not identical to any of them, proceed to semantic comparison

2. For SEMANTIC COMPARISON, use these MANDATORY RULES:
   - Judge "True" if the generated answer matches the SEMANTIC MEANING of EITHER the expected answer OR ANY of the answer aliases
   - Judge "True" WHENEVER the general meaning or core concept is the same as either the expected answer or any alias
   - Judge "True" if one answer is GENERAL and one is SPECIFIC about the same thing
   - Judge "True" if one answer names a CATEGORY (e.g., "missionaries") and the other provides SPECIFIC INSTANCES of that category (e.g., "Augustine was sent by Pope Gregory")
   - Judge "True" if one answer gives a BRIEF fact and the other ELABORATES with more details
   - Judge "True" if one answer is more detailed but does NOT contradict the other
   - Judge "False" ONLY if the answers directly CONTRADICT all of the expected answer and all aliases, or discuss ENTIRELY different topics

3. EXTREMELY IMPORTANT RULES ABOUT SPECIFICITY:
   - When one answer is general and one is specific → TRUE
   - When one uses a category term and one gives examples → TRUE
   - When one gives "who/what" and the other adds "when/where/how/why" → TRUE
   - When one gives a person's role and the other gives their name → TRUE
   - When one refers to a group and the other names individuals → TRUE

4. Always check if the specific answer is an INSTANCE or EXAMPLE of the general answer
   - If it is, the judgment MUST be TRUE regardless of how detailed the specific answer is

5. The query is provided ONLY for context - do NOT use it in your judgment

6. IMPORTANT: The generated answer should be considered TRUE if it matches EITHER the expected answer OR ANY of the answer aliases in meaning

FINAL CHECK BEFORE SUBMITTING:
- If the generated answer could reasonably be considered matching ANY of the expected answer or aliases → TRUE
- If after reading all answers, they feel like they're talking about the same basic concept → TRUE
- If you think "the generated answer is not contradicting the expected answer or any of its aliases" → TRUE

Your response MUST follow this format:
{
  "judgment": true/false,
  "explanation": "One clear sentence explaining why the answers are compatible or contradictory."
}
"""

  # 3. Rendu Jinja2 : Injecte les vraies variables dans le prompt
  rendered_prompt = Template(TRUE_FALSE_PROMPT).render(
      query=query,
      expected_answer=expected_answer,
      answer_aliases=answer_aliases,
      generated_answer=generated_answer,
  )

  # 4. Construction des messages pour le chat template
  messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator. Do NOT generate any reasoning, thinking process, or preamble. "
                "Respond ONLY with a JSON object."
            ),
        },
        {"role": "user", "content": rendered_prompt},
    ]

  return messages


import gc
import json
import os
import re
import time
import torch

def compute_correctness_truthfulqa(
    answer_path,
    client: OpenAI,
    model_name: str,
    max_concurrent_requests: int = 16,
    max_new_tokens: int = 128,
    max_retries: int = 3,
):
    with open(answer_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError(f"Empty JSON file: {answer_path}")
    try:
        results = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {answer_path}: {e}") from e

    total = len(results)
    print(
        f"[eval] Starting evaluation of {total} samples "
        f"(max_concurrent_requests={max_concurrent_requests})",
        flush=True,
    )
    t0 = time.time()

    correctness = [0] * total

    def _judge_one(idx):
        sample = results[idx]
        messages = get_prompt(sample)

        last_err = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},

                )
                decoded_str = response.choices[0].message.content.strip()
                return idx, decoded_str, None
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))  # backoff simple
        return idx, "", last_err

    done = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        futures = [executor.submit(_judge_one, idx) for idx in range(total)]

        from concurrent.futures import as_completed

        for future in as_completed(futures):
            idx, decoded_str, err = future.result()

            if err is not None:
                # Requête définitivement échouée après retries -> traité comme "unclear"
                print(f"[eval] WARNING sample {idx} failed after retries: {err}", flush=True)
                results[idx]["is_hallucination"] = "unclear"
                results[idx]["explanation"] = f"eval_error: {err}"
                correctness[idx] = 0
            else:
                output_dict = extract_answer(decoded_str)
                judgment = output_dict.get("judgment", "unclear")
                explanation = output_dict.get("explanation", "")

                if judgment == "false":
                    results[idx]["is_hallucination"] = "yes"
                    results[idx]["explanation"] = explanation
                    correctness[idx] = 0
                elif judgment == "true":
                    results[idx]["is_hallucination"] = "no"
                    results[idx]["explanation"] = explanation
                    correctness[idx] = 1
                else:
                    results[idx]["is_hallucination"] = "unclear"
                    results[idx]["explanation"] = explanation
                    correctness[idx] = 0

            done += 1
            if done % max_concurrent_requests == 0 or done == total:
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                acc_so_far = sum(correctness[:done]) / done if done > 0 else 0
                print(
                    f"[eval] {done}/{total} ({done*100//total}%) | "
                    f"acc~={acc_so_far:.2%} | {elapsed:.1f}s elapsed | ETA {eta:.0f}s",
                    flush=True,
                )

    print(
        f"[eval] Done. Final accuracy: {sum(correctness)/total:.2%} in"
        f" {time.time()-t0:.1f}s",
        flush=True,
    )

    eval_dir = os.path.abspath(
        os.path.join(os.path.dirname(answer_path), "..", "eval")
    )
    os.makedirs(eval_dir, exist_ok=True)

    eval_filename = os.path.basename(answer_path)
    eval_path = os.path.join(eval_dir, eval_filename)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return correctness





def save_eval_file(answer_path, results):
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(answer_path), "..", "eval"))
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, os.path.basename(answer_path)), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

import json
import re

def extract_answer(text: str) -> dict:
  """Extrait le jugement et l'explication depuis le texte de l'évaluateur.

  Si un JSON valide est trouvé, ses champs sont renvoyés.
  Sinon, tout le texte généré est placé dans 'explanation' avec un jugement
  'unclear'.
  """
  if not text or not isinstance(text, str):
    return {
        "judgment": "unclear",
        "explanation": "Empty or invalid input text",
    }

  # Recherche du premier bloc JSON { ... } dans le texte
  json_match = re.search(r"\{.*\}", text, re.DOTALL)
  if json_match:
    try:
      response_dict = json.loads(json_match.group(0))

      if isinstance(response_dict, dict) and "judgment" in response_dict:
        raw_judgment = response_dict["judgment"]

        # Normalisation de judgment (booléen ou string -> "true" / "false")
        if isinstance(raw_judgment, bool):
          judgment_str = "true" if raw_judgment else "false"
        elif isinstance(raw_judgment, str):
          judgment_str = raw_judgment.strip().lower()
          if judgment_str not in ["true", "false"]:
            judgment_str = "unclear"
        else:
          judgment_str = "unclear"

        return {
            "judgment": judgment_str,
            "explanation": response_dict.get(
                "explanation", "No explanation provided"
            ),
        }
    except json.JSONDecodeError:
      pass

  # Pas de JSON valide trouvé : renvoie tout le texte généré
  return {"judgment": "unclear", "explanation": text.strip()}

# =========================================================================
# MAIN
# =========================================================================



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

        print(f"\nEvaluating {filename}...")

        correctness = compute_correctness_truthfulqa(answer_path, model, tokenizer)

        correctness = torch.tensor(correctness)
        torch.save(correctness, output_path)
        print(f"  -> Final Accuracy: {correctness.float().mean().item():.2%} | Saved to {output_path}")