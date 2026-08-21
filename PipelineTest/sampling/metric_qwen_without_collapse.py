from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import os
import re
from collections import Counter
from tqdm import tqdm
import time

# =========================================================================
# MODE COLLAPSE DETECTION (heuristic, independent of Qwen)
# =========================================================================

def _char_run_ratio(text, min_run=8):
    if not text:
        return 0.0
    total_run_chars = 0
    run_char = None
    run_len = 0
    for ch in text:
        if ch == run_char:
            run_len += 1
        else:
            if run_char is not None and run_len >= min_run:
                total_run_chars += run_len
            run_char = ch
            run_len = 1
    if run_char is not None and run_len >= min_run:
        total_run_chars += run_len
    return total_run_chars / max(len(text), 1)


def _token_repetition_ratio(tokens):
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / len(tokens))


def _max_ngram_repetition_ratio(tokens, n=3, min_repeats=4):
    if len(tokens) < n * min_repeats:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0
    most_common_ngram, count = counts.most_common(1)[0]
    if count < min_repeats:
        return 0.0
    covered = count * n
    return min(covered / len(tokens), 1.0)


def _has_garbled_words(text, max_consonant_run=5, min_word_len=5):
    """
    Detect garbled/corrupted words typical of diffusion model failures.
    E.g.: "Burnching", "liquein", "Volicket", "isuncher", "Professorumbledore"
    
    Returns (is_garbled: bool, garbled_words: list[str])
    """
    import string
    vowels = set("aeiouyAEIOUY")
    words = text.split()
    garbled = []
    for w in words:
        # Strip punctuation for analysis
        clean = w.strip(string.punctuation)
        if len(clean) < min_word_len:
            continue
        
        # Skip words that are ALL CAPS (acronyms) or contain hyphens (compound words)
        if clean.isupper() or '-' in w:
            continue
        
        # Check for merged/concatenated words (unexpected lowercase after uppercase mid-word)
        # e.g. "Professorumbledore", "GeorgianGeorgian"
        # But NOT normal CamelCase like "McDonald" or proper nouns like "Wordsworth"
        inner = clean[1:]
        if re.search(r'[a-z][A-Z][a-z]', clean):
            # Has a lowercase-Uppercase-lowercase transition mid-word
            # Check it's not a normal pattern (Mc/Mac prefix)
            if not re.match(r'^(Mc|Mac|De|Le|La|Van|Von)', clean):
                garbled.append(w)
                continue
        
        # Check for long consonant runs (>=5 consonants with no vowel)
        lower = clean.lower()
        consonant_run = 0
        max_run = 0
        for ch in lower:
            if ch.isalpha() and ch not in vowels:
                consonant_run += 1
                max_run = max(max_run, consonant_run)
            else:
                consonant_run = 0
        if max_run >= max_consonant_run:
            garbled.append(w)
            continue
    
    return len(garbled) > 0, garbled


def _is_very_short_garbled(text):
    """
    Detect very short answers that are single garbled tokens.
    E.g.: "D S", "Lhena", "Chovar", "liquein", "Volicket"
    """
    text = text.strip()
    words = text.split()
    
    # Single letter answers (but not numbers like "2")
    if len(text) <= 2 and text.isalpha():
        return True
    
    # 1-2 word answer, total length ≤ 10, and doesn't look like a real short answer
    # Real short answers: "Sony", "Mars", "Gold", "Cuba", etc.
    if len(words) <= 2 and len(text) <= 15:
        # Check if ALL words have unusual character patterns
        vowels = set("aeiouy")
        for w in words:
            clean = w.strip(".,!?\"'").lower()
            if len(clean) < 3:
                continue
            # Check vowel ratio - garbled words often have very few vowels
            n_vowels = sum(1 for c in clean if c in vowels)
            vowel_ratio = n_vowels / len(clean)
            if vowel_ratio < 0.15 and len(clean) >= 4:
                return True
    
    return False


def detect_mode_collapse(
    text,
    char_run_threshold=0.3,
    char_run_min_len=6,
    token_repetition_threshold=0.6,
    ngram_repetition_threshold=0.5,
    min_tokens_for_judgment=3,
):
    text = (text or "").strip()
    reason = None
    is_collapse = False

    if not text:
        return {
            "is_mode_collapse": True, "collapse_reason": "empty_output",
            "char_run_ratio": 0.0, "token_repetition_ratio": 0.0, "ngram_repetition_ratio": 0.0
        }

    # Règle impérative TDGNet Appendix E.2 : Troncatures sévères / Traces incomplètes de diffusion
    # Détecte une lettre unique isolée (ex: "Z", "B") ou une coupure de mot brutale en fin de chaîne (ex: "was sc.")
    if len(text) == 1 and text.isalpha():
        return {
            "is_mode_collapse": True, "collapse_reason": "truncated_single_letter",
            "char_run_ratio": 0.0, "token_repetition_ratio": 0.0, "ngram_repetition_ratio": 0.0
        }
    
    if  text.endswith("..") or re.search(r'\b(the|was|by|of|an)\s+[a-z]{2,4}$', text, re.IGNORECASE):
        # Filtre les faux positifs évidents comme les nombres ou symboles chimiques complets (ex: O+)
        if not re.search(r'^\d+$', text) and not "+" in text:
            is_collapse = True
            reason = "abrupt_truncation_prefix"

    char_run_ratio = _char_run_ratio(text, min_run=char_run_min_len)
    tokens = text.split()
    token_repetition_ratio = _token_repetition_ratio(tokens)
    ngram_repetition_ratio = _max_ngram_repetition_ratio(tokens, n=3, min_repeats=3)

    if not is_collapse:
        if char_run_ratio >= char_run_threshold:
            is_collapse = True
            reason = f"char_run_ratio={char_run_ratio:.2f}"
        elif len(tokens) >= min_tokens_for_judgment and token_repetition_ratio >= token_repetition_threshold:
            is_collapse = True
            reason = f"token_repetition_ratio={token_repetition_ratio:.2f}"
        elif ngram_repetition_ratio >= ngram_repetition_threshold:
            is_collapse = True
            reason = f"ngram_repetition_ratio={ngram_repetition_ratio:.2f}"

    # Garbled word detection (diffusion token corruption)
    if not is_collapse:
        has_garbled, garbled_words = _has_garbled_words(text)
        if has_garbled:
            garbled_ratio = len(garbled_words) / max(len(tokens), 1)
            if garbled_ratio >= 0.3 or (len(tokens) <= 3 and len(garbled_words) >= 1):
                is_collapse = True
                reason = f"garbled_words={garbled_words[:3]}"

    # Very short garbled answer detection
    if not is_collapse:
        if _is_very_short_garbled(text):
            is_collapse = True
            reason = "short_garbled"

    return {
        "is_mode_collapse": is_collapse,
        "collapse_reason": reason,
        "char_run_ratio": round(char_run_ratio, 4),
        "token_repetition_ratio": round(token_repetition_ratio, 4),
        "ngram_repetition_ratio": round(ngram_repetition_ratio, 4),
    }


# =========================================================================
# QWEN-BASED PROMPTING (FACTUALITY & STRUCTURAL DEGRADATION)
# =========================================================================

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

def get_collapse_prompt(sample):
    messages = [
        {'role': 'system', 'content': 'You are an expert text quality judge.'},
        {'role': 'user', 'content': '''Your task is to determine whether a generated text suffers from MODE COLLAPSE or structural degeneration. This is NOT about factual correctness -- only about whether the text is well-formed.

A text has mode collapse if ANY of the following is true:
1. It is empty or contains only whitespace/punctuation.
2. It contains garbled, nonsensical, or truncated words (e.g. "Burnching", "liquein", "Camongo", "Poca-Cola").
3. It repeats the same word, phrase, or n-gram excessively (e.g. "the the the the" or "Euronext Euronext Euronext").
4. It is largely unintelligible or unreadable as natural language.
5. It contains obvious token-level corruption (random characters inserted, words cut off mid-syllable).

A text does NOT have mode collapse if:
- It is a coherent, readable sentence, even if factually wrong.
- It has minor typos but is still understandable.
- It is short but well-formed (e.g. a single correct word as an answer).

Please answer "yes" if the text has mode collapse, or "no" if it is well-formed. Put your answer in <answer> </answer>.'''}
    ]
    messages.append({'role': 'assistant',
                     'content': 'I understand. Please provide the text to judge.'})
    messages.append({'role': 'user', 'content': f'The bot\'s answer to a question was:\n"{sample.get("answer", "")}"\n\nDoes this text suffer from mode collapse? Answer "yes" or "no". Dont show thinking and put your answer in <answer> </answer>.'})
    return messages



# =========================================================================
# CORE CORE PIPELINE DETECTION (UNIFIED FOR ALL DATASETS)
# =========================================================================

def run_mode_collapse_pipeline(results, model, tokenizer, batch_size=16):
    """
    Applique la détection d'effondrement de mode de bout en bout (Heuristique + LLM).
    Optimisation : Évite d'appeler le LLM si le filtre heuristique a déjà flaggé le collapse.
    """
    total = len(results)
    
    # 1. Filtre Heuristique initial
    for sample in results:
        h_res = detect_mode_collapse(sample.get("answer", ""))
        sample["is_mode_collapse_heuristic"] = "yes" if h_res["is_mode_collapse"] else "no"
        sample["collapse_reason"] = h_res["collapse_reason"]
        sample["char_run_ratio"] = h_res["char_run_ratio"]
        sample["token_repetition_ratio"] = h_res["token_repetition_ratio"]
        sample["ngram_repetition_ratio"] = h_res["ngram_repetition_ratio"]
        # On initialise par défaut avec la valeur heuristique
        sample["is_mode_collapse"] = sample["is_mode_collapse_heuristic"]

    # 2. Validation par LLM (uniquement sur les éléments ambigus non encore flaggés)
    llm_indices = [i for i in range(total) if results[i]["is_mode_collapse_heuristic"] == "no"]
    
    if llm_indices:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        collapse_prompts = []
        for i in llm_indices:
            collapse_prompts.append(tokenizer.apply_chat_template(get_collapse_prompt(results[i]), add_generation_prompt=True, tokenize=False))

        for batch_start in range(0, len(llm_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(llm_indices))
            batch_prompts = collapse_prompts[batch_start:batch_end]
            batch_indices = llm_indices[batch_start:batch_end]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=32)

            input_len = inputs["input_ids"].shape[1]
            for i, idx in enumerate(batch_indices):
                generated_tokens = output_ids[i][input_len:]
                output_text = extract_answer(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower())
                
                if "yes" in output_text:
                    results[idx]["is_mode_collapse"] = "yes"
                    if results[idx]["collapse_reason"] is None:
                        results[idx]["collapse_reason"] = "llm_confirmed_collapse"
                else:
                    results[idx]["is_mode_collapse"] = "no"

        tokenizer.padding_side = original_padding_side

    return results
import gc
import json
import os
import re
import time
import torch


def compute_correctness_truthfulqa(
    answer_path, model, tokenizer, batch_size=4
):  # Réduit à 4 (ou 8) pour éviter l'OOM
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
      f"[eval] Starting evaluation of {total} samples (batch_size={batch_size})",
      flush=True,
  )
  t0 = time.time()

  prompts = []
  for sample in results:
    messages = get_prompt(sample)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    prompts.append(prompt)

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
          max_new_tokens=128,  # 128 tokens suffisent largement pour le JSON
          pad_token_id=tokenizer.pad_token_id,
      )

    input_len = inputs["input_ids"].shape[1]
    for i, idx in enumerate(range(batch_start, batch_end)):
      generated_tokens = output_ids[i][input_len:]
      decoded_str = tokenizer.decode(
          generated_tokens, skip_special_tokens=True
      ).strip()

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

    # --- NETTOYAGE OBLIGATOIRE DE LA VRAM ---
    del inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()

    # Progress log
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

  tokenizer.padding_side = original_padding_side

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



def compute_correctness_sciqa(answer_path, model, tokenizer, batch_size=16):
    """
    Vérification de SciQA/CommonsenseQA augmentée par notre pipeline de collapse synchrone.
    """
    with open(answer_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    print(f"[eval] Processing SciQA/Multiple-Choice style: {total} samples", flush=True)

    # Injection du même niveau de détection de collapse que TruthfulQA
    results = run_mode_collapse_pipeline(results, model, tokenizer, batch_size)
    correctness = []

    for sample in results:
        label_list = sample['label']
        answer = sample['answer']
        num_label = str(label_list[0])

        if sample["is_mode_collapse"] == "yes":
            sample['is_hallucination'] = "yes"  # Aligné avec la taxonomie sémantique
            correctness.append(0)
        elif num_label in answer or label_list[1].lower() in answer.lower():
            sample['is_hallucination'] = "no"
            correctness.append(1)
        else:
            sample['is_hallucination'] = "yes"
            correctness.append(0)

    save_eval_file(answer_path, results)
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
# EXACT-MATCH / TOKEN-OVERLAP LABELING (TDGNet-style, no LLM judge)
# =========================================================================

def _normalize_answer(s):
    """Lower, strip, remove articles/punctuation for fair comparison."""
    import string
    s = s.lower().strip()
    # Remove articles
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _token_f1(prediction_tokens, ground_truth_tokens):
    """Compute token-level F1 between prediction and ground truth."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_correctness_exact_match(answer_path, model=None, tokenizer=None, batch_size=16, f1_threshold=0.5):
    """
    Label hallucination using exact match / token-overlap (TDGNet protocol).
    If model+tokenizer are provided, uses Qwen for collapse detection (same as qwen mode).
    Otherwise falls back to heuristic-only collapse detection.
    """
    with open(answer_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    print(f"[exact-match] Processing {total} samples", flush=True)

    # Collapse detection: Qwen if available, heuristic otherwise
    if model is not None and tokenizer is not None:
        print(f"[exact-match] Using Qwen collapse detection", flush=True)
        results = run_mode_collapse_pipeline(results, model, tokenizer, batch_size)
    else:
        print(f"[exact-match] Using heuristic-only collapse detection (no Qwen)", flush=True)
        for sample in results:
            h_res = detect_mode_collapse(sample.get("answer", ""))
            sample["is_mode_collapse_heuristic"] = "yes" if h_res["is_mode_collapse"] else "no"
            sample["is_mode_collapse"] = sample["is_mode_collapse_heuristic"]
            sample["collapse_reason"] = h_res["collapse_reason"]

    correctness = []
    for sample in results:
        if sample.get("is_mode_collapse") == "yes":
            sample["is_hallucination"] = "yes"
            correctness.append(0)
            continue

        answer = sample.get("answer", "")
        labels = sample.get("label", [])

        norm_answer = _normalize_answer(answer)
        answer_tokens = norm_answer.split()

        is_correct = False
        for alias in labels:
            if isinstance(alias, list):
                alias = alias[0] if alias else ""
            norm_alias = _normalize_answer(str(alias))

            if norm_answer == norm_alias or norm_alias in norm_answer:
                is_correct = True
                break

            alias_tokens = norm_alias.split()
            if alias_tokens and answer_tokens:
                f1 = _token_f1(answer_tokens, alias_tokens)
                if f1 >= f1_threshold:
                    is_correct = True
                    break

        sample["is_hallucination"] = "no" if is_correct else "yes"
        correctness.append(1 if is_correct else 0)

    save_eval_file(answer_path, results)

    acc = sum(correctness) / len(correctness) if correctness else 0
    n_collapse = sum(1 for s in results if s.get("is_mode_collapse") == "yes")
    print(f"[exact-match] accuracy={acc:.2%} ({sum(correctness)}/{len(correctness)}) | collapse={n_collapse}")

    return correctness


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

        if "sciqa" in filename or "commonsenseqa" in filename:
            correctness = compute_correctness_sciqa(answer_path, model, tokenizer)
        else:
            correctness = compute_correctness_truthfulqa(answer_path, model, tokenizer)

        correctness = torch.tensor(correctness)
        torch.save(correctness, output_path)
        print(f"  -> Final Accuracy: {correctness.float().mean().item():.2%} | Saved to {output_path}")