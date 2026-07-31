import json
import re
import string
from collections import Counter

# =========================================================================
# HEURISTIQUES DE NORMALISATION ET RECOUVREMENT LEXICAL (SQuAD / TriviaQA)
# =========================================================================

def normalize_answer(s):
    """
    Normalise le texte en enlevant la ponctuation, les articles (a, an, the)
    et les espaces superflus, conformément aux standards TriviaQA.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(prediction, ground_truth):
    """
    Calcule le score F1 au niveau des tokens entre la prédiction et la vérité terrain.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
        
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def check_factuality(item, f1_threshold=0.5):
    """
    Détermine si la réponse est une hallucination ("yes") ou correcte ("no").
    Applique d'abord l'Exact Match/inclusion de sous-chaîne, puis le score F1.
    """
    prediction = item.get("answer", "")
    gold_labels = item.get("label", [])
    
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return "yes"  # Génération vide = hallucination par défaut

    for gt in gold_labels:
        norm_gt = normalize_answer(gt)
        if not norm_gt:
            continue
            
        # 1. Exact Match ou Inclusion (L'entité cible est incluse dans la phrase générée)
        if norm_gt == norm_pred or norm_gt in norm_pred:
            return "no"
            
        # 2. Seuil de recouvrement lexical (Token-level F1 >= 0.5)
        if compute_token_f1(prediction, gt) >= f1_threshold:
            return "no"
            
    return "yes"


# =========================================================================
# PIPELINE DE TRAITEMENT DU FICHIER JSON
# =========================================================================

def relabel_json_file(input_path, output_path):
    print(f"[INFO] Lecture du fichier : {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    n_hallucinations = 0
    n_correct = 0

    print("[INFO] Application de la labellisation lexicale...")
    for item in data:
        # Détermination du label factuel
        label_decision = check_factuality(item, f1_threshold=0.5)
        
        # Modification des champs tout en préservant le reste du JSON
        item["is_hallucination"] = label_decision
        item["is_hallucinated"] = label_decision  # Ajouté pour correspondre à ta demande
        
        if label_decision == "yes":
            n_hallucinations += 1
        else:
            n_correct += 1

    # Sauvegarde des résultats
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCÈS] Traitement terminé et enregistré dans : {output_path}")
    print(f" ➔ Réponses valides (no) : {n_correct}")
    print(f" ➔ Hallucinations détectées (yes) : {n_hallucinations}")


if __name__ == "__main__":
    # input_file = "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.json"
    # output_file = "PipelineTest/res/eval/results_triviaqa_llada_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_relabelled.json"
    input_file = "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60.json"
    output_file = "PipelineTest/res/eval/results_triviaqa_dream_16steps_32tokens_triviaqa_2100samples_mixedtemp40-60_relabelled.json"

    relabel_json_file(input_file, output_file)