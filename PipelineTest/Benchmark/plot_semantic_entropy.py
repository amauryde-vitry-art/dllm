import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================================================================
# 1. CHARGEMENT DES DONNÉES
# =========================================================================
file_path = "PipelineTest/Benchmark/eval/TDGnet/semantic_entropy_results_llada16_32tokens_2100samples_triviaqa_without_collapse_exact_matches.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

config_name = "llada16_32tokens_2100samples_triviaqa_without_collapse_exact_matches"
samples_data = data[config_name]["samples"]

all_keys = list(samples_data.keys())
y_scores = [samples_data[k]["semantic_entropy"] for k in all_keys]
y_true = [samples_data[k]["label_hallucination"] for k in all_keys]

# Separateur des valeurs pour l'histogramme global
entropy_values = np.array(y_scores)
label_values = np.array(y_true)
entropy_correct = entropy_values[label_values == 0]
entropy_halluc = entropy_values[label_values == 1]

# =========================================================================
# 2. SPLIT ET EXTRACTION DES INDEX SELECTIONNES
# =========================================================================
# Sequential 80/20 split - same as all other benchmark scripts
split_point = int(len(all_keys) * 0.8)
keys_train = all_keys[:split_point]
keys_test = all_keys[split_point:]
X_train = y_scores[:split_point]
X_test = y_scores[split_point:]
y_train = y_true[:split_point]
y_test = y_true[split_point:]

# --- SECTION AFFICHAGE DES INDEX ---
print("\n" + "="*50)
print(f"  INDEX SÉLECTIONNÉS POUR LE TEST SPLIT (N={len(keys_test)})")
print("="*50)
print(keys_test)  # Affiche la liste brute des clés du test split

print("\n" + "="*50)
print(f"  INDEX SÉLECTIONNÉS POUR LE TRAIN SPLIT (N={len(keys_train)})")
print("="*50)
print(keys_train) # Affiche la liste brute des clés du train split
print("="*50 + "\n")


# =========================================================================
# 3. CALCUL DES MÉTRIQUES GLOBALES
# =========================================================================
auroc_score = roc_auc_score(y_test, X_test)
pr_auc_score = average_precision_score(y_test, X_test)

print(f"  AUROC Globale : {auroc_score:.4f}") 
print(f"  PR-AUC Globale : {pr_auc_score:.4f}")
print(f"{'='*40}")


# =========================================================================
# 4. HISTOGRAMME
# =========================================================================
fig, ax = plt.subplots(figsize=(15, 6))
bins = np.linspace(-0.1, max(entropy_values), 40)

ax.hist(entropy_correct, bins=bins, color='#2ecc71', alpha=0.7, 
        label='Correct (Label 0)', edgecolor='#27ae60', linewidth=0.5)

ax.hist(entropy_halluc, bins=bins, color='#e74c3c', alpha=0.7, 
        label='Hallucination (Label 1)', edgecolor='#c0392b', linewidth=0.5)

ax.set_title("Distribution de l'Entropie Sémantique par Classe", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Entropie Sémantique (NLI)", fontsize=12, labelpad=10)
ax.set_ylabel("Nombre d'échantillons (Count)", fontsize=12, labelpad=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=11, frameon=True)

plt.tight_layout()
output_image_path = "PipelineTest/Benchmark/eval/semantic_entropy_histogram.png"
plt.savefig(output_image_path, dpi=300)
print(f"[SUCCÈS] Histogramme sauvegardé avec succès dans : {output_image_path}")