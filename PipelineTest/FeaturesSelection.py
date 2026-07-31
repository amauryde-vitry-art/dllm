import pandas as pd
import numpy as np
from CreateMetrics import getTestSimpleFeatures, getAlphaBetaFromEntropyAvgStudy, getAlphaBetaFromEntropyAvgStudyv2
from sklearn.decomposition import PCA
import seaborn as sns

import os
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score

from PipelineTest.AnalyseResults import mergeOutputsList


RESULTS_DIR = "PipelineTest/res/results_LLaDa_64steps_64tokens_entropy"

outputpath = f"{RESULTS_DIR}/outputs_LLaDa_64steps_64tokens_entropy.pt"
eval_json_path = f"PipelineTest/res/eval/results_triviaqa_LLaDa_64steps_64tokens_entropy.json"

features, feature_names, labels = getTestSimpleFeatures(outputpath, eval_json_path, k_tokens= None)

# 1. Standardisation de tes variables (parfait, tu le faisais déjà)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 2. Ta Heatmap de corrélation (inchangée, elle fonctionne très bien avec des labels 0/1)
plt.figure(figsize=(30, 20))
X = np.concatenate([features, labels.reshape(-1, 1)], axis=1)
correl = np.corrcoef(X, rowvar=False)
all_names = feature_names + ["label"]
sns.heatmap(correl, cmap='viridis', cbar=True, annot=True, fmt=".2f", xticklabels=all_names, yticklabels=all_names)
plt.xlabel("Features and Label Index")
plt.ylabel("Features and Label Index")
plt.title("Heatmap of Features and Labels")
os.makedirs(f"{RESULTS_DIR}/FeaturesSelection/heatmaps", exist_ok=True)
plt.savefig(f"{RESULTS_DIR}/FeaturesSelection/heatmaps/heatmap_features_simple_labels.png")
plt.show()




def perform_pca(data):
  """Exécute l'analyse en composantes principales (ACP)."""
  pca = PCA()
  pca.fit(data)
  return pca

def calculate_variance_info(pca):
  """Calcule la variance expliquée et cumulée."""
  explained_variance = pca.explained_variance_ratio_
  cumulative_variance = np.cumsum(explained_variance)
  return explained_variance, cumulative_variance

def plot_scree(pca, explained_variance, cumulative_variance, save_path=None):
  """Génère le graphique de la variance expliquée (Scree plot)."""
  plt.figure(figsize=(10, 6))

  # Tracé de la variance cumulée
  plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b', label='Variance cumulée')

  # Barres pour la variance de chaque composant individuel
  plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, color='g', label='Variance individuelle')

  # Mises en forme
  plt.title('Variance expliquée en fonction du nombre de composants (ACP)', fontsize=14)
  plt.xlabel('Nombre de composants', fontsize=12)
  plt.ylabel('Proportion de Variance Expliquée', fontsize=12)
  plt.axhline(y=0.95, color='r', linestyle=':', label='Seuil 95%')
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.legend(loc='best')

  if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Graphique sauvegardé sous : {save_path}")

  plt.show()

def plot_correlation_circle(pca, feature_names, save_path=None):
  """Génère le cercle des corrélations."""
  loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

  fig, ax = plt.subplots(figsize=(10, 10))

  # Cercle unité
  circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
  ax.add_artist(circle)

  # Axes
  ax.axhline(0, color='black', linewidth=0.5)
  ax.axvline(0, color='black', linewidth=0.5)

  # Vecteurs et labels pour chaque variable
  for i in range(loadings.shape[0]):
    x, y = loadings[i, 0], loadings[i, 1]
    ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='red', ec='red', length_includes_head=True)
    ax.text(x * 1.1, y * 1.1, feature_names[i], color='darkgreen', ha='center', va='center')

  # Mises en forme
  plt.title('Cercle des corrélations des variables (ACP₁ vs ACP₂)', fontsize=14)
  plt.xlabel('Composante Principale 1 (ACP₁)', fontsize=12)
  plt.ylabel('Composante Principale 2 (ACP₂)', fontsize=12)
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.xlim(-1.2, 1.2)
  plt.ylim(-1.2, 1.2)
  ax.set_aspect('equal') # Pour que le cercle soit bien rond

  if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Graphique sauvegardé sous : {save_path}")

  plt.show()

# --- Exécution principale ---

# 1. Analyse ACP
pca = perform_pca(features)
explained_variance, cumulative_variance = calculate_variance_info(pca)

# 2. Graphique de la variance expliquée
scree_plot_path = os.path.join(RESULTS_DIR, "FeaturesSelection/Plots/variance_expliquee_pca.png")
plot_scree(pca, explained_variance, cumulative_variance, save_path=scree_plot_path)

# 3. Cercle des corrélations
correlation_circle_path = os.path.join(RESULTS_DIR, "FeaturesSelection/Plots/cercle_correlations_pca.png")
plot_correlation_circle(pca, feature_names, save_path=correlation_circle_path)






lasso_cls = LogisticRegressionCV(
    Cs=[ 0.1], # Grille personnalisée et moins agressive
    penalty="l1", 
    solver="saga", 
    scoring="average_precision",
    n_jobs=4, 
    random_state=42,
    max_iter = 10000
)


lasso_cls.fit(features, labels)

# En classification binaire, les coefficients sont stockés dans lasso_cls.coef_[0]
coef_df = pd.DataFrame({
    "Variable": feature_names,
    "Coefficient": lasso_cls.coef_[0]
})

# Une variable est sélectionnée si son coefficient n'est pas égal à 0
coef_df["Selectionnee"] = coef_df["Coefficient"] != 0

print("=== Résultat de la sélection L1 (Classification) ===")
print(coef_df.sort_values(by="Coefficient", key=abs, ascending=False))

# 4. Filtrage de ton tableau NumPy 'features'
variables_gardees = coef_df[coef_df["Selectionnee"]]["Variable"].tolist()
print(f"\nNombre de variables avant sélection L1 : {features.shape[1]}")

print(f"Variables conservées : {variables_gardees}")


pr_auc = average_precision_score(labels, lasso_cls.predict_proba(features)[:, 1])
roc_auc = roc_auc_score(labels, lasso_cls.predict_proba(features)[:, 1])

print("PR AUC Score of the L1 Logistic Regression model on the training data: {:.4f}".format(pr_auc))
print("ROC AUC Score of the L1 Logistic Regression model on the training data: {:.4f}".format(roc_auc))

selection_results = {
  "features_total": int(features.shape[1]),
  "features_selectionnees": int(len(variables_gardees)),
  "variables_conservees": variables_gardees,
  "coefficients": [
    {
      "Variable": row["Variable"],
      "Coefficient": float(row["Coefficient"]),
      "Selectionnee": bool(row["Selectionnee"]),
    }
    for _, row in coef_df.iterrows()
  ],
  "metrics": {
    "pr_auc_train": float(pr_auc),
    "roc_auc_train": float(roc_auc),
  },
}

json_save_path = os.path.join(RESULTS_DIR, "FeaturesSelection", "lasso_selection_results.json")
os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
with open(json_save_path, "w", encoding="utf-8") as f:
  json.dump(selection_results, f, ensure_ascii=False, indent=2)

print(f"Résultats JSON sauvegardés dans : {json_save_path}")






# outputs = mergeOutputsList(outputpath)
# alpha, beta = getAlphaBetaFromEntropyAvgStudyv2(outputs)
# correl_alpha_label = []
# correl_beta_label = []  
# for k in range(1, 65):
#     corr_alpha_label = np.corrcoef(alpha[k], labels)
#     corr_beta_label = np.corrcoef(beta[k], labels)
#     print(f"Corrélation entre Alpha et les Labels pour k={k}: {corr_alpha_label[0, 1]:.4f}")
#     print(f"Corrélation entre Beta et les Labels pour k={k}: {corr_beta_label[0, 1]:.4f}")
#     correl_alpha_label.append(corr_alpha_label[0, 1])
#     correl_beta_label.append(corr_beta_label[0, 1])
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 65), correl_alpha_label, marker='o', label='Corrélation Alpha-Label')
# plt.plot(range(1, 65), correl_beta_label, marker='o', label='Corrélation Beta-Label')
# plt.title('Corrélation entre Alpha/Beta et les Labels')
# plt.xlabel('k (Nombre de clusters)')
# plt.ylabel('Corrélation')
# plt.axhline(0, color='gray', linestyle='--')
# plt.legend()
# plt.grid()
# plt.savefig(f"{RESULTS_DIR}/FeaturesSelection/heatmaps/correlation_alpha_beta_labels.png")
# plt.show()




# roc_auc_scores = []
# pr_auc_scores = []
# for i in range(1, 65):
#   features = getTestSimpleFeatures(outputpath, k_tokens=i)[0]
#   lasso_cls.fit(features, labels)
#   coef_df = pd.DataFrame({
#       "Variable": feature_names,
#       "Coefficient": lasso_cls.coef_[0]
#   })
#   roc_auc_scores.append(roc_auc_score(labels, lasso_cls.predict_proba(features)[:, 1]))
#   pr_auc_scores.append(average_precision_score(labels, lasso_cls.predict_proba(features)[:, 1]))

# plt.figure(figsize=(15, 6))
# plt.plot(range(1, 65), roc_auc_scores, marker='o', label='ROC AUC Score')
# plt.plot(range(1, 65), pr_auc_scores, marker='o', label='PR AUC Score')
# plt.title('Performance du modèle L1 Logistic Regression en fonction de k')
# plt.xlabel('k (Nombre de clusters)')
# plt.ylabel('Score')
# plt.legend()
# plt.grid()
# plt.savefig(f"{RESULTS_DIR}/FeaturesSelection/Plots/performance_l1_logistic_regression.png")
# plt.show()
