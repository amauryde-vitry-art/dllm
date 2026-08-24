import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from PipelineTest.historic_scripts.CreateMetrics import getTestSimpleFeatures, getTestSimpleFeaturesEvaluation
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import os
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV



RESULTS_DIR = "PipelineTest/res/LLADA_64steps_64tokens_lowconf"

outputpath = f"{RESULTS_DIR}/outputs_LLADA_64steps_64tokens_lowconf.pt"
eval_json_path = f"PipelineTest/res/eval/results_triviaqa_LLADA_64steps_64tokens_lowconf.json"
features, feature_names, labels = getTestSimpleFeaturesEvaluation(outputpath, eval_json_path, k_tokens=None)

# 1. Standardisation de tes variables (parfait, tu le faisais déjà)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 2. Ta Heatmap de corrélation (inchangée, elle fonctionne très bien avec des labels 0/1)
plt.figure(figsize=(15, 10))
X = np.concatenate([features, labels.reshape(-1, 1)], axis=1)
correl = np.corrcoef(X, rowvar=False)
all_names = feature_names + ["label"]
sns.heatmap(correl, cmap='viridis', cbar=True, annot=True, fmt=".2f", xticklabels=all_names, yticklabels=all_names)
plt.xlabel("Features and Label Index")
plt.ylabel("Features and Label Index")
plt.title("Heatmap of Features and Labels")
os.makedirs(f"{RESULTS_DIR}/metricsResults/heatmaps", exist_ok=True)
plt.savefig(f"{RESULTS_DIR}/metricsResults/heatmaps/heatmap_features_simple_labels.png")
plt.show()

# pca = PCA()
# pca.fit(features)

# # 4. Calcul de la variance expliquée cumulée
# variance_expliquee = pca.explained_variance_ratio_
# variance_cumulee = np.cumsum(variance_expliquee)

# # 5. Création du graphique
# plt.figure(figsize=(10, 6))

# # Évolution de la variance cumulée
# plt.plot(range(1, len(variance_cumulee) + 1), variance_cumulee, marker='o', linestyle='--', color='b', label='Variance cumulée')

# # Barres pour la variance de chaque composant individuel
# plt.bar(range(1, len(variance_expliquee) + 1), variance_expliquee, alpha=0.5, color='g', label='Variance individuelle')

# # Mises en forme
# plt.title('Variance expliquée en fonction du nombre de composants (PCA)', fontsize=14)
# plt.xlabel('Nombre de composants (Variables PCA)', fontsize=12)
# plt.ylabel('Proportion de Variance Expliquée', fontsize=12)
# plt.axhline(y=0.95, color='r', linestyle=':', label='Seuil 95%') # Ligne repère à 95%
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(loc='best')
# os.makedirs(f"{RESULTS_DIR}/metricsResults/Plots", exist_ok=True)
# plt.savefig(f"{RESULTS_DIR}/metricsResults/Plots/variance_expliquee_pca.png")
# # Affichage du plot
# plt.show()

# pca = PCA(n_components=5)
# pca_features = pca.fit_transform(features)

# lasso_cls = LogisticRegressionCV(
#     Cs=[0.1, 0.5, 1], # Grille personnalisée et moins agressive
#     penalty="l1", 
#     solver="saga", 
#     scoring="average_precision",
#     n_jobs=4, 
#     random_state=42
# )


# lasso_cls.fit(features, labels)

# # En classification binaire, les coefficients sont stockés dans lasso_cls.coef_[0]
# coef_df = pd.DataFrame({
#     "Variable": feature_names,
#     "Coefficient": lasso_cls.coef_[0]
# })

# # Une variable est sélectionnée si son coefficient n'est pas égal à 0
# coef_df["Selectionnee"] = coef_df["Coefficient"] != 0

# print("=== Résultat de la sélection L1 (Classification) ===")
# print(coef_df.sort_values(by="Coefficient", key=abs, ascending=False))

# # 4. Filtrage de ton tableau NumPy 'features'
# variables_gardees = coef_df[coef_df["Selectionnee"]]["Variable"].tolist()
# print(f"\nNombre de variables avant sélection L1 : {features.shape[1]}")

# # Récupération des indices numériques des variables sélectionnées
# indices_gardes = [feature_names.index(var) for var in variables_gardees]
# features = features[:, indices_gardes]

# print(f"Nombre de variables après sélection L1 : {features.shape[1]}")
# print(f"Variables conservées : {variables_gardees}")

# # Optionnel : Mettre à jour ta liste de noms de variables pour la suite de ton pipeline
# feature_names = variables_gardees

def evalRF(features,  labels, save_path_plot, save_path_json, nbcomponents=None, plot=False):
    
    if nbcomponents is not None:
        pca = PCA(n_components=nbcomponents)
        features = pca.fit_transform(features)
    # Entraînement d'un classifieur Random Forest
    proportion_train = 0.8
    #shuffle les données avant de les séparer en train/test
    indices = np.arange(len(features))
    np.random.seed(42)
    np.random.shuffle(indices)
    features = features[indices]
    labels = np.array(labels)[indices]

    split_index = int(len(features) * proportion_train)
    X_train = features[:split_index]
    y_train = labels[:split_index]
    
    print("Training features shape:", X_train.shape)
    print("Training labels shape:", len(y_train))

    X_test = features[split_index:]
    y_test = labels[split_index:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = RandomForestClassifier(random_state=42,)
    # grid search pour trouver les meilleurs hyperparamètres

    # param_grid = {
    #     "n_estimators": [1500, 2500, 3500],
    #     "max_depth": [1, 2, 3, ],
    #     "max_features": [1, 2, "sqrt", None],
    #     "min_samples_split": [2, 5, 10, 20, 30],
    #     "min_samples_leaf": [1, 2, 4],
    #     "random_state": [42],
    # }
    param_grid = {
    "n_estimators": [1500, 2500, 3500],       
    "max_depth": [None, 3, 5, 10, 15],        
    "max_features": [2, 4, None],         
    "min_samples_split": [2, 5, 10],       
    "min_samples_leaf": [1, 2, 4],        
    "random_state": [42],
}


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        n_jobs=20,
        scoring=["roc_auc", "average_precision"],
        refit="average_precision",  
        return_train_score=True,
        verbose=1,
    )
    

    grid.fit(X_train, y_train)

    print("\n=== Best hyperparameters ===")
    for k, v in grid.best_params_.items():
        print(f"{k}: {v}")

    print(f"\nBest CV average precision: {grid.best_score_:.4f}")

    cv_results = grid.cv_results_
    best_idx = grid.best_index_
    print(
        "Best train PR AUC: "
        f"{cv_results['mean_train_average_precision'][best_idx]:.4f} +/- {cv_results['std_train_average_precision'][best_idx]:.4f}"
    )
    print(
        "Best val PR AUC:   "
        f"{cv_results['mean_test_average_precision'][best_idx]:.4f} +/- {cv_results['std_test_average_precision'][best_idx]:.4f}"
    )
    print(
        "Best train ROC AUC: "
        f"{cv_results['mean_train_roc_auc'][best_idx]:.4f} +/- {cv_results['std_train_roc_auc'][best_idx]:.4f}"
    )
    print(
        "Best val ROC AUC:   "
        f"{cv_results['mean_test_roc_auc'][best_idx]:.4f} +/- {cv_results['std_test_roc_auc'][best_idx]:.4f}"
    )

    test_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)
    pr_auc_baseline = y_test.sum() / len(y_test)  # prevalence = random classifier PR AUC
    print(f"Best ROC-AUC: {test_roc_auc:.4f}")
    print(f"Best PR-AUC:  {test_pr_auc:.4f} (baseline: {pr_auc_baseline:.4f})")

    # Sauvegarder les résultats en JSON

    dirname = os.path.dirname(save_path_json)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    results = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "mean_train_roc_auc": float(cv_results['mean_train_roc_auc'][best_idx]),
        "std_train_roc_auc": float(cv_results['std_train_roc_auc'][best_idx]),
        "mean_val_roc_auc": float(cv_results['mean_test_roc_auc'][best_idx]),
        "std_val_roc_auc": float(cv_results['std_test_roc_auc'][best_idx]),
        "mean_train_pr_auc": float(cv_results['mean_train_average_precision'][best_idx]),
        "std_train_pr_auc": float(cv_results['std_train_average_precision'][best_idx]),
        "mean_val_pr_auc": float(cv_results['mean_test_average_precision'][best_idx]),
        "std_val_pr_auc": float(cv_results['std_test_average_precision'][best_idx]),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }

    with open(save_path_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Résultats sauvegardés: {save_path_json}")
    if plot:
        # Scatter plot avec frontières de décision (PCA 2D)
    
        X_all_2d = features
        X_train_2d = X_train
        X_test_2d = X_test

        # Entraîner un RF sur les 2 composantes pour tracer les frontières
        clf_2d = RandomForestClassifier(**grid.best_params_)
        clf_2d.fit(X_train_2d, y_train)

        # Grille de décision
        x_min, x_max = X_all_2d[:, 0].min() - 1, X_all_2d[:, 0].max() + 1
        y_min, y_max = X_all_2d[:, 1].min() - 1, X_all_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 7))
        plt.contourf(xx, yy, Z, levels=50, cmap='RdYlGn_r', alpha=0.6)
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40, label='Train')
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='viridis', edgecolor='r', s=60, marker='^', label='Test')
        plt.xlim(x_min, x_max)
        plt.ylim(0, 0.22)
        plt.colorbar(label='P(Fabricated)')
        
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("RF Decision Boundary")
        plt.legend()
        plt.grid(True)
        dirname_plot = os.path.dirname(save_path_plot)
        if dirname_plot:
            os.makedirs(dirname_plot, exist_ok=True)
        
        plt.savefig(save_path_plot, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot sauvegardé: {save_path_plot}")



# evalRF(features, labels, 
#        save_path_plot=f"{RESULTS_DIR}/metricsResults/Plots/rf_decision_boundary.png",
#        save_path_json=f"{RESULTS_DIR}/metricsResults/json/rf_results_entropy_SemanticEntropy_SemanticDispersion.json", nbcomponents=None, plot=False)

def evalLogiticRegression(features, labels, save_path_json, save_path_plot=None):
    proportion_train = 0.8
    #shuffle les données avant de les séparer en train/test
    indices = np.arange(len(features))
    np.random.seed(42)
    
    np.random.shuffle(indices)
    features = features[indices]
    labels = np.array(labels)[indices]

    split_index = int(len(features) * proportion_train)
    X_train = features[:split_index]
    y_train = labels[:split_index]
    print("Training features shape:", X_train.shape)
    print("Training labels shape:", len(y_train))
    X_test = features[split_index:]
    y_test = labels[split_index:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression(random_state=42, max_iter=10000)

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
        "l1_ratio": [0, 0.5, 1],   # équivalent L1
        "solver": ["saga"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
        scoring=["roc_auc", "average_precision"],
        refit="average_precision",
    )
    grid.fit(X_train, y_train)

    print("\n=== Best hyperparameters ===")
    for k, v in grid.best_params_.items():
        print(f"{k}: {v}")

    print(f"\nBest CV average precision: {grid.best_score_:.4f}")
    print("Best train average precision: " f"{grid.best_score_:.4f}")
    test_proba_lr = grid.best_estimator_.predict_proba(X_test)[:, 1]
    test_roc_auc_lr = roc_auc_score(y_test, test_proba_lr)
    test_pr_auc_lr = average_precision_score(y_test, test_proba_lr)
    pr_auc_baseline_lr = y_test.sum() / len(y_test)
    print("Best ROC-AUC:       " f"{test_roc_auc_lr:.4f}")
    print(f"Best PR-AUC:        {test_pr_auc_lr:.4f} (baseline: {pr_auc_baseline_lr:.4f})")

    # Afficher les coefficients
    best_model = grid.best_estimator_
    print("\n=== Coefficients ===")
    for name, coef in zip(feature_names, best_model.coef_[0]):
        print(f"  {name}: {coef:.6f}" + (" (éliminé)" if abs(coef) < 1e-8 else ""))
    print(f"  intercept: {best_model.intercept_[0]:.6f}")

    # --- Plot: sigmoid output vs sorted samples, colored by label ---
    if save_path_plot is not None:
        # Restrict the visualization to the test split so class separation is easier to read.
        test_proba_plot = best_model.predict_proba(X_test)[:, 1]

        # Compute the linear projection (w^T x + b) for sigmoid curve
        linear_proj = X_test @ best_model.coef_[0] + best_model.intercept_[0]

        # Sort by linear projection
        sort_idx = np.argsort(linear_proj)
        sorted_proj = linear_proj[sort_idx]
        sorted_proba = test_proba_plot[sort_idx]
        sorted_labels = y_test[sort_idx]

        # Draw the theoretical sigmoid over the projection range
        proj_range = np.linspace(sorted_proj.min() - 1, sorted_proj.max() + 1, 500)

        plt.figure(figsize=(12, 5))
        # x-axis = sample index (sorted by projection)
        x_axis = np.arange(len(sorted_labels))

        # Scatter each point at its sorted index, y = predicted proba, colored by label
        colors_map = np.where(sorted_labels == 1, 'red', 'green')
        plt.scatter(x_axis, sorted_proba,
                    c=colors_map, s=25, alpha=0.7, edgecolor='none', zorder=3)

        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Threshold 0.5')
        plt.xlabel("Test sample index (sorted by w·x + b)")
        plt.ylabel("P(Hallucinated)")
        plt.title(f"Logistic Regression on Test Set - ROC-AUC: {test_roc_auc_lr:.3f} | PR-AUC: {test_pr_auc_lr:.3f}")

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Correct (0)'),
            Patch(facecolor='red', label='Hallucinated (1)'),
        ]
        plt.legend(handles=legend_elements, loc='best')
        plt.grid(True, alpha=0.3)

        dirname_plot = os.path.dirname(save_path_plot)
        if dirname_plot:
            os.makedirs(dirname_plot, exist_ok=True)
        plt.savefig(save_path_plot, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot sauvegardé: {save_path_plot}")

    # Sauvegarder les résultats en JSON
    dirname = os.path.dirname(save_path_json)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    cv_results = grid.cv_results_
    best_idx = grid.best_index_

    results = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "mean_train_roc_auc": float(cv_results['mean_train_roc_auc'][best_idx]),
        "std_train_roc_auc": float(cv_results['std_train_roc_auc'][best_idx]),
        "mean_val_roc_auc": float(cv_results['mean_test_roc_auc'][best_idx]),
        "std_val_roc_auc": float(cv_results['std_test_roc_auc'][best_idx]),
        "mean_train_pr_auc": float(cv_results['mean_train_average_precision'][best_idx]),
        "std_train_pr_auc": float(cv_results['std_train_average_precision'][best_idx]),
        "mean_val_pr_auc": float(cv_results['mean_test_average_precision'][best_idx]),
        "std_val_pr_auc": float(cv_results['std_test_average_precision'][best_idx]),
        "test_roc_auc": float(test_roc_auc_lr),
        "test_pr_auc": float(test_pr_auc_lr),
    }

    with open(save_path_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Résultats sauvegardés: {save_path_json}")
    

# features = features[:, [0, 1, 2, 3]]  

evalLogiticRegression(features, labels,
       save_path_json=f"{RESULTS_DIR}/metricsResults/json/logistic_regression_results_entropy_SemanticEntropy_SemanticDispersion.json",
       save_path_plot=f"{RESULTS_DIR}/metricsResults/Plots/logistic_regression_decision_boundary.png")




def evalLGBM(features, labels, save_path_json, nbcomponents=None):
    features = np.asarray(features)  
    if nbcomponents is not None:
        pca = PCA(n_components=nbcomponents)
        features = pca.fit_transform(features)
        
    # Mélange des données avant séparation Train/Test
    proportion_train = 0.8
    indices = np.arange(len(features))
    np.random.seed(42)
    np.random.shuffle(indices)
    features = features[indices]
    labels = np.array(labels)[indices]

    split_index = int(len(features) * proportion_train)
    X_train = features[:split_index]
    y_train = labels[:split_index]
    print("Training features shape:", X_train.shape)
    print("Training labels shape:", len(y_train))

    X_test = features[split_index:]
    y_test = labels[split_index:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialisation du classifieur LightGBM
    # verbosity=-1 permet d'éviter les logs trop verbeux pendant la GridSearch
    classifier = LGBMClassifier(random_state=42, verbosity=-1, n_jobs=4)
    # Grille d'hyperparamètres adaptée à LightGBM et à ton volume de données
    param_grid = {
        "n_estimators": [100, 300, 500,],
        "learning_rate": [0.005, 0.01, 0.05],   # Vitesse d'apprentissage (spécifique au Boosting)
        "max_depth": [-1, 3, 5, 8],            # -1 signifie aucune limite de profondeur
        "num_leaves": [7, 15, 31],             # LightGBM ajuste par feuilles, max_depth seul ne suffit pas
        "reg_alpha": [0.0, 0.1, 1.0],          # Régularisation L1 (Lasso)
        "reg_lambda": [0.0, 1.0, 5.0],         # Régularisation L2 (Ridge)
        "random_state": [42],
    }
#     param_grid = {
#     "n_estimators": [100, 300],          # 100 ou 300 suffisent largement pour 1600 lignes
#     "learning_rate": [0.05, 0.1],        # On retire le 0.01 qui demande trop d'arbres pour converger
#     "max_depth": [3, 5],                 # On fixe une petite profondeur, idéale pour du boosting
#     "num_leaves": [7, 15],               # Doit rester cohérent avec max_depth (2^3=8 et 2^5=32)
#     "reg_alpha": [0.0, 1.0],             # On teste L1 : Sans ou Fort
#     "reg_lambda": [1.0, 5.0],            # On teste L2 : Moyen ou Fort (LightGBM aime avoir un peu de L2 par défaut)
#     "random_state": [42],
# }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        n_jobs=4,  
        scoring=["roc_auc", "average_precision"],
        refit="average_precision",  
        return_train_score=True,
        verbose=1,
    )
    
    # Entraînement
    grid.fit(X_train, y_train)

    print("\n=== Best hyperparameters ===")
    for k, v in grid.best_params_.items():
        print(f"{k}: {v}")

    print(f"\nBest CV average precision: {grid.best_score_:.4f}")

    cv_results = grid.cv_results_
    best_idx = grid.best_index_
    
    print(
        "Best train PR AUC: "
        f"{cv_results['mean_train_average_precision'][best_idx]:.4f} +/- {cv_results['std_train_average_precision'][best_idx]:.4f}"
    )
    print(
        "Best val PR AUC:   "
        f"{cv_results['mean_test_average_precision'][best_idx]:.4f} +/- {cv_results['std_test_average_precision'][best_idx]:.4f}"
    )
    print(
        "Best train ROC AUC: "
        f"{cv_results['mean_train_roc_auc'][best_idx]:.4f} +/- {cv_results['std_train_roc_auc'][best_idx]:.4f}"
    )
    print(
        "Best val ROC AUC:   "
        f"{cv_results['mean_test_roc_auc'][best_idx]:.4f} +/- {cv_results['std_test_roc_auc'][best_idx]:.4f}"
    )

    # Évaluation sur le jeu de test
    test_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)
    pr_auc_baseline = y_test.sum() / len(y_test)
    
    print(f"Best ROC-AUC: {test_roc_auc:.4f}")
    print(f"Best PR-AUC:  {test_pr_auc:.4f} (baseline: {pr_auc_baseline:.4f})")

    # Sauvegarde des résultats en JSON
    dirname = os.path.dirname(save_path_json)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    results = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "mean_train_roc_auc": float(cv_results['mean_train_roc_auc'][best_idx]),
        "std_train_roc_auc": float(cv_results['std_train_roc_auc'][best_idx]),
        "mean_val_roc_auc": float(cv_results['mean_test_roc_auc'][best_idx]),
        "std_val_roc_auc": float(cv_results['std_test_roc_auc'][best_idx]),
        "mean_train_pr_auc": float(cv_results['mean_train_average_precision'][best_idx]),
        "std_train_pr_auc": float(cv_results['std_train_average_precision'][best_idx]),
        "mean_val_pr_auc": float(cv_results['mean_test_average_precision'][best_idx]),
        "std_val_pr_auc": float(cv_results['std_test_average_precision'][best_idx]),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }

    with open(save_path_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Résultats sauvegardés: {save_path_json}")


evalLGBM(features, labels,
       save_path_json=f"{RESULTS_DIR}/metricsResults/json/lgbm_results_entropy_SemanticEntropy_SemanticDispersion.json",
       nbcomponents=None)


def evalXGBoost(features, labels, save_path_json, nbcomponents=None):
    features = np.asarray(features)  
    if nbcomponents is not None:
        pca = PCA(n_components=nbcomponents)
        features = pca.fit_transform(features)
        
    # Mélange des données avant séparation Train/Test
    proportion_train = 0.8
    indices = np.arange(len(features))
    np.random.seed(42)
    np.random.shuffle(indices)
    features = features[indices]
    labels = np.array(labels)[indices]

    split_index = int(len(features) * proportion_train)
    X_train = features[:split_index]
    y_train = labels[:split_index]
    print("Training features shape:", X_train.shape)
    print("Training labels shape:", len(y_train))

    X_test = features[split_index:]
    y_test = labels[split_index:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialisation du classifieur XGBoost
    # n_jobs=4 ici limite l'utilisation interne de XGBoost à 4 cœurs par modèle
    classifier = XGBClassifier(
        random_state=42, 
        n_jobs=4, 
        eval_metric="logloss"  # Évite un warning de configuration d'XGBoost
    )

    # Grille d'hyperparamètres optimisée et allégée pour XGBoost
    param_grid = {
        "n_estimators": [100, 300],          # Équivalent de max_iter
        "learning_rate": [1e-3, 0.05, 0.1],        # Pas d'apprentissage
        "max_depth": [None, 3, 5],                 # XGBoost grandit par niveau (level-wise), max_depth est crucial
        "reg_alpha": [0.0, 1.0],             # Régularisation L1 (Lasso) -> Nom XGBoost pour reg_alpha
        "reg_lambda": [1.0, 5.0],            # Régularisation L2 (Ridge) -> Nom XGBoost pour reg_lambda
        "random_state": [42],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        n_jobs=4,  # 4 modèles en parallèle max -> Total max = 4 x 4 = 16 cœurs
        scoring=["roc_auc", "average_precision"],
        refit="average_precision",  
        return_train_score=True,
        verbose=1,
    )
    
    # Entraînement
    grid.fit(X_train, y_train)

    print("\n=== Best hyperparameters ===")
    for k, v in grid.best_params_.items():
        print(f"{k}: {v}")

    print(f"\nBest CV average precision: {grid.best_score_:.4f}")

    cv_results = grid.cv_results_
    best_idx = grid.best_index_
    
    print(
        "Best train PR AUC: "
        f"{cv_results['mean_train_average_precision'][best_idx]:.4f} +/- {cv_results['std_train_average_precision'][best_idx]:.4f}"
    )
    print(
        "Best val PR AUC:   "
        f"{cv_results['mean_test_average_precision'][best_idx]:.4f} +/- {cv_results['std_test_average_precision'][best_idx]:.4f}"
    )
    print(
        "Best train ROC AUC: "
        f"{cv_results['mean_train_roc_auc'][best_idx]:.4f} +/- {cv_results['std_train_roc_auc'][best_idx]:.4f}"
    )
    print(
        "Best val ROC AUC:   "
        f"{cv_results['mean_test_roc_auc'][best_idx]:.4f} +/- {cv_results['std_test_roc_auc'][best_idx]:.4f}"
    )

    # Évaluation sur le jeu de test
    test_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)
    pr_auc_baseline = y_test.sum() / len(y_test)
    
    print(f"Best ROC-AUC: {test_roc_auc:.4f}")
    print(f"Best PR-AUC:  {test_pr_auc:.4f} (baseline: {pr_auc_baseline:.4f})")

    # Sauvegarde des résultats en JSON
    dirname = os.path.dirname(save_path_json)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    results = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "mean_train_roc_auc": float(cv_results['mean_train_roc_auc'][best_idx]),
        "std_train_roc_auc": float(cv_results['std_train_roc_auc'][best_idx]),
        "mean_val_roc_auc": float(cv_results['mean_test_roc_auc'][best_idx]),
        "std_val_roc_auc": float(cv_results['std_test_roc_auc'][best_idx]),
        "mean_train_pr_auc": float(cv_results['mean_train_average_precision'][best_idx]),
        "std_train_pr_auc": float(cv_results['std_train_average_precision'][best_idx]),
        "mean_val_pr_auc": float(cv_results['mean_test_average_precision'][best_idx]),
        "std_val_pr_auc": float(cv_results['std_test_average_precision'][best_idx]),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }

    with open(save_path_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Résultats sauvegardés: {save_path_json}")



# evalXGBoost(features, labels,
#        save_path_json=f"{RESULTS_DIR}/metricsResults/json/xgboost_results_entropy_SemanticEntropy_SemanticDispersion.json",
#        nbcomponents=None)