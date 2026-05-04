import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PipelineTest.AnalyseResults import mergeOutputsList
from StudyAutoRegressionBehavior.GetInfoFromBaseSamplerOutput import (
    getEachStepChange,
    getEachStepMask,
    getEntropy,
    getH,
    getLogProbs,
)


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def average_precision_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return float("nan")

    tp = 0
    ap_acc = 0.0
    for i, yi in enumerate(y_sorted, start=1):
        if yi == 1:
            tp += 1
            ap_acc += tp / i
    return float(ap_acc / n_pos)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float]:
    thresholds = np.unique(y_score)
    best = (float("nan"), 0.0, 0.0, 0.0)
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best[1]:
            best = (float(thr), float(f1), float(prec), float(rec))
    return best


def get_entropy_with_fallback(outputs, logprobs):
    entropies = getEntropy(outputs)
    if entropies is not None:
        return entropies

    h_vals = getH(outputs) if outputs.histories_H is not None else None
    if h_vals is not None:
        print("[warn] histories_entropy is None; using histories_H fallback")
        return h_vals

    print("[warn] histories_entropy and histories_H are None; using abs(logprobs) proxy")
    return [np.abs(np.asarray(lp, dtype=float)) for lp in logprobs]


def compute_pci_per_sample(masks, changes, entropies, beta=0.7):
    pci = []
    for m_j, c_j, h_j in zip(masks, changes, entropies):
        m = np.asarray(m_j, dtype=float)
        c = np.asarray(c_j, dtype=float)
        h = np.asarray(h_j, dtype=float)

        unmasked = 1.0 - m
        k = unmasked.sum(axis=1)
        denom = np.maximum(k, 1e-8)
        changes_after = (c * unmasked).sum(axis=1) / denom
        entropy_after = (h * unmasked).sum(axis=1) / denom
        p = beta * changes_after + (1.0 - beta) * entropy_after
        p[k == 0] = 0.0
        pci.append(float(np.mean(p)))
    return np.asarray(pci, dtype=float)


def evaluate_score(y_true: np.ndarray, score: np.ndarray) -> dict:
    auc_raw = roc_auc_score_manual(y_true, score)
    auc_flip = roc_auc_score_manual(y_true, -score)

    use_flipped = auc_flip > auc_raw
    s = -score if use_flipped else score

    auc = roc_auc_score_manual(y_true, s)
    ap = average_precision_manual(y_true, s)
    thr, f1, prec, rec = best_f1_threshold(y_true, s)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "best_threshold": float(thr),
        "best_f1": float(f1),
        "precision_at_best_f1": float(prec),
        "recall_at_best_f1": float(rec),
        "direction": "higher_is_more_fabricated" if not use_flipped else "lower_is_more_fabricated",
    }


def main():
    correct = [4, 9, 10, 13, 15, 17, 18, 19, 20, 21, 24, 26, 28, 29, 31, 33, 34, 36, 45, 48, 49, 54, 61, 62, 65, 66, 67, 68, 69, 70, 75, 76, 78, 79, 80, 82, 84, 88, 90, 91, 93, 94, 103, 104, 105, 106, 107, 109, 115, 116, 119, 120, 122, 123, 124, 130, 132, 133, 134, 135, 137, 146, 149, 151, 152, 155, 156, 157, 158, 160, 163, 164, 166, 168, 169, 170, 171, 172, 173, 175, 177, 178, 179, 181, 182, 183, 184, 185, 188, 190, 193, 202, 203, 204, 205, 211, 214, 215, 216, 219, 220, 221, 227, 229, 230, 232, 233, 236, 237, 240, 242, 245, 246, 247, 248, 252, 253, 255]
    partial = [0, 6, 25, 30, 35, 42, 47, 53, 55, 56, 71, 72, 73, 95, 96, 97, 136, 145, 148, 159, 165, 174, 176, 187, 191, 194, 196, 198, 213, 218, 223, 224, 238, 243, 254]
    fabricated = [1, 2, 3, 5, 7, 8, 11, 12, 14, 16, 22, 23, 27, 32, 37, 38, 39, 40, 41, 43, 44, 46, 50, 51, 52, 57, 58, 59, 60, 63, 64, 74, 77, 81, 83, 85, 86, 87, 89, 92, 98, 99, 100, 101, 102, 108, 110, 111, 112, 113, 114, 117, 118, 121, 125, 126, 127, 128, 129, 131, 138, 139, 140, 141, 142, 143, 144, 147, 150, 153, 154, 161, 162, 167, 180, 186, 189, 192, 195, 197, 199, 200, 201, 206, 207, 209, 210, 212, 217, 222, 225, 226, 228, 231, 234, 235, 239, 241, 244, 249, 250, 251]
    empty_garbled = [208]
    all_idx = sorted(correct + fabricated + partial + empty_garbled)
    y = np.array([0 if i in correct else 1 for i in all_idx], dtype=int)

    output_path = "PipelineTest/results_64_step/outputs_64_step.pt"
    report_path = "PipelineTest/results_64_step/eval_metric_correct_vs_fabricated.json"

    outputs = mergeOutputsList(output_path)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    changes = getEachStepChange(outputs)
    entropies = get_entropy_with_fallback(outputs, logprobs)

    var_entropy_all = np.asarray([
        float(np.mean(np.var(np.asarray(entropies[i], dtype=float), axis=1)))
        for i in range(len(entropies))
    ])
    var_logprobs_all = np.asarray([
        float(np.var(np.asarray(logprobs[i], dtype=float)))
        for i in range(len(logprobs))
    ])
    mean_entropy_all = np.asarray([
        float(np.mean(np.asarray(entropies[i], dtype=float)))
        for i in range(len(entropies))
    ])
    mean_logprobs_all = np.asarray([
        float(np.mean(np.asarray(logprobs[i], dtype=float)))
        for i in range(len(logprobs))
    ])
    var_entropy = var_entropy_all[all_idx]
    var_logprobs = var_logprobs_all[all_idx]
    mean_logprobs = mean_logprobs_all[all_idx]
    results = {}
    results["mean_entropy"] = evaluate_score(y, mean_entropy_all[all_idx])
    results["var_entropy"] = evaluate_score(y, var_entropy)
    results["var_logprobs"] = evaluate_score(y, var_logprobs)
    results["mean_logprobs"] = evaluate_score(y, mean_logprobs)

    pci_07_all = compute_pci_per_sample(masks, changes, entropies, beta=0.7)
    results["pci_beta_0.7"] = evaluate_score(y, pci_07_all[all_idx])

    best_csg = None
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            pci_all = compute_pci_per_sample(masks, changes, entropies, beta=beta)
            csg_all = np.asarray([
                float(np.mean(np.asarray(lp, dtype=float)) - alpha * pci_val)
                for lp, pci_val in zip(logprobs, pci_all)
            ])
            stats = evaluate_score(y, csg_all[all_idx])
            key = f"csg_alpha_{alpha}_beta_{beta}"
            results[key] = stats

            auc = stats["auc"]
            if best_csg is None or auc > best_csg["auc"]:
                best_csg = {
                    "name": key,
                    "alpha": alpha,
                    "beta": beta,
                    "auc": auc,
                    "stats": stats,
                }

    summary = {
        "n_samples": int(len(all_idx)),
        "n_correct": int((y == 0).sum()),
        "n_fabricated": int((y == 1).sum()),
        "best_csg": best_csg,
        "metrics": results,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Correct vs Fabricated separation ===")
    print(f"samples={summary['n_samples']} correct={summary['n_correct']} fabricated={summary['n_fabricated']}")
    print("Top metrics by AUC:")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    for name, stats in ranked:
        print(
            f"- {name}: AUC={stats['auc']:.4f}, AP={stats['ap']:.4f}, "
            f"F1={stats['best_f1']:.4f}, thr={stats['best_threshold']:.4f}, dir={stats['direction']}"
        )
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
