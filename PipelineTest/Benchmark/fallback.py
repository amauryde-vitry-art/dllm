import sys
import os
import json
import argparse
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PipelineTest.scripts.run_evaluation import CONFIGS
from PipelineTest.Benchmark.main import (
    BASELINE_REGISTRY,
    _run_gpu_baseline,
    _import_func,
    _extract_metrics,
    print_summary,
)

DEFAULT_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval", "benchmarks_allV4.json")


def _is_successful(metrics):
    """Mirror print_summary's notion of a usable (non-N/A) cell."""
    if not isinstance(metrics, dict):
        return False
    if "roc_auc" in metrics:
        return True
    if "Baseline+Markov" in metrics:
        return True
    return False


def find_todo(all_results, config_names, baseline_keys):
    """Return [(config_name, baseline_key), ...] for pairs missing or failed in all_results."""
    todo = []
    for config_name in config_names:
        cfg_results = all_results.get(config_name, {})
        for bl_key in baseline_keys:
            display = BASELINE_REGISTRY[bl_key]["display"]
            if not _is_successful(cfg_results.get(display)):
                todo.append((config_name, bl_key))
    return todo


def rerun_one(config_name, bl_key, nproc):
    bl = BASELINE_REGISTRY[bl_key]
    print(f"\n  --- Rerunning: {config_name} / {bl['display']} ---")
    t0 = time.time()
    try:
        if bl["needs_gpu"]:
            result = _run_gpu_baseline(bl_key, config_name, nproc)
        else:
            run_fn = _import_func(bl["module"], bl["func"])
            result = run_fn(config_name)
    except Exception:
        print(f"  [ERROR] {bl['display']} failed again for {config_name}:")
        traceback.print_exc()
        result = None
    elapsed = time.time() - t0

    metrics = _extract_metrics(result, bl_key)
    if metrics is not None:
        if "roc_auc" in metrics:
            metrics["elapsed_seconds"] = round(elapsed, 2)
        else:
            metrics["_elapsed_seconds"] = round(elapsed, 2)
        print(f"  => success ({elapsed:.1f}s)")
        return metrics

    print(f"  => still no result ({elapsed:.1f}s)")
    return {"elapsed_seconds": round(elapsed, 2), "result": None}


def main():
    parser = argparse.ArgumentParser(
        description="Rerun only the (config, baseline) pairs that are missing or N/A "
                    "in an existing consolidated benchmarks_*.json, and patch the file in place."
    )
    parser.add_argument(
        "--results", type=str, default=DEFAULT_RESULTS_PATH,
        help="Path to the consolidated JSON to patch (default: %(default)s)"
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help=f"Restrict to these baseline keys (default: all). Available: {list(BASELINE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Restrict to these config keys (default: every config already present in --results)"
    )
    parser.add_argument("--nproc", type=int, default=2, help="Number of GPUs for GPU baselines via torchrun")
    parser.add_argument("--dry-run", action="store_true", help="Only list what would be rerun, don't execute")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        parser.error(f"Results file not found: {args.results}")

    with open(args.results, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    baseline_keys = args.benchmarks if args.benchmarks else list(BASELINE_REGISTRY.keys())
    for b in baseline_keys:
        if b not in BASELINE_REGISTRY:
            parser.error(f"Unknown baseline '{b}'. Available: {list(BASELINE_REGISTRY.keys())}")

    config_names = args.configs if args.configs else list(all_results.keys())
    for c in config_names:
        if c not in CONFIGS:
            parser.error(f"Unknown config '{c}'. Available: {list(CONFIGS.keys())}")

    todo = find_todo(all_results, config_names, baseline_keys)

    if not todo:
        print("Nothing to rerun: every (config, baseline) pair already has a usable result.")
        return

    print(f"Found {len(todo)} (config, baseline) pair(s) to rerun:")
    for cfg, bl_key in todo:
        print(f"  - {cfg}  /  {BASELINE_REGISTRY[bl_key]['display']}")

    if args.dry_run:
        return

    # Auto-adjust nproc to available GPUs, same as main.py
    nproc_to_use = args.nproc
    if any(BASELINE_REGISTRY[bl_key]["needs_gpu"] for _, bl_key in todo):
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus == 0:
            print("  [WARN] No GPU detected by PyTorch, falling back to nproc=1")
            nproc_to_use = 1
        else:
            nproc_to_use = min(args.nproc, available_gpus)
            print(f"  [INFO] {available_gpus} GPU(s) detected, using nproc={nproc_to_use}")

    for config_name, bl_key in todo:
        metrics = rerun_one(config_name, bl_key, nproc_to_use)
        all_results.setdefault(config_name, {})[BASELINE_REGISTRY[bl_key]["display"]] = metrics

        # Persist after every pair so a crash mid-run doesn't lose earlier fixes
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated results saved to: {args.results}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
