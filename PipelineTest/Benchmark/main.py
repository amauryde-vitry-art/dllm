import sys
import os
import json
import argparse
import subprocess
import tempfile
import traceback
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PipelineTest.scripts.run_evaluation import CONFIGS


def _resolve_python():
    """Return the Python executable that has the project dependencies installed."""
    # Try current interpreter first
    try:
        import transformers  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    # Walk up from Benchmark/ to find .venv/bin/python
    here = os.path.dirname(os.path.abspath(__file__))
    for ancestor in [here] + [
        os.path.abspath(os.path.join(here, *[".."] * i)) for i in range(1, 6)
    ]:
        venv_python = os.path.join(ancestor, ".venv", "bin", "python")
        if os.path.isfile(venv_python):
            return venv_python
    return sys.executable

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------
BASELINE_REGISTRY = {
    "perplexity": {
        "display": "Perplexity",
        "module": "PipelineTest.Benchmark.perplexity",
        "func": "run_config",
        "needs_gpu": False,
    },
    "ln_entropy": {
        "display": "LN-Entropy",
        "module": "PipelineTest.Benchmark.LNEntropy",
        "func": "run_config",
        "needs_gpu": False,
    },
    "baseline_markov": {
        "display": "Baseline+Markovian",
        "module": "PipelineTest.Benchmark.Baseline_and_markovian_features",
        "func": "main",
        "needs_gpu": False,
    },
    "semantic_entropy": {
        "display": "Semantic Entropy",
        "module": "PipelineTest.Benchmark.semantic_entropy",
        "func": "run_config",
        "needs_gpu": True,
    },
    "lexical_similarity": {
        "display": "Lexical Similarity",
        "module": "PipelineTest.Benchmark.lexical_similarity",
        "func": "run_config",
        "needs_gpu": True,
    },
}


def _import_func(module_path, func_name):
    """Lazy-import a function from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _extract_metrics(result, baseline_key):
    """
    Normalize baseline results into {roc_auc, pr_auc}.
    Handles both flat dicts and nested (Baseline_and_markovian_features) results.
    """
    if result is None:
        return None

    if "test_roc_auc" in result and "test_pr_auc" in result:
        return {
            "roc_auc": result["test_roc_auc"],
            "pr_auc": result["test_pr_auc"],
            "n_samples": result.get("n_samples") or result.get("n_test"),
        }

    if "results" in result:
        nested = {}
        for sub_name, sub in result["results"].items():
            nested[sub_name] = {
                "roc_auc": sub.get("test_roc_auc"),
                "pr_auc": sub.get("test_pr_auc"),
            }
        return nested

    return None

def _run_gpu_baseline(bl_key, config_name, nproc):
    """
    Launch a GPU baseline via torchrun with nproc processes.
    Writes a temp script, runs it via torchrun, and streams the output live.
    """
    bl = BASELINE_REGISTRY[bl_key]
    module_name = bl["module"].split(".")[-1]
    func_name = bl["func"]

    # Absolute project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Extra args per baseline
    extra_args = ""
    if bl_key in ("semantic_entropy", "lexical_similarity"):
        extra_args = ", n_variants=5"

    # Write a launcher script to a temp file
    script_content = f"""\
import sys, os, json
sys.path.insert(0, "{project_root}")

from PipelineTest.Benchmark.{module_name} import {func_name}

result = {func_name}("{config_name}"{extra_args})

# Only rank 0 writes the result
import torch.distributed as dist
if dist.is_initialized():
    rank = dist.get_rank()
else:
    rank = 0

if rank == 0 and result is not None:
    out_dir = "/tmp/benchmark_gpu_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{bl_key}_{config_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[RESULT_SAVED] {{out_path}}")
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"bench_{bl_key}_",
        dir="/tmp", delete=False
    ) as f:
        f.write(script_content)
        script_path = f.name

    try:
        python_exe = _resolve_python()
        
        # CORRECTION 1 : Génération d'un vrai port aléatoire pour éviter le freeze du Rendezvous
        import random
        random_port = random.randint(30000, 50000)
        
        cmd = [
            python_exe, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={random_port}",
            script_path,
        ]

        # Validation et affichage du GPU actif
        import torch
        if torch.cuda.is_available():
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f"    [GPU HARDWARE ACTIVE] Alloué(s) au job : {', '.join(gpu_names)}", flush=True)
        else:
            print("    [CRITICAL WARN] Aucun GPU détecté par le script parent !", flush=True)

        print(f"    Launching: {python_exe} -m torch.distributed.run --nproc_per_node={nproc} (Port: {random_port})")
        print(f"    --- DÉBUT DES LOGS EN DIRECT DE {bl['display'].upper()} ---", flush=True)

        # CORRECTION 2 : Forcer l'environnement à être NON-TAMPONNÉ (Unbuffered)
        # Cela oblige le processus enfant à cracher ses logs immédiatement dans le pipe
        env_unbuffered = os.environ.copy()
        env_unbuffered["PYTHONUNBUFFERED"] = "1"

        # Utilisation de Popen pour lire le flux en temps réel
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env_unbuffered  # Injection de l'environnement corrigé
        )

        # Affiche les lignes au fur et à mesure sans aucun décalage de temps
        for line in proc.stdout:
            print(f"      [torchrun-live] {line}", end="", flush=True)

        proc.wait(timeout=7200)
        print(f"    --- FIN DES LOGS EN DIRECT (Exit Code: {proc.returncode}) ---", flush=True)

        if proc.returncode != 0:
            print(f"    [ERROR] torchrun failed (rc={proc.returncode})")
            return None

        # Read result from temp file
        result_path = f"/tmp/benchmark_gpu_results/{bl_key}_{config_name}.json"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            os.remove(result_path)
            return result

        print(f"    [WARN] No result file found at {result_path}")
        return None

    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_all(config_names, baselines_to_run, include_gpu=False, nproc=2):
    """
    Run selected baselines across all configs.
    CPU baselines run in-process. GPU baselines launch via torchrun.
    Returns nested dict: {config_name: {baseline: metrics}}.
    """
    all_results = {}

    for config_name in config_names:
        print(f"\n{'#' * 70}")
        print(f"  CONFIG: {config_name}")
        print(f"{'#' * 70}")
        all_results[config_name] = {}

        for bl_key in baselines_to_run:
            if bl_key not in BASELINE_REGISTRY:
                continue
            bl = BASELINE_REGISTRY[bl_key]

            if bl["needs_gpu"] and not include_gpu:
                print(f"  [SKIP] {bl['display']} requires GPU (use --include-gpu to enable)")
                continue

            print(f"\n  --- Running baseline: {bl['display']} ---")

            bl_t0 = time.time()
            try:
                if bl["needs_gpu"]:
                    result = _run_gpu_baseline(bl_key, config_name, nproc)
                else:
                    run_fn = _import_func(bl["module"], bl["func"])
                    result = run_fn(config_name)
            except Exception:
                print(f"  [ERROR] {bl['display']} failed for {config_name}:")
                traceback.print_exc()
                result = None
            bl_t1 = time.time()
            elapsed = bl_t1 - bl_t0
            
            metrics = _extract_metrics(result, bl_key)
            if metrics is not None:
                # Ajoute le timing directement dans le dict de métriques,
                # que ce soit un dict plat (roc_auc/pr_auc) ou nested
                # (Baseline_and_markovian_features -> plusieurs sous-résultats)
                if "roc_auc" in metrics:
                    metrics["elapsed_seconds"] = round(elapsed, 2)
                else:
                    # dict nested : on ajoute le timing global du run
                    # (partagé par tous les sous-résultats de cette baseline)
                    metrics["_elapsed_seconds"] = round(elapsed, 2)

                all_results[config_name][bl["display"]] = metrics
                if isinstance(metrics, dict) and "roc_auc" in metrics:
                    print(f"  => ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}  "
                          f"({elapsed:.1f}s)")
                else:
                    print(f"  => completed in {elapsed:.1f}s (see detailed output)")
            else:
                # Même sans résultat exploitable, on garde une trace du temps passé
                all_results[config_name][bl["display"]] = {"elapsed_seconds": round(elapsed, 2), "result": None}
                print(f"  => no result returned ({elapsed:.1f}s)")
           
    return all_results


def print_summary(all_results):
    """Print a compact summary table."""
    print(f"\n{'=' * 80}")
    print(f"  CONSOLIDATED BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    all_baselines = set()
    for cfg_results in all_results.values():
        all_baselines.update(cfg_results.keys())
    all_baselines = sorted(all_baselines)

    if not all_baselines:
        print("  No results collected.")
        return

    header = f"  {'Config':<50}"
    for bl in all_baselines:
        header += f" {bl[:12]:>13}"
    print(header)
    print(f"  {'-' * (50 + 13 * len(all_baselines))}")

    for config_name, cfg_results in all_results.items():
        short_name = config_name[:50]
        row = f"  {short_name:<50}"
        for bl in all_baselines:
            if bl in cfg_results:
                m = cfg_results[bl]
                if isinstance(m, dict) and "roc_auc" in m:
                    row += f" {m['roc_auc']:>6.4f}/{m['pr_auc']:<6.4f}"
                else:
                    if "Baseline+Markov" in m:
                        bm = m["Baseline+Markov"]
                        row += f" {bm.get('roc_auc',0):>6.4f}/{bm.get('pr_auc',0):<6.4f}"
                    else:
                        row += f" {'N/A':>13}"
            else:
                row += f" {'--':>13}"
        print(row)

    print(f"\n  Format per cell: ROC-AUC / PR-AUC")


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks across all CONFIGS and save consolidated results."
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help=f"Available: {list(BASELINE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Config keys to evaluate (default: all CONFIGS)"
    )
    parser.add_argument(
        "--include-gpu", action="store_true",
        help="Also run GPU-required baselines"
    )
    parser.add_argument(
        "--nproc", type=int, default=2,
        help="Number of GPUs for GPU baselines via torchrun"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "eval")),
        help="Output directory for consolidated JSON"
    )
    args = parser.parse_args()

    if args.benchmarks:
        for b in args.benchmarks:
            if b not in BASELINE_REGISTRY:
                parser.error(f"Unknown baseline '{b}'. Available: {list(BASELINE_REGISTRY.keys())}")
        baselines_to_run = args.benchmarks
    else:
        if args.include_gpu:
            baselines_to_run = list(BASELINE_REGISTRY.keys())
        else:
            baselines_to_run = [
                k for k, v in BASELINE_REGISTRY.items() if not v["needs_gpu"]
            ]

    config_names = args.configs if args.configs else list(CONFIGS.keys())

    for c in config_names:
        if c not in CONFIGS:
            parser.error(f"Unknown config '{c}'. Available: {list(CONFIGS.keys())}")

    # ... (début de la fonction main() inchangé) ...

    print(f"Configs to evaluate: {config_names}")
    print(f"Baselines to run:    {baselines_to_run}")
    print(f"Include GPU:         {args.include_gpu}")
    
    # --- AJUSTEMENT AUTOMATIQUE DU NOMBRE DE GPU ---
    nproc_to_use = args.nproc
    if args.include_gpu:
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus == 0:
            print("  [WARN] Aucun GPU détecté par PyTorch dans cet environnement ! Repli sur 1.")
            nproc_to_use = 1
        else:
            # Si tu demandes plus de GPU (ex: 4) que ce que ton "-G 1" t'a donné (ex: 1),
            # le script se bride automatiquement pour éviter le crash.
            nproc_to_use = min(args.nproc, available_gpus)
            print(f"  [INFO] GPU(s) détecté(s) dans le job : {available_gpus} | nproc ajusté à : {nproc_to_use}")

    # Lancement de l'orchestration avec le bon nombre de GPU
    all_results = run_all(
        config_names, baselines_to_run,
        include_gpu=args.include_gpu, nproc=nproc_to_use
    )

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "all" if len(config_names) > 1 else config_names[0]
    out_path = os.path.join(args.output_dir, f"benchmarks_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Consolidated results saved to: {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()