"""RQ1/RQ2 evaluation harness: Oracle, ORT Default, Heuristic, Random, BO/TPE, and LLM."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from compiler.llm.llm_client import HeuristicLlmClient
from compiler.pipeline import compile_and_benchmark
from compiler.strategies import Strategy, enumerate_strategy_space, get_strategy


def compute_regret(chosen_ms: float, oracle_ms: float) -> float:
    if oracle_ms <= 0:
        return 0.0
    return (chosen_ms - oracle_ms) / oracle_ms


def _eval_one(
    model_path: Path,
    strategy: Strategy,
    *,
    warmup: int,
    iters: int,
    metric: str = "p95",
) -> tuple[float, dict[str, Any]]:
    rec = compile_and_benchmark(
        model_path,
        backend_name="ort_cpu",
        strategy_obj=strategy,
        check_fidelity=True,
        warmup=warmup,
        iters=iters,
        results_dir=None,
    )
    if not rec["compile"]["ok"]:
        return float("inf"), rec
    val = float(rec["benchmark"]["latency_ms"][metric])
    return val, rec


def run_oracle_and_baselines(
    model_path: Path,
    *,
    budget_evals: int = 12,
    warmup: int = 5,
    iters: int = 20,
    metric: str = "p95",
    seed: int = 0,
    include_int8_static: bool = False,
    llm_client: Any = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the multi-dimensional space and compare Oracle vs. baselines vs. LLM."""

    space = enumerate_strategy_space(
        include_tensorrt=False,
        include_fp16=False,
        include_int8_static=include_int8_static,
        only_available_eps=True,
    )

    # 1. Exhaustive Oracle over space
    t_oracle0 = time.perf_counter()
    oracle_results: list[dict[str, Any]] = []
    for idx, strat in enumerate(space):
        lat_ms, rec = _eval_one(model_path, strat, warmup=warmup, iters=iters, metric=metric)
        oracle_results.append(
            {
                "index": idx,
                "config": strat.config_dict(),
                "compile_ok": rec["compile"]["ok"],
                "latency_ms": lat_ms,
                "fidelity": rec.get("fidelity"),
            }
        )
    oracle_time_s = time.perf_counter() - t_oracle0

    valid = [r for r in oracle_results if r["compile_ok"] and r["latency_ms"] < float("inf")]
    if not valid:
        raise RuntimeError(f"No valid strategy compiled for {model_path}")

    best_entry = min(valid, key=lambda r: r["latency_ms"])
    oracle_ms = float(best_entry["latency_ms"])
    lookup = {json.dumps(r["config"], sort_keys=True): r for r in oracle_results}

    def lookup_or_eval(strat: Strategy) -> tuple[float, dict[str, Any]]:
        key = json.dumps(strat.config_dict(), sort_keys=True)
        if key in lookup and lookup[key]["compile_ok"]:
            return float(lookup[key]["latency_ms"]), lookup[key]
        lat, rec = _eval_one(model_path, strat, warmup=warmup, iters=iters, metric=metric)
        return lat, {"config": strat.config_dict(), "latency_ms": lat, "fidelity": rec.get("fidelity")}

    methods: dict[str, Any] = {
        "oracle": {
            "latency_ms": oracle_ms,
            "regret": 0.0,
            "search_time_s": oracle_time_s,
            "evals": len(space),
            "config": best_entry["config"],
            "fidelity": best_entry.get("fidelity"),
        }
    }

    # 2. ORT Default (ort_all, FP32, default threads)
    t0 = time.perf_counter()
    ort_default_strat = get_strategy("ort_all")
    ort_ms, ort_rec = lookup_or_eval(ort_default_strat)
    methods["ort_default"] = {
        "latency_ms": ort_ms,
        "regret": compute_regret(ort_ms, oracle_ms),
        "search_time_s": time.perf_counter() - t0,
        "evals": 1,
        "config": ort_default_strat.config_dict(),
        "fidelity": ort_rec.get("fidelity"),
    }

    # 3. Heuristic (3-rule baseline)
    t0 = time.perf_counter()
    heur_rec = compile_and_benchmark(
        model_path,
        client=HeuristicLlmClient(),
        warmup=warmup,
        iters=iters,
        results_dir=None,
    )
    heur_ms = float(heur_rec["benchmark"]["latency_ms"][metric]) if heur_rec["compile"]["ok"] else float("inf")
    methods["heuristic"] = {
        "latency_ms": heur_ms,
        "regret": compute_regret(heur_ms, oracle_ms),
        "search_time_s": time.perf_counter() - t0,
        "evals": 1,
        "config": heur_rec["strategy"].get("config"),
        "fidelity": heur_rec.get("fidelity"),
    }

    # 4. Random Search (budget_evals)
    rng = random.Random(seed)
    t0 = time.perf_counter()
    sample = rng.sample(space, min(budget_evals, len(space)))
    rand_evals = [lookup_or_eval(s) for s in sample]
    rand_best_ms, rand_best = min(rand_evals, key=lambda x: x[0])
    methods["random_search"] = {
        "latency_ms": rand_best_ms,
        "regret": compute_regret(rand_best_ms, oracle_ms),
        "search_time_s": time.perf_counter() - t0,
        "evals": len(sample),
        "config": rand_best["config"],
        "fidelity": rand_best.get("fidelity"),
    }

    # 5. Bayesian Optimization / TPE (optuna if installed, else coordinate-hill-climb)
    t0 = time.perf_counter()
    bo_best_ms = float("inf")
    bo_best_cfg: dict[str, Any] = {}
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            idx = trial.suggest_int("strategy_idx", 0, len(space) - 1)
            lat, _ = lookup_or_eval(space[idx])
            return lat

        study.optimize(objective, n_trials=min(budget_evals, len(space)))
        bo_best_ms = float(study.best_value)
        bo_best_cfg = space[int(study.best_params["strategy_idx"])].config_dict()
        bo_kind = "optuna_tpe"
    except ImportError:
        bo_kind = "local_search_fallback"
        best_s = rng.choice(space)
        bo_best_ms, rec = lookup_or_eval(best_s)
        bo_best_cfg = rec["config"]
        for _ in range(max(0, budget_evals - 1)):
            cand = rng.choice(space)
            c_ms, c_rec = lookup_or_eval(cand)
            if c_ms < bo_best_ms:
                bo_best_ms = c_ms
                bo_best_cfg = c_rec["config"]

    methods["bayesian_opt"] = {
        "kind": bo_kind,
        "latency_ms": bo_best_ms,
        "regret": compute_regret(bo_best_ms, oracle_ms),
        "search_time_s": time.perf_counter() - t0,
        "evals": min(budget_evals, len(space)),
        "config": bo_best_cfg,
    }

    # 6. Optional Live LLM Advisor
    if llm_client is not None:
        t0 = time.perf_counter()
        llm_rec = compile_and_benchmark(
            model_path,
            client=llm_client,
            fallback_on_error=True,
            warmup=warmup,
            iters=iters,
            results_dir=None,
        )
        llm_ms = float(llm_rec["benchmark"]["latency_ms"][metric]) if llm_rec["compile"]["ok"] else float("inf")
        methods["llm_advisor"] = {
            "latency_ms": llm_ms,
            "regret": compute_regret(llm_ms, oracle_ms),
            "search_time_s": time.perf_counter() - t0,
            "evals": 1,
            "fallback_triggered": llm_rec["strategy"].get("fallback_triggered", False),
            "config": llm_rec["strategy"].get("config"),
            "fidelity": llm_rec.get("fidelity"),
        }

    report = {
        "model": str(model_path),
        "metric": metric,
        "space_size": len(space),
        "methods": methods,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
