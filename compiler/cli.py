"""Command-line interface: fixture, analyze, recommend, compile, baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compiler.errors import UnknownStrategyError
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.groq_client import GroqLlmClient, DEFAULT_GROQ_MODEL, load_groq_api_key
from compiler.llm.llm_client import LlmClient
from compiler.llm.prompts import user_prompt
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.parsers.tiny_cnn import flatten_features, write_tiny_cnn
from compiler.pipeline import compile_and_benchmark, get_backend
from compiler.schema_validate import SchemaError, validate_llm_proposal
from compiler.strategies import ALLOWED_STRATEGY_NAMES
from nnc.artifact import benchmark_artifact, load_ort_artifact
from nnc.backends import known_backends
from nnc.probe import probe_host

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "tiny_cnn.onnx"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


def _redact(value: object) -> object:
    text = json.dumps(value, default=str)
    if "gsk_" in text:
        raise SchemaError("refusing to print a Groq-looking secret")
    return value


def _print(data: object) -> None:
    json.dump(_redact(data), sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def cmd_emit_fixture(args: argparse.Namespace) -> int:
    path = write_tiny_cnn(
        Path(args.out),
        gemm_k=args.gemm_k,
        validate=not args.broken,
    )
    _print(
        {
            "path": str(path),
            "flatten_k": flatten_features(),
            "gemm_k": args.gemm_k or flatten_features(),
            "broken": bool(args.broken),
        }
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    loaded = load_graph(Path(args.model))
    summary = summarize_graph(loaded)
    _print(summary.to_dict())
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    loaded = load_graph(Path(args.model))
    summary = summarize_graph(loaded)
    rec = recommend_strategy(summary, override=args.strategy)
    payload = rec.to_dict()
    validate_llm_proposal(
        {
            "strategy": payload["strategy"],
            "rationale": payload["rationale"],
            "source": payload["source"],
        }
    )
    if args.show_prompt:
        payload["prompt"] = user_prompt(summary)
    _print(payload)
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    artifact = load_ort_artifact(Path(args.model), graph_opt=args.graph_opt)
    record = benchmark_artifact(artifact, warmup=args.warmup, iters=args.iters)
    _print(record)
    return 0 if record["latency_ms"]["mean"] > 0 else 1


def cmd_compile(args: argparse.Namespace) -> int:
    record = compile_and_benchmark(
        Path(args.model),
        backend_name=args.backend,
        strategy_name=args.strategy,
        warmup=args.warmup,
        iters=args.iters,
        results_dir=Path(args.results) if args.results else None,
        model_kind=args.kind,
    )
    _print(record)
    return 0 if record["compile"]["ok"] or record.get("skip") else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    results = Path(args.results)
    fixture = Path(args.fixture)
    if not fixture.exists():
        write_tiny_cnn(fixture)

    records = []
    records.append(
        compile_and_benchmark(
            fixture,
            backend_name="ort_cpu",
            strategy_name="baseline",
            warmup=args.warmup,
            iters=args.iters,
            results_dir=results,
            model_kind="fixture",
        )
    )

    trt = get_backend("tensorrt")
    ok, reason = trt.available()
    if ok:
        records.append(
            compile_and_benchmark(
                fixture,
                backend_name="tensorrt",
                strategy_name="baseline",
                warmup=args.warmup,
                iters=args.iters,
                results_dir=results,
                model_kind="fixture",
            )
        )
    else:
        skip_record = {
            "schema_version": 1,
            "backend": "tensorrt",
            "model": {"path": str(fixture), "kind": "fixture"},
            "compile": {"ok": False, "ms": None, "error": None},
            "benchmark": None,
            "skip": {"backend": "tensorrt", "reason": reason},
        }
        from compiler.history import append_history, write_run_json
        from compiler.utils.timeutil import utc_now_iso
        import uuid

        skip_record["run_id"] = str(uuid.uuid4())
        skip_record["created_at"] = utc_now_iso()
        write_run_json(results / "runs" / f"{skip_record['run_id']}.json", skip_record)
        append_history(results / "history.jsonl", skip_record)
        records.append(skip_record)

    yolo = Path(args.yolo) if args.yolo else REPO_ROOT / "experiments" / "models" / "yolov8n.onnx"
    if yolo.exists():
        records.append(
            compile_and_benchmark(
                yolo,
                backend_name="ort_cpu",
                strategy_name="ort_extended",
                warmup=max(3, args.warmup // 2),
                iters=max(5, args.iters // 2),
                results_dir=results,
                model_kind="yolov8n",
            )
        )
    else:
        records.append(
            {
                "backend": "ort_cpu",
                "model": {"path": str(yolo), "kind": "yolov8n"},
                "skip": {
                    "backend": "ort_cpu",
                    "reason": f"YOLOv8n ONNX not present at {yolo}; run scripts/export_yolov8n.py",
                },
            }
        )

    host = probe_host()
    summary = {
        "host": host,
        "runs": records,
        "flatten_k": flatten_features(),
    }
    out = results / "baseline_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _print({"wrote": str(out), "run_count": len(records)})
    ort_ok = any(r.get("backend") == "ort_cpu" and r.get("compile", {}).get("ok") for r in records)
    return 0 if ort_ok else 1


def cmd_probe(args: argparse.Namespace) -> int:
    _print(probe_host())
    return 0


def _advise_one(model: Path, client: LlmClient, *, kind: str, results: Path, warmup: int, iters: int) -> dict:
    loaded = load_graph(model)
    summary = summarize_graph(loaded)
    try:
        rec = recommend_strategy(summary, client=client)
        proposal_error = None
    except SchemaError as exc:
        rec = None
        proposal_error = str(exc)
    entry: dict = {
        "model": {"path": str(model), "kind": kind, "sha256": loaded.sha256},
        "advisor": {
            "backend": getattr(client, "name", "unknown"),
            "model": getattr(client, "model", None),
            "key_present": load_groq_api_key() is not None if getattr(client, "name", "") == "groq" else False,
        },
        "proposal": None if rec is None else rec.to_dict(),
        "proposal_error": proposal_error,
        "compile": None,
        "compare_baseline": None,
    }
    if rec is None:
        return entry
    compile_record = compile_and_benchmark(
        model,
        backend_name="ort_cpu",
        strategy_name=rec.strategy.name,
        warmup=warmup,
        iters=iters,
        results_dir=results,
        model_kind=kind,
    )
    entry["compile"] = {
        "ok": compile_record.get("compile", {}).get("ok"),
        "ms": compile_record.get("compile", {}).get("ms"),
        "error": compile_record.get("compile", {}).get("error"),
        "skip": compile_record.get("skip"),
        "benchmark": compile_record.get("benchmark"),
        "run_id": compile_record.get("run_id"),
        "strategy": rec.strategy.name,
    }
    if kind == "yolov8n" and rec.strategy.name != "ort_extended":
        baseline = compile_and_benchmark(
            model,
            backend_name="ort_cpu",
            strategy_name="ort_extended",
            warmup=warmup,
            iters=iters,
            results_dir=results,
            model_kind=kind,
        )
        entry["compare_baseline"] = {
            "strategy": "ort_extended",
            "ok": baseline.get("compile", {}).get("ok"),
            "benchmark": baseline.get("benchmark"),
            "run_id": baseline.get("run_id"),
            "error": baseline.get("compile", {}).get("error"),
        }
    return entry


def cmd_live(args: argparse.Namespace) -> int:
    if load_groq_api_key() is None:
        print("GROQ_API_KEY missing; set env or ~/.config/nnc/groq.env", file=sys.stderr)
        return 2
    client = GroqLlmClient(model=args.model_name)
    results = Path(args.results)
    jobs: list[tuple[Path, str]] = []
    if not args.no_fixture:
        fixture = Path(args.fixture)
        if not fixture.exists():
            write_tiny_cnn(fixture)
        jobs.append((fixture, "fixture"))
    model = Path(args.onnx)
    jobs.append((model, args.kind))

    trt = get_backend("tensorrt")
    trt_ok, trt_reason = trt.available()
    entries = [
        _advise_one(path, client, kind=kind, results=results, warmup=args.warmup, iters=args.iters)
        for path, kind in jobs
    ]
    strategies = sorted(
        {e["proposal"]["strategy"] for e in entries if e.get("proposal")}
    )
    summary = {
        "advisor": {"backend": client.name, "model": getattr(client, "model", None)},
        "tensorrt": {"available": trt_ok, "reason": None if trt_ok else trt_reason},
        "host": probe_host(),
        "live_strategy_count": len(strategies),
        "live_strategies": strategies,
        "runs": entries,
    }
    out = results / "groq_live.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _print({"wrote": str(out), "live_strategy_count": len(strategies), "strategies": strategies})
    compile_ok = any(e.get("compile", {}) and e["compile"].get("ok") for e in entries)
    return 0 if compile_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nnc", description="LLM-guided neural compiler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit-fixture", help="Write the tiny CNN ONNX fixture")
    emit.add_argument("--out", default=str(DEFAULT_FIXTURE))
    emit.add_argument("--gemm-k", type=int, default=None, help="Override Gemm K (default: Flatten width 64)")
    emit.add_argument("--broken", action="store_true", help="Allow Gemm K != Flatten width (regression case)")
    emit.set_defaults(func=cmd_emit_fixture)

    analyze = sub.add_parser("analyze", help="Print a graph summary")
    analyze.add_argument("model")
    analyze.set_defaults(func=cmd_analyze)

    rec = sub.add_parser("recommend", help="Allowlisted strategy recommendation")
    rec.add_argument("model")
    rec.add_argument("--strategy", default=None, help="Override (must be allowlisted)")
    rec.add_argument("--show-prompt", action="store_true")
    rec.set_defaults(func=cmd_recommend)

    infer = sub.add_parser("infer", help="Load an ORT CPU artifact and measure real latency")
    infer.add_argument("model")
    infer.add_argument("--graph-opt", default="extended", choices=("disable", "basic", "extended", "all"))
    infer.add_argument("--warmup", type=int, default=1)
    infer.add_argument("--iters", type=int, default=3)
    infer.set_defaults(func=cmd_infer)

    compile_p = sub.add_parser("compile", help="Compile and benchmark")
    compile_p.add_argument("model")
    compile_p.add_argument("--backend", choices=known_backends() or ("ort_cpu", "tensorrt"), default="ort_cpu")
    compile_p.add_argument("--strategy", default="baseline", choices=ALLOWED_STRATEGY_NAMES)
    compile_p.add_argument("--warmup", type=int, default=10)
    compile_p.add_argument("--iters", type=int, default=50)
    compile_p.add_argument("--results", default=str(DEFAULT_RESULTS))
    compile_p.add_argument("--kind", default="onnx")
    compile_p.set_defaults(func=cmd_compile)

    baseline = sub.add_parser("baseline", help="ORT CPU baseline JSON; TensorRT skipped unless NVIDIA exists")
    baseline.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    baseline.add_argument("--results", default=str(DEFAULT_RESULTS))
    baseline.add_argument("--yolo", default="")
    baseline.add_argument("--warmup", type=int, default=10)
    baseline.add_argument("--iters", type=int, default=50)
    baseline.set_defaults(func=cmd_baseline)

    probe = sub.add_parser("probe", help="Print host/PX4/ROS/Gazebo/NVIDIA facts")
    probe.set_defaults(func=cmd_probe)

    live = sub.add_parser("live", help="Groq advisor then ORT compile/profile (requires GROQ_API_KEY)")
    live.add_argument("onnx")
    live.add_argument("--kind", default="yolov8n")
    live.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Also advise+compile the tiny fixture")
    live.add_argument("--no-fixture", action="store_true")
    live.add_argument("--model-name", default=DEFAULT_GROQ_MODEL)
    live.add_argument("--results", default=str(DEFAULT_RESULTS))
    live.add_argument("--warmup", type=int, default=3)
    live.add_argument("--iters", type=int, default=8)
    live.set_defaults(func=cmd_live)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except UnknownStrategyError as exc:
        print(exc, file=sys.stderr)
        return 2
    except SchemaError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
