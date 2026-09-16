"""Command-line interface: fixture, analyze, recommend, compile, baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compiler.errors import UnknownStrategyError
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.llm.prompts import user_prompt
from compiler.parsers.tiny_cnn import flatten_features, write_tiny_cnn
from compiler.pipeline import compile_and_benchmark, get_backend
from compiler.schema_validate import SchemaError, validate_llm_proposal
from compiler.strategies import ALLOWED_STRATEGY_NAMES
from nnc.artifact import benchmark_artifact, load_ort_artifact
from nnc.probe import probe_host

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "tiny_cnn.onnx"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


def _print(data: object) -> None:
    json.dump(data, sys.stdout, indent=2, sort_keys=True, default=str)
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
    compile_p.add_argument("--backend", choices=("ort_cpu", "tensorrt"), default="ort_cpu")
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
