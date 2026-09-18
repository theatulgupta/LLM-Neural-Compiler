"""Command-line interface: fixture, analyze, recommend, compile, baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compiler.catalog import REPO_ROOT, get_model, load_zoo, zoo_kinds
from compiler.errors import FrontendSkip, UnknownFrontendError, UnknownStrategyError
from compiler.exporters import skip_run_record
from compiler.frontends import frontend_catalog
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.history import append_history, history_for_model, write_run_json
from compiler.llm.prompting import build_messages, user_prompt
from compiler.llm.provider import known_providers
from compiler.llm.recommendation_engine import recommend_plan, recommend_strategy
from compiler.parsers.tiny_cnn import flatten_features, write_tiny_cnn
from compiler.parsers.tiny_depth import write_tiny_depth
from compiler.pipeline import compile_and_benchmark, get_backend, measure_path, optimize_model, run_zoo_matrix
from compiler.planner import ALLOWED_STRATEGY_NAMES
from compiler.report.report_generator import write_report
from compiler.schema import SchemaError, validate_llm_proposal
from nnc.artifact import benchmark_artifact, load_ort_artifact
from nnc.backends import known_backends
from nnc.probe import probe_host

DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "tiny_cnn.onnx"
DEFAULT_DEPTH_FIXTURE = REPO_ROOT / "fixtures" / "tiny_depth.onnx"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


_SECRET_MARKERS = ("gsk_", "sk-ant-", "sk-or-", "sk-proj-", "AIzaSy", "nvapi-")


def _redact(value: object) -> object:
    text = json.dumps(value, default=str)
    if any(marker in text for marker in _SECRET_MARKERS):
        raise SchemaError("refusing to print an API-key-looking secret")
    return value


def _print(data: object) -> None:
    json.dump(_redact(data), sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def cmd_emit_fixture(args: argparse.Namespace) -> int:
    kind = str(args.kind)
    if kind == "depth":
        out = Path(args.out)
        if Path(args.out) == DEFAULT_FIXTURE:
            out = DEFAULT_DEPTH_FIXTURE
        path = write_tiny_depth(out)
        _print({"path": str(path), "kind": "tiny_depth", "input": [1, 3, 8, 8]})
        return 0
    path = write_tiny_cnn(
        Path(args.out),
        gemm_k=args.gemm_k,
        validate=not args.broken,
    )
    _print(
        {
            "path": str(path),
            "kind": "fixture",
            "flatten_k": flatten_features(),
            "gemm_k": args.gemm_k or flatten_features(),
            "broken": bool(args.broken),
        }
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    loaded = load_graph(Path(args.model))
    summary = summarize_graph(loaded)
    hardware = probe_hardware()
    constraints = {}
    for item in getattr(args, "constraint", []) or []:
        key, _, value = item.partition("=")
        constraints[key] = value
    rec = recommend_plan(
        summary, hardware, constraints, history=history_for_model(args.kind or loaded.sha256)
    )
    payload = rec.outcome.to_dict() if rec.outcome is not None else rec.strategy.to_dict()
    if args.show_prompt:
        payload["prompt"] = build_messages(summary, hardware=hardware, constraints=constraints)[1]["content"]
    _print(payload)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    payload = optimize_model(
        Path(args.model),
        kind=args.kind,
        task=args.task,
        mode=args.candidates,
        warmup=args.warmup,
        iters=args.iters,
        results_dir=Path(args.results),
    )
    _print(
        {
            "wrote": payload.get("wrote"),
            "chosen": payload.get("chosen"),
            "llm": payload.get("llm"),
            "fps_claimed": False,
        }
    )
    return 0 if payload.get("chosen") else 1


def cmd_report(args: argparse.Namespace) -> int:
    path = write_report(Path(args.results))
    _print({"wrote": str(path)})
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

    records: list[dict] = []
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
        skip_record = skip_run_record(
            kind="fixture",
            path=str(fixture),
            reason=reason or "tensorrt unavailable",
            backend="tensorrt",
        )
        write_run_json(results / "runs" / f"{skip_record['run_id']}.json", skip_record)
        append_history(results / "history.jsonl", skip_record)
        records.append(skip_record)

    for spec in load_zoo():
        onnx_path = spec.onnx_path()
        if args.kind and spec.kind != args.kind:
            continue
        if not onnx_path.exists():
            skip_record = skip_run_record(
                kind=spec.kind,
                path=str(onnx_path),
                reason=f"{spec.kind} ONNX not present at {onnx_path}; run {' '.join(spec.export_cmd())}",
            )
            write_run_json(results / "runs" / f"{skip_record['run_id']}.json", skip_record)
            append_history(results / "history.jsonl", skip_record)
            records.append(skip_record)
            continue
        records.append(
            compile_and_benchmark(
                onnx_path,
                backend_name="ort_cpu",
                strategy_name=spec.default_strategy,
                warmup=max(3, args.warmup // 2),
                iters=max(5, args.iters // 2),
                results_dir=results,
                model_kind=spec.kind,
            )
        )

    host = probe_host()
    summary = {
        "host": host,
        "runs": records,
        "flatten_k": flatten_features(),
        "zoo_kinds": list(zoo_kinds()),
        "fps_claimed": False,
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


def cmd_formats(args: argparse.Namespace) -> int:
    _print(
        {
            "frontends": frontend_catalog(),
            "backends": list(known_backends()),
            "llm_providers": list(known_providers()),
            "ir": ["onnx"],
            "fps_claimed": False,
        }
    )
    return 0


def cmd_zoo(args: argparse.Namespace) -> int:
    rows = []
    for spec in load_zoo():
        onnx_path = spec.onnx_path()
        rows.append(
            {
                "kind": spec.kind,
                "task": spec.task,
                "family": spec.family,
                "onnx": spec.onnx,
                "onnx_present": onnx_path.is_file(),
                "export": " ".join(spec.export_cmd()),
                "native_strategy": spec.native_strategy,
                "default_strategy": spec.default_strategy,
                "imgsz": spec.imgsz,
                "uav_role": spec.uav_role,
                "justification": spec.justification,
            }
        )
    _print({"models": rows, "count": len(rows)})
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import subprocess

    kinds = [args.kind] if args.kind else list(zoo_kinds())
    codes = []
    for kind in kinds:
        spec = get_model(kind)
        cmd = spec.export_cmd(python=sys.executable)
        print("+", " ".join(cmd), file=sys.stderr)
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        codes.append(completed.returncode)
    # 0 if any export succeeded; 2 if all skipped/failed
    if any(code == 0 for code in codes):
        return 0
    return 2 if codes else 1


def cmd_matrix(args: argparse.Namespace) -> int:
    results = Path(args.results)
    kinds = [args.kind] if args.kind else None
    summary = run_zoo_matrix(
        kinds=kinds,
        warmup=args.warmup,
        iters=args.iters,
        results_dir=results,
        candidates=(args.candidates or None),
    )
    if args.fixture:
        fixture = Path(args.fixture)
        if fixture.is_file():
            summary.setdefault("models", []).insert(
                0,
                measure_path(
                    fixture,
                    kind="fixture",
                    task="fixture",
                    native_strategy="baseline",
                    warmup=args.warmup,
                    iters=args.iters,
                    results_dir=results,
                ),
            )
            wrote = results / "paper_matrix.json"
            wrote.write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
            )
            summary["wrote"] = str(wrote)
    _print(
        {
            "wrote": summary.get("wrote"),
            "model_count": len(summary.get("models", [])),
            "kinds": [row.get("kind") for row in summary.get("models", [])],
            "fps_claimed": False,
        }
    )
    measured = [
        row
        for row in summary.get("models", [])
        if (row.get("native") and row["native"].get("compile_ok"))
        or (
            row.get("chosen") and row["chosen"].get("compile_ok") is not False and row["chosen"].get("p50_ms")
        )
    ]
    return 0 if measured else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nnc", description="LLM-guided neural compiler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit-fixture", help="Write the tiny CNN ONNX fixture")
    emit.add_argument("--out", default=str(DEFAULT_FIXTURE))
    emit.add_argument("--kind", choices=("cnn", "depth"), default="cnn")
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

    plan_p = sub.add_parser("plan", help="Verified LLM/heuristic plan JSON")
    plan_p.add_argument("model")
    plan_p.add_argument("--kind", default="")
    plan_p.add_argument("--constraint", action="append", default=[])
    plan_p.add_argument("--show-prompt", action="store_true")
    plan_p.set_defaults(func=cmd_plan)

    opt_p = sub.add_parser("optimize", help="Compile and rank candidate plans")
    opt_p.add_argument("model")
    opt_p.add_argument("--kind", required=True)
    opt_p.add_argument("--task", default="detect")
    opt_p.add_argument("--candidates", default="default", choices=("default", "all"))
    opt_p.add_argument("--warmup", type=int, default=5)
    opt_p.add_argument("--iters", type=int, default=20)
    opt_p.add_argument("--results", default=str(DEFAULT_RESULTS))
    opt_p.set_defaults(func=cmd_optimize)

    report_p = sub.add_parser("report", help="Write report.md from paper_matrix.json")
    report_p.add_argument("--results", default=str(DEFAULT_RESULTS))
    report_p.set_defaults(func=cmd_report)

    infer = sub.add_parser("infer", help="Load an ORT CPU artifact and measure real latency")
    infer.add_argument("model")
    infer.add_argument("--graph-opt", default="extended", choices=("disable", "basic", "extended", "all"))
    infer.add_argument("--warmup", type=int, default=1)
    infer.add_argument("--iters", type=int, default=3)
    infer.set_defaults(func=cmd_infer)

    compile_p = sub.add_parser("compile", help="Compile and benchmark")
    compile_p.add_argument("model")
    compile_p.add_argument(
        "--backend", choices=known_backends() or ("ort_cpu", "tensorrt"), default="ort_cpu"
    )
    compile_p.add_argument("--strategy", default="baseline", choices=ALLOWED_STRATEGY_NAMES)
    compile_p.add_argument("--warmup", type=int, default=10)
    compile_p.add_argument("--iters", type=int, default=50)
    compile_p.add_argument("--results", default=str(DEFAULT_RESULTS))
    compile_p.add_argument("--kind", default="onnx")
    compile_p.set_defaults(func=cmd_compile)

    baseline = sub.add_parser("baseline", help="ORT CPU baseline JSON; TensorRT skipped unless NVIDIA exists")
    baseline.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    baseline.add_argument("--results", default=str(DEFAULT_RESULTS))
    baseline.add_argument("--kind", default="", help="If set, only this zoo kind plus the fixture")
    baseline.add_argument("--warmup", type=int, default=10)
    baseline.add_argument("--iters", type=int, default=50)
    baseline.set_defaults(func=cmd_baseline)

    probe = sub.add_parser("probe", help="Print host/PX4/ROS/Gazebo/NVIDIA facts")
    probe.set_defaults(func=cmd_probe)

    formats = sub.add_parser("formats", help="List frontends, backends, prompt engines, and GraphIR")
    formats.set_defaults(func=cmd_formats)

    zoo = sub.add_parser("zoo", help="List UAV companion models from experiments/zoo.yaml")
    zoo.set_defaults(func=cmd_zoo)

    export_p = sub.add_parser("export", help="Run the per-model ONNX export script (skip JSON on failure)")
    export_p.add_argument("--kind", default="", help="Zoo kind; omit to export every kind")
    export_p.set_defaults(func=cmd_export)

    matrix = sub.add_parser(
        "matrix",
        help="Native ORT baseline vs allowlisted strategy on each zoo ONNX (same host)",
    )
    matrix.add_argument("--kind", default="", help="Single zoo kind; omit for the full zoo")
    matrix.add_argument("--fixture", default="", help="Also measure a local ONNX fixture")
    matrix.add_argument("--results", default=str(DEFAULT_RESULTS))
    matrix.add_argument("--warmup", type=int, default=3)
    matrix.add_argument("--iters", type=int, default=8)
    matrix.add_argument("--candidates", default="", help="default|all to run optimize_model per zoo kind")
    matrix.set_defaults(func=cmd_matrix)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except UnknownStrategyError as exc:
        print(exc, file=sys.stderr)
        return 2
    except (SchemaError, FrontendSkip, UnknownFrontendError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
