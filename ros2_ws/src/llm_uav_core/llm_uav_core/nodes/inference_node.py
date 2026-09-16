"""ROS 2 node that loads a real ORT CPU artifact and publishes measured latency.

Does not invent FPS. If the ONNX file is missing, the node reports the error
and stays up so the rest of the stack can still run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = []
    if len(here.parents) > 5:
        candidates.append(here.parents[5])
    candidates.append(Path.home() / "LLM-Neural-Compiler")
    for candidate in candidates:
        if (candidate / "src" / "nnc" / "artifact.py").is_file():
            return candidate
    return candidates[-1]


def _ensure_nnc_on_path() -> None:
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site = root / ".venv" / "lib" / version / "site-packages"
        if venv_site.is_dir() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))



class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("inference_node")
        self.declare_parameter("model_path", str(_repo_root() / "experiments" / "models" / "yolov8n.onnx"))
        self.declare_parameter("graph_opt", "extended")
        self.declare_parameter("period_sec", 2.0)
        self.declare_parameter("source", "synthetic")

        self._status_pub = self.create_publisher(String, "/nnc/status", 10)
        self._latency_pub = self.create_publisher(Float32, "/nnc/latency_ms", 10)
        self._ok_pub = self.create_publisher(Bool, "/nnc/ok", 10)

        self._artifact = None
        self._error: str | None = None
        self._load()

        period = float(self.get_parameter("period_sec").value)
        self.create_timer(max(0.2, period), self._tick)
        self.get_logger().info("Inference node started.")

    def _load(self) -> None:
        _ensure_nnc_on_path()
        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        graph_opt = str(self.get_parameter("graph_opt").value)
        try:
            from nnc.artifact import load_ort_artifact

            self._artifact = load_ort_artifact(model_path, graph_opt=graph_opt)
            self._error = None
            self.get_logger().info(
                f"Loaded ORT artifact {model_path} sha256={self._artifact.sha256[:12]} "
                f"shape={list(self._artifact.input_shape)} opt={graph_opt}"
            )
        except Exception as exc:  # noqa: BLE001 — publish the real load failure
            self._artifact = None
            self._error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"ORT artifact load failed: {self._error}")

    def _tick(self) -> None:
        source = str(self.get_parameter("source").value)
        if self._artifact is None:
            payload = {
                "ok": False,
                "error": self._error,
                "source": source,
                "fps_claimed": False,
            }
            self._publish(payload, latency_ms=None, ok=False)
            return
        try:
            from nnc.artifact import infer_once, synthetic_feed

            feeds = synthetic_feed(self._artifact)
            t0 = time.perf_counter()
            sample = infer_once(self._artifact, feeds)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "ok": True,
                "error": None,
                "source": source,
                "model": {
                    "path": str(self._artifact.path),
                    "sha256": self._artifact.sha256,
                    "bytes": self._artifact.bytes_len,
                },
                "providers": list(self._artifact.providers),
                "graph_opt": self._artifact.graph_opt,
                "input_shape": list(self._artifact.input_shape),
                "output_shapes": [list(s) for s in sample.output_shapes],
                "latency_ms": sample.latency_ms,
                "wall_ms": wall_ms,
                "throughput_ips": (1000.0 / sample.latency_ms) if sample.latency_ms > 0 else 0.0,
                "fps_claimed": False,
            }
            self._publish(payload, latency_ms=sample.latency_ms, ok=True)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "source": source,
                "model_path": str(self._artifact.path),
                "fps_claimed": False,
            }
            self._publish(payload, latency_ms=None, ok=False)

    def _publish(self, payload: dict, *, latency_ms: float | None, ok: bool) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(msg)
        ok_msg = Bool()
        ok_msg.data = ok
        self._ok_pub.publish(ok_msg)
        if latency_ms is not None:
            lat = Float32()
            lat.data = float(latency_ms)
            self._latency_pub.publish(lat)
        self.get_logger().info(msg.data[:500])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
