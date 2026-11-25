"""ROS 2 node that runs compiled ORT artifacts on live camera or high-rate synthetic frames."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

try:
    from sensor_msgs.msg import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


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


def _resize_chw_from_image_msg(msg: Any, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    channels = 3 if "rgb" in msg.encoding.lower() or "bgr" in msg.encoding.lower() else 1
    arr = raw.reshape((h, w, channels)).astype(np.float32) / 255.0
    if "bgr" in msg.encoding.lower() and channels == 3:
        arr = arr[:, :, ::-1]
    th, tw = target_hw
    if h != th or w != tw:
        # Lightweight nearest-neighbor resize avoids hard OpenCV/cv_bridge version mismatches.
        y_idx = (np.linspace(0, h - 1, th)).astype(np.int64)
        x_idx = (np.linspace(0, w - 1, tw)).astype(np.int64)
        arr = arr[y_idx][:, x_idx]
    chw = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw, dtype=np.float32)


class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("inference_node")
        self.declare_parameter("model_path", str(_repo_root() / "experiments" / "models" / "yolov8n.onnx"))
        self.declare_parameter("graph_opt", "extended")
        self.declare_parameter("execution_provider", "CPUExecutionProvider")
        self.declare_parameter("intra_op_num_threads", 0)
        self.declare_parameter("inter_op_num_threads", 0)
        self.declare_parameter("period_sec", 0.05)  # 20 Hz default for realistic UAV perception
        self.declare_parameter("deadline_ms", 80.0)
        self.declare_parameter("source", "synthetic")  # "camera" or "synthetic"
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("log_json_path", str(_repo_root() / "experiments" / "results" / "inference_ros_live.json"))

        self._status_pub = self.create_publisher(String, "/nnc/status", 10)
        self._latency_pub = self.create_publisher(Float32, "/nnc/latency_ms", 10)
        self._ok_pub = self.create_publisher(Bool, "/nnc/ok", 10)

        self._artifact = None
        self._error: str | None = None
        self._samples_ms: list[float] = []
        self._missed_deadlines = 0
        self._load()

        source = str(self.get_parameter("source").value).lower()
        cam_topic = str(self.get_parameter("camera_topic").value)
        if source == "camera" and Image is not None:
            self.create_subscription(Image, cam_topic, self._on_image, 5)
            self.get_logger().info(f"InferenceNode subscribed to camera topic {cam_topic}")
        else:
            period = max(0.02, float(self.get_parameter("period_sec").value))
            self.create_timer(period, self._tick)
            self.get_logger().info(f"InferenceNode started synthetic timer at {1.0/period:.1f} Hz")

    def _load(self) -> None:
        _ensure_nnc_on_path()
        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        graph_opt = str(self.get_parameter("graph_opt").value)
        ep = str(self.get_parameter("execution_provider").value)
        intra = int(self.get_parameter("intra_op_num_threads").value)
        inter = int(self.get_parameter("inter_op_num_threads").value)
        try:
            from nnc.backends.ort_cpu import OrtCpuBackend
            from nnc.artifact import LoadedArtifact
            from compiler.utils import sha256_file

            if not model_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            model_bytes = model_path.read_bytes()
            compiled = OrtCpuBackend().compile(
                model_bytes,
                graph_opt=graph_opt,
                execution_provider=ep,
                intra_op_num_threads=intra,
                inter_op_num_threads=inter,
            )
            sess_in = compiled.session.get_inputs()[0]
            shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in sess_in.shape)
            self._artifact = LoadedArtifact(
                path=model_path,
                sha256=sha256_file(model_path),
                bytes_len=len(model_bytes),
                graph_opt=graph_opt,
                compiled=compiled,
                input_name=sess_in.name,
                input_shape=shape,
                output_names=compiled.output_names,
                providers=compiled.providers,
            )
            self._error = None
            self.get_logger().info(
                f"Loaded ORT artifact {model_path} sha256={self._artifact.sha256[:12]} "
                f"shape={list(self._artifact.input_shape)} opt={graph_opt} ep={ep}"
            )
        except Exception as exc:  # noqa: BLE001
            self._artifact = None
            self._error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"ORT artifact load failed: {self._error}")

    def _run_feeds(self, feeds: dict[str, np.ndarray], source: str) -> None:
        from nnc.artifact import infer_once

        t0 = time.perf_counter()
        sample = infer_once(self._artifact, feeds)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        deadline = float(self.get_parameter("deadline_ms").value)
        self._samples_ms.append(sample.latency_ms)
        if sample.latency_ms > deadline:
            self._missed_deadlines += 1

        ordered = sorted(self._samples_ms)
        n = len(ordered)
        p50 = ordered[min(n - 1, int(round(0.50 * (n - 1))))]
        p95 = ordered[min(n - 1, int(round(0.95 * (n - 1))))]
        p99 = ordered[min(n - 1, int(round(0.99 * (n - 1))))]
        payload = {
            "ok": True,
            "error": None,
            "source": source,
            "n_samples": n,
            "deadline_ms": deadline,
            "missed_deadlines": self._missed_deadlines,
            "missed_deadline_pct": (100.0 * self._missed_deadlines / n),
            "latency_ms": sample.latency_ms,
            "wall_ms": wall_ms,
            "stats": {
                "mean": sum(ordered) / n,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "min": ordered[0],
                "max": ordered[-1],
            },
            "providers": list(self._artifact.providers),
            "graph_opt": self._artifact.graph_opt,
            "fps_claimed": False,
        }
        self._publish(payload, latency_ms=sample.latency_ms, ok=True)
        if n % 50 == 0:
            log_path = Path(str(self.get_parameter("log_json_path").value))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _on_image(self, msg: Any) -> None:
        if self._artifact is None:
            self._publish({"ok": False, "error": self._error, "source": "camera", "fps_claimed": False}, None, False)
            return
        try:
            shape = self._artifact.input_shape
            th, tw = (shape[2], shape[3]) if len(shape) == 4 else (640, 640)
            tensor = _resize_chw_from_image_msg(msg, (th, tw))
            self._run_feeds({self._artifact.input_name: tensor}, source="camera")
        except Exception as exc:  # noqa: BLE001
            self._publish({"ok": False, "error": str(exc), "source": "camera", "fps_claimed": False}, None, False)

    def _tick(self) -> None:
        source = str(self.get_parameter("source").value)
        if self._artifact is None:
            self._publish({"ok": False, "error": self._error, "source": source, "fps_claimed": False}, None, False)
            return
        try:
            from nnc.artifact import synthetic_feed

            self._run_feeds(synthetic_feed(self._artifact), source=source)
        except Exception as exc:  # noqa: BLE001
            self._publish({"ok": False, "error": str(exc), "source": source, "fps_claimed": False}, None, False)

    def _publish(self, payload: dict[str, Any], latency_ms: float | None, ok: bool) -> None:
        status_msg = String()
        status_msg.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(status_msg)

        ok_msg = Bool()
        ok_msg.data = bool(ok)
        self._ok_pub.publish(ok_msg)

        if latency_ms is not None:
            lat_msg = Float32()
            lat_msg.data = float(latency_ms)
            self._latency_pub.publish(lat_msg)


def main(args: list[str] | None = None) -> None:
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
