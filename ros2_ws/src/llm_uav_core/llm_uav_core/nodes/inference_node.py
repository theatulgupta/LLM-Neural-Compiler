"""ROS 2 node: Gazebo camera (or synthetic) -> ORT artifact -> detections + latency."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
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
        self.declare_parameter("model_kind", "yolov8n")
        self.declare_parameter("task", "detect")
        self.declare_parameter("graph_opt", "disable")
        self.declare_parameter("period_sec", 2.0)
        self.declare_parameter("source", "camera")
        self.declare_parameter("image_topic", "/nnc/camera/image_raw")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("max_hz", 10.0)

        self._status_pub = self.create_publisher(String, "/nnc/status", 10)
        self._latency_pub = self.create_publisher(Float32, "/nnc/latency_ms", 10)
        self._ok_pub = self.create_publisher(Bool, "/nnc/ok", 10)
        self._det_pub = None
        self._ann_pub = None
        self._artifact = None
        self._error: str | None = None
        self._frame_seq = 0
        self._dropped = 0
        self._busy = False
        self._last_stamp = 0.0
        self._load()
        source = str(self.get_parameter("source").value)
        if source == "camera":
            self._start_camera()
        else:
            period = float(self.get_parameter("period_sec").value)
            self.create_timer(max(0.2, period), self._tick_synthetic)
        self.get_logger().info("Inference node started.")

    def _load(self) -> None:
        _ensure_nnc_on_path()
        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        try:
            from nnc.artifact import load_artifact_with_sidecar, load_ort_artifact

            sidecar = Path(str(model_path) + ".json")
            if sidecar.is_file() or model_path.with_suffix(".json").is_file():
                self._artifact = load_artifact_with_sidecar(model_path)
            else:
                self._artifact = load_ort_artifact(
                    model_path, graph_opt=str(self.get_parameter("graph_opt").value)
                )
            self._error = None
            self.get_logger().info(
                f"Loaded ORT artifact {model_path} sha256={self._artifact.sha256[:12]} "
                f"shape={list(self._artifact.input_shape)}"
            )
        except Exception as exc:  # noqa: BLE001
            self._artifact = None
            self._error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"ORT artifact load failed: {self._error}")

    def _start_camera(self) -> None:
        from sensor_msgs.msg import Image
        from vision_msgs.msg import Detection2DArray

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        topic = str(self.get_parameter("image_topic").value)
        self.create_subscription(Image, topic, self._on_image, qos)
        self._det_pub = self.create_publisher(Detection2DArray, "/nnc/detections", 10)
        if bool(self.get_parameter("publish_annotated").value):
            self._ann_pub = self.create_publisher(Image, "/nnc/annotated", 10)

    def _image_to_rgb(self, msg) -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        encoding = str(msg.encoding).lower()
        try:
            from cv_bridge import CvBridge

            bgr = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return bgr[:, :, ::-1].copy()
        except Exception:  # noqa: BLE001
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if encoding == "rgb8":
                return raw.reshape((h, w, 3))
            if encoding == "bgr8":
                return raw.reshape((h, w, 3))[:, :, ::-1].copy()
            raise

    def _on_image(self, msg) -> None:
        if self._busy:
            self._dropped += 1
            return
        max_hz = float(self.get_parameter("max_hz").value)
        now = time.monotonic()
        if max_hz > 0 and (now - self._last_stamp) < (1.0 / max_hz):
            self._dropped += 1
            return
        self._busy = True
        self._last_stamp = now
        try:
            self._infer_image(msg)
        finally:
            self._busy = False

    def _infer_image(self, msg) -> None:
        if self._artifact is None:
            self._publish({"ok": False, "error": self._error, "source": "camera", "fps_claimed": False}, None, False)
            return
        t_all = time.perf_counter()
        rgb = self._image_to_rgb(msg)
        from nnc.artifact import infer_once
        from nnc.postprocess import decode
        from nnc.preprocess import letterbox

        imgsz = int(self.get_parameter("imgsz").value)
        t0 = time.perf_counter()
        nchw, meta = letterbox(rgb, imgsz, rgb=True)
        pre_ms = (time.perf_counter() - t0) * 1000.0
        feeds = {self._artifact.input_name: nchw}
        sample = infer_once(self._artifact, feeds)
        t1 = time.perf_counter()
        det = decode(
            str(self.get_parameter("task").value),
            list(sample.outputs or []),
            conf=float(self.get_parameter("conf").value),
            iou=float(self.get_parameter("iou").value),
        )
        post_ms = (time.perf_counter() - t1) * 1000.0
        self._frame_seq += 1
        self._publish_detections(msg, det, meta)
        e2e = (time.perf_counter() - t_all) * 1000.0
        payload = {
            "ok": True,
            "error": None,
            "source": "camera",
            "frame_seq": self._frame_seq,
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "ms": {"pre": pre_ms, "infer": sample.latency_ms, "post": post_ms, "e2e": e2e},
            "n_det": int(len(det.get("boxes", []))),
            "dropped_frames": self._dropped,
            "model": {
                "path": str(self._artifact.path),
                "sha256": self._artifact.sha256,
                "kind": str(self.get_parameter("model_kind").value),
                "plan_id": (self._artifact.plan or {}).get("plan_id") if self._artifact.plan else None,
            },
            "options": self._artifact.options.to_dict() if self._artifact.options else None,
            "fps_claimed": False,
        }
        self._publish(payload, sample.latency_ms, True)

    def _publish_detections(self, msg, det: dict, meta: dict) -> None:
        if self._det_pub is None:
            return
        from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

        out = Detection2DArray()
        out.header = msg.header
        pad_x, pad_y = meta.get("pad", (0, 0))
        scale = float(meta.get("scale") or 1.0)
        boxes = det.get("boxes")
        scores = det.get("scores")
        classes = det.get("classes")
        if boxes is None:
            self._det_pub.publish(out)
            return
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            # letterbox space -> original pixels
            x1 = (x1 - pad_x) / max(scale, 1e-6)
            x2 = (x2 - pad_x) / max(scale, 1e-6)
            y1 = (y1 - pad_y) / max(scale, 1e-6)
            y2 = (y2 - pad_y) / max(scale, 1e-6)
            item = Detection2D()
            item.header = msg.header
            item.bbox = BoundingBox2D()
            item.bbox.center.position.x = (x1 + x2) / 2.0
            item.bbox.center.position.y = (y1 + y2) / 2.0
            item.bbox.size_x = max(0.0, x2 - x1)
            item.bbox.size_y = max(0.0, y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cls))
            hyp.hypothesis.score = float(score)
            item.results.append(hyp)
            out.detections.append(item)
        self._det_pub.publish(out)

    def _tick_synthetic(self) -> None:
        source = str(self.get_parameter("source").value)
        if self._artifact is None:
            self._publish({"ok": False, "error": self._error, "source": source, "fps_claimed": False}, None, False)
            return
        try:
            from nnc.artifact import infer_once, synthetic_feed

            feeds = synthetic_feed(self._artifact)
            sample = infer_once(self._artifact, feeds)
            payload = {
                "ok": True,
                "error": None,
                "source": source,
                "model": {"path": str(self._artifact.path), "sha256": self._artifact.sha256},
                "latency_ms": sample.latency_ms,
                "fps_claimed": False,
            }
            self._publish(payload, sample.latency_ms, True)
        except Exception as exc:  # noqa: BLE001
            self._publish({"ok": False, "error": f"{type(exc).__name__}: {exc}", "source": source, "fps_claimed": False}, None, False)

    def _publish(self, payload: dict, latency_ms: float | None, ok: bool) -> None:
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
