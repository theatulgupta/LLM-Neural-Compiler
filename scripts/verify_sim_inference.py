#!/usr/bin/env python3
"""Collect camera + inference + PX4 evidence. Exit non-zero if no inferences."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from vision_msgs.msg import Detection2DArray

    try:
        from px4_msgs.msg import VehicleLocalPosition
    except ImportError:
        VehicleLocalPosition = None

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "experiments" / "results" / "sim_inference.json"

    frames = 0
    statuses: list[dict] = []
    detections_total = 0
    classes_seen: set[str] = set()
    positions = 0
    xyz = []

    qos_be = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class Probe(Node):
        def __init__(self) -> None:
            super().__init__("nnc_verify_sim")
            self.create_subscription(Image, "/nnc/camera/image_raw", self._img, qos_be)
            self.create_subscription(String, "/nnc/status", self._status, 10)
            self.create_subscription(Detection2DArray, "/nnc/detections", self._det, 10)
            if VehicleLocalPosition is not None:
                self.create_subscription(
                    VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._pos, qos_be
                )

        def _img(self, _msg) -> None:
            nonlocal frames
            frames += 1

        def _status(self, msg: String) -> None:
            try:
                statuses.append(json.loads(msg.data))
            except json.JSONDecodeError:
                pass

        def _det(self, msg: Detection2DArray) -> None:
            nonlocal detections_total
            detections_total += len(msg.detections)
            for item in msg.detections:
                for hyp in item.results:
                    classes_seen.add(str(hyp.hypothesis.class_id))

        def _pos(self, msg) -> None:
            nonlocal positions
            positions += 1
            xyz.append((float(msg.x), float(msg.y), float(msg.z)))

    rclpy.init()
    node = Probe()
    deadline = time.monotonic() + float(args.seconds)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    ok_status = [row for row in statuses if row.get("ok")]
    e2e = []
    infer = []
    for row in ok_status:
        ms = row.get("ms")
        if isinstance(ms, dict):
            if "e2e" in ms:
                e2e.append(float(ms["e2e"]))
            if "infer" in ms:
                infer.append(float(ms["infer"]))
        elif row.get("latency_ms") is not None:
            infer.append(float(row["latency_ms"]))

    def stats(samples: list[float]) -> dict:
        if not samples:
            return {}
        ordered = sorted(samples)
        return {"p50": ordered[len(ordered) // 2], "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], "n": len(ordered)}

    changed = False
    if len(xyz) >= 2:
        changed = xyz[0] != xyz[-1]
    payload = {
        "ok": len(ok_status) > 0,
        "frames_rx": frames,
        "inferences": len(ok_status),
        "achieved_hz": (len(ok_status) / args.seconds) if args.seconds else 0.0,
        "e2e_ms": stats(e2e),
        "infer_ms": stats(infer),
        "detections_total": detections_total,
        "classes_seen": sorted(classes_seen),
        "position_samples": positions,
        "xyz_changed": changed,
        "artifact": (ok_status[-1].get("model") if ok_status else None),
        "fps_claimed": False,
        "seconds": args.seconds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["inferences"] > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
