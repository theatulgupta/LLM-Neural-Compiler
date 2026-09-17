#!/usr/bin/env python3
"""Record Gazebo camera frames from ROS 2. Must run under system Python + rclpy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _to_rgb(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    encoding = str(msg.encoding).lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in {"rgb8", "bgr8"}:
        img = raw.reshape((h, w, 3))
        if encoding == "bgr8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    if encoding in {"rgba8", "bgra8"}:
        img = raw.reshape((h, w, 4))[:, :, :3]
        if encoding == "bgra8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    raise RuntimeError(f"unsupported encoding {msg.encoding!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record /nnc/camera/image_raw to npz")
    parser.add_argument("--topic", default="/nnc/camera/image_raw")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--world", default="default")
    parser.add_argument("--airframe", default="gz_x500_mono_cam")
    parser.add_argument("--render", default="")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "experiments" / "calib" / "gz_frames.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    frames: list[np.ndarray] = []
    stamps: list[float] = []
    encoding = ""
    width = 0
    height = 0
    seen = 0

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("nnc_record_frames")
            self.create_subscription(Image, args.topic, self._cb, qos)

        def _cb(self, msg: Image) -> None:
            nonlocal encoding, width, height, seen
            seen += 1
            if seen % max(1, args.every) != 0:
                return
            if len(frames) >= args.n:
                return
            rgb = _to_rgb(msg)
            frames.append(rgb)
            stamps.append(float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9)
            encoding = str(msg.encoding)
            width, height = int(msg.width), int(msg.height)

    rclpy.init()
    node = Recorder()
    deadline = time.monotonic() + float(args.seconds)
    try:
        while rclpy.ok() and len(frames) < args.n and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    payload = {
        "ok": len(frames) >= args.n,
        "count": len(frames),
        "wanted": args.n,
        "topic": args.topic,
        "encoding": encoding,
        "width": width,
        "height": height,
        "world": args.world,
        "airframe": args.airframe,
        "render_engine": args.render or None,
        "fps_claimed": False,
    }
    if not frames:
        payload["error"] = "no frames received"
        meta = out.with_suffix(".json")
        meta.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload), file=sys.stderr)
        return 3

    stacked = np.stack(frames, axis=0)
    np.savez_compressed(out, frames=stacked, stamps=np.asarray(stamps, dtype=np.float64), encoding=np.asarray(encoding))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    payload["path"] = str(out)
    payload["sha256"] = digest
    payload["shape"] = list(stacked.shape)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
