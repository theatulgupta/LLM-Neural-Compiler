#!/usr/bin/env python3
"""Subscribe to PX4 local position and record whether x,y,z actually change.

Must run under ROS 2 Jazzy + px4_msgs (system interpreter, not the compiler venv).
Does not report success unless at least one message arrives.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _px4_qos():
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
    )


def _pick_topic(explicit: str | None) -> tuple[str | None, list[str]]:
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("nnc_topic_probe")
    names = node.get_topic_names_and_types()
    topics = sorted(name for name, _ in names)
    node.destroy_node()
    rclpy.shutdown()
    if explicit:
        return explicit, topics
    matches = [name for name in topics if "vehicle_local_position" in name]
    if not matches:
        return None, topics
    preferred = [name for name in matches if name.endswith("_v1")]
    return (preferred[0] if preferred else matches[0]), topics


def record(topic: str, seconds: float) -> dict:
    import rclpy
    from px4_msgs.msg import VehicleLocalPosition
    from rclpy.node import Node

    samples: list[dict] = []

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("nnc_position_recorder")
            self.create_subscription(VehicleLocalPosition, topic, self._cb, _px4_qos())

        def _cb(self, msg: VehicleLocalPosition) -> None:
            samples.append(
                {
                    "t": time.time(),
                    "timestamp": int(getattr(msg, "timestamp", 0)),
                    "x": float(msg.x),
                    "y": float(msg.y),
                    "z": float(msg.z),
                    "xy_valid": bool(getattr(msg, "xy_valid", False)),
                    "z_valid": bool(getattr(msg, "z_valid", False)),
                }
            )

    rclpy.init()
    node = Recorder()
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    zs = [s["z"] for s in samples]
    changed = False
    if len(samples) >= 2:
        span = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
        )
        # Grounded SITL still jitters; require a real span, not float noise.
        changed = span > 1e-4
    return {
        "topic": topic,
        "seconds": seconds,
        "message_count": len(samples),
        "changed_xyz": changed,
        "x": {"min": min(xs) if xs else None, "max": max(xs) if xs else None},
        "y": {"min": min(ys) if ys else None, "max": max(ys) if ys else None},
        "z": {"min": min(zs) if zs else None, "max": max(zs) if zs else None},
        "samples_head": samples[:5],
        "samples_tail": samples[-5:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record PX4 local position samples")
    parser.add_argument("--topic", default="")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    topic, topics = _pick_topic(args.topic or None)
    result = {
        "ros_topics_head": topics[:40],
        "topic_count": len(topics),
        "selected_topic": topic,
    }
    if topic is None:
        result["ok"] = False
        result["reason"] = "no vehicle_local_position topic on the ROS graph"
    else:
        sampled = record(topic, args.seconds)
        result.update(sampled)
        result["ok"] = sampled["message_count"] > 0
        if not result["ok"]:
            result["reason"] = f"subscribed to {topic} but received 0 messages"
        elif not sampled["changed_xyz"]:
            result["reason"] = (
                f"received {sampled['message_count']} messages on {topic} "
                "but x,y,z did not change beyond 1e-4 m"
            )
        else:
            result["reason"] = f"x,y,z changed on {topic}"
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0 if result.get("ok") and result.get("changed_xyz") else 3


if __name__ == "__main__":
    raise SystemExit(main())
