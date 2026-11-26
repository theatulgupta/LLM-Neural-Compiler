"""ROS 2 node that replays COCO128 images onto /camera/image_raw at 20 Hz using pure PIL + NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageReplayerNode(Node):
    def __init__(self) -> None:
        super().__init__("image_replayer_node")
        default_dir = str(Path.home() / "LLM-Neural-Compiler" / "experiments" / "data" / "coco128" / "images" / "train2017")
        self.declare_parameter("image_dir", default_dir)
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("period_sec", 0.05)  # 20 Hz
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 640)

        img_dir = Path(str(self.get_parameter("image_dir").value)).expanduser()
        self._paths = sorted(img_dir.glob("*.jpg"))
        if not self._paths:
            raise FileNotFoundError(f"No .jpg images found in {img_dir}")

        self._idx = 0
        self._w = int(self.get_parameter("width").value)
        self._h = int(self.get_parameter("height").value)
        topic = str(self.get_parameter("topic").value)
        self._pub = self.create_publisher(Image, topic, 5)
        period = max(0.02, float(self.get_parameter("period_sec").value))
        self.create_timer(period, self._tick)
        self.get_logger().info(f"ImageReplayerNode publishing {len(self._paths)} images on {topic} at {1.0/period:.1f} Hz")

    def _tick(self) -> None:
        path = self._paths[self._idx % len(self._paths)]
        self._idx += 1
        with PilImage.open(path) as img:
            rgb = img.convert("RGB").resize((self._w, self._h), PilImage.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.uint8)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.height = self._h
        msg.width = self._w
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = self._w * 3
        msg.data = arr.tobytes()
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ImageReplayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
