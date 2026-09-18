import rclpy
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


class TelemetryNode(Node):
    def __init__(self):
        super().__init__("telemetry_node")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.declare_parameter("topic", "/fmu/out/vehicle_local_position_v1")
        topic = str(self.get_parameter("topic").value)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.msg_count = 0
        self.xy_valid = False
        self.z_valid = False
        self.last_timestamp = 0

        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            topic,
            self.position_callback,
            qos_profile,
        )

        self.timer = self.create_timer(1.0, self.print_position)
        self.get_logger().info(f"Telemetry node started. topic={topic}")

    def position_callback(self, msg):
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.z = float(msg.z)
        self.msg_count += 1
        self.xy_valid = bool(getattr(msg, "xy_valid", False))
        self.z_valid = bool(getattr(msg, "z_valid", False))
        self.last_timestamp = int(getattr(msg, "timestamp", 0))

    def print_position(self):
        self.get_logger().info(
            f"Position -> "
            f"X={self.x:.4f}, "
            f"Y={self.y:.4f}, "
            f"Z={self.z:.4f} "
            f"msgs={self.msg_count} "
            f"xy_valid={self.xy_valid} z_valid={self.z_valid} "
            f"t={self.last_timestamp}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = TelemetryNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
