import rclpy
from rclpy.node import Node

from px4_msgs.msg import VehicleLocalPosition

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
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

        # Latest position
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # Subscriber
        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos_profile,
        )

        # Print once every second
        self.timer = self.create_timer(1.0, self.print_position)

        self.get_logger().info("Telemetry node started.")

    def position_callback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z

    def print_position(self):
        self.get_logger().info(
            f"Position -> "
            f"X={self.x:.2f}, "
            f"Y={self.y:.2f}, "
            f"Z={self.z:.2f}"
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
