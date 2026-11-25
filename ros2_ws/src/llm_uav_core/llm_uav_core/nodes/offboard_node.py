"""PX4 Offboard flight node for ROS 2 Jazzy: arms, switches to Offboard, takes off to 5m, and flies a square mission."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition


class OffboardFlightNode(Node):
    def __init__(self) -> None:
        super().__init__("offboard_node")
        self.declare_parameter("altitude_m", 5.0)
        self.declare_parameter("square_size_m", 4.0)
        self.declare_parameter("step_period_sec", 6.0)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._mode_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self._setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self._cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._pos_cb, qos)

        self._counter = 0
        self._armed = False
        self._pos = [0.0, 0.0, 0.0]
        self._start_time = self.get_clock().now()
        # 10 Hz control heartbeat required by PX4 (> 2 Hz)
        self.create_timer(0.1, self._tick)
        self.get_logger().info("OffboardFlightNode started (waiting for setpoint pre-stream).")

    def _pos_cb(self, msg: VehicleLocalPosition) -> None:
        self._pos = [float(msg.x), float(msg.y), float(msg.z)]

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)

    def _send_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._now_us()
        self._cmd_pub.publish(msg)

    def _current_waypoint(self) -> list[float]:
        alt = -abs(float(self.get_parameter("altitude_m").value))
        side = float(self.get_parameter("square_size_m").value)
        step_sec = max(2.0, float(self.get_parameter("step_period_sec").value))
        elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if elapsed < 6.0:
            return [0.0, 0.0, alt]
        phase = int((elapsed - 6.0) // step_sec) % 4
        waypoints = [
            [side, 0.0, alt],
            [side, side, alt],
            [0.0, side, alt],
            [0.0, 0.0, alt],
        ]
        return waypoints[phase]

    def _tick(self) -> None:
        mode = OffboardControlMode()
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        mode.timestamp = self._now_us()
        self._mode_pub.publish(mode)

        wp = self._current_waypoint()
        sp = TrajectorySetpoint()
        sp.position = [float(wp[0]), float(wp[1]), float(wp[2])]
        sp.yaw = float(math.atan2(wp[1], wp[0])) if (wp[0] or wp[1]) else 0.0
        sp.timestamp = self._now_us()
        self._setpoint_pub.publish(sp)

        if self._counter == 15:
            self._send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self._send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            self._armed = True
            self.get_logger().info("Sent Offboard mode + Arm command.")
        self._counter += 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OffboardFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
