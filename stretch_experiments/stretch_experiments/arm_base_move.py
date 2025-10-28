#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import stretch_body.robot
import time


class SimplePickNode(Node):
    def __init__(self):
        super().__init__('simple_pick_node')

        # Initialize robot
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        self.get_logger().info("Stretch robot initialized.")

        # Run the sequence
        self.pick_sequence()

    def pick_sequence(self):
        # Start
        self.get_logger().info(" Starting pick test...")

        # Step 1: Drive forward to object
        drive_distance = 0.75  # 1 meter forward
        self.get_logger().info(f"Step 1: Driving forward {drive_distance:.2f} m...")
        self.robot.base.translate_by(drive_distance)
        self.robot.push_command()
        time.sleep(7)   # wait for base to stop

        # Step 2: Position arm & wrist
        self.get_logger().info("Step 2: Positioning arm and wrist...")
        self.robot.lift.move_to(0.15)                  # 15 cm above ground
        self.robot.arm.move_to(0.3)                    # extend ~30 cm
        self.robot.end_of_arm.move_to('wrist_pitch', -0.6)  # tilt wrist down
        self.robot.end_of_arm.move_to('stretch_gripper', 100)  # open gripper
        self.robot.push_command()
        time.sleep(7)   # wait for arm to move + gripper to open

        # Step 3: Grab object
        self.get_logger().info("Step 3: Closing gripper to grab object...")
        self.robot.end_of_arm.move_to('stretch_gripper', 0)   # close gripper
        self.robot.push_command()
        time.sleep(2)

        # Step 4: Lift slightly
        self.get_logger().info("Step 4: Lifting object slightly...")
        self.robot.lift.move_to(0.3)     # raise ~30 cm
        self.robot.arm.move_to(0.2)      # retract arm a bit
        self.robot.push_command()
        time.sleep(3)

        # Done
        self.get_logger().info(" Pick sequence complete!")

    def destroy_node(self):
        self.robot.stop()
        self.robot.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimplePickNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
