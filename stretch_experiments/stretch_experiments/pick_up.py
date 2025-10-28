#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import stretch_body.robot
import time


class PickAndLiftNode(Node):
    def __init__(self):
        super().__init__('pick_and_lift_node')

        # Initialize the robot
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        self.get_logger().info("Stretch robot initialized.")

        # Run the pick-up sequence once
        self.pick_and_lift()


    def pick_and_lift(self):
        # 🚀 Starting sequence
        self.get_logger().info("🚀 Starting pick-and-lift sequence...")

        # # Step 1: Drive forward a bit to get closer to the object
        # self.get_logger().info("Step 1: Driving forward 30cm...")
        # self.robot.base.translate_by(0.3)

        # Step 2: Lower lift and extend arm toward the floor
        self.get_logger().info("Step 2: Lowering lift to 10cm and extending arm 40cm...")
        self.robot.lift.move_to(0.1)
        self.robot.arm.move_to(0.2)
        self.robot.end_of_arm.move_to('wrist_pitch', -1.5)

        # Step 3: Close gripper to grasp object
        self.get_logger().info("Step 3: Closing gripper to grasp object...")
        self.robot.end_of_arm.move_to('stretch_gripper', 0)

        # Execute grasp sequence
        self.robot.push_command()
        self.get_logger().info("Commands sent — waiting for grasp to complete...")
        time.sleep(4)

        # Step 4: Lift the object up
        self.get_logger().info("Step 4: Raising lift to 30cm and retracting arm slightly...")
        self.robot.lift.move_to(0.3)
        self.robot.arm.move_to(0.2)

        # Execute lift sequence
        self.robot.push_command()
        self.get_logger().info("Sent lift commands.")

        # ✅ Final confirmation
        self.get_logger().info("✅ Pick-and-lift sequence complete!")


    def destroy_node(self):
        # Safely stop robot before shutdown
        self.robot.stop()
        self.robot.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PickAndLiftNode()

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

