#!/usr/bin/env python3
"""detection_follower — drive toward one kind of object that YOLO can see.

The same idea as follower_node in the object-following lesson, but the "eyes"
are now a neural network, so it can follow a bottle, a person, a cup...

Subscribes: /detections (vision_msgs/Detection2DArray)  from yolo_detector
Publishes:  cmd_topic   (geometry_msgs/Twist)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray


def clamp(value, limit):
    return max(-limit, min(limit, value))


class DetectionFollower(Node):
    def __init__(self):
        super().__init__('detection_follower')
        self.declare_parameter('target', 'bottle')      # any COCO class: person, cup, sports ball...
        self.declare_parameter('image_width', 640)      # width of the camera image in pixels
        self.declare_parameter('target_width', 150)     # box width (px) that means "close enough"
        self.declare_parameter('kp_turn', 0.004)        # rad/s per pixel off-centre
        self.declare_parameter('kp_drive', 0.003)       # m/s per pixel of size error
        self.declare_parameter('max_linear', 0.15)      # m/s  (keep it slow indoors)
        self.declare_parameter('max_angular', 0.6)      # rad/s
        self.declare_parameter('cmd_topic', '/cmd_vel')

        self.cmd_pub = self.create_publisher(Twist, self.p('cmd_topic'), 10)
        self.create_subscription(Detection2DArray, '/detections', self.on_detections, 10)

        self.last_seen = self.get_clock().now()
        self.stopped = True
        self.create_timer(0.2, self.watchdog)
        self.get_logger().info(f"Following '{self.p('target')}' -> {self.p('cmd_topic')}")

    def p(self, name):
        return self.get_parameter(name).value

    def on_detections(self, msg):
        matches = [d for d in msg.detections
                   if d.results and d.results[0].hypothesis.class_id == self.p('target')]
        if not matches:
            return

        # Several matches? Follow the biggest box, which is usually the closest.
        best = max(matches, key=lambda d: d.bbox.size_x * d.bbox.size_y)

        error_x = best.bbox.center.position.x - self.p('image_width') / 2   # + means target is right
        error_size = self.p('target_width') - best.bbox.size_x              # + means too far away

        cmd = Twist()
        cmd.angular.z = clamp(-self.p('kp_turn') * error_x, self.p('max_angular'))
        cmd.linear.x = clamp(self.p('kp_drive') * error_size, self.p('max_linear'))
        self.cmd_pub.publish(cmd)

        self.last_seen = self.get_clock().now()
        self.stopped = False

    def watchdog(self):
        # Target lost for more than 1 s -> stop. Never leave a robot driving blind.
        lost = (self.get_clock().now() - self.last_seen).nanoseconds > 1e9
        if lost and not self.stopped:
            self.cmd_pub.publish(Twist())
            self.stopped = True
            self.get_logger().info('Target lost, stopping')


def main():
    rclpy.init()
    node = DetectionFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # stop the wheels on Ctrl+C
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
