#!/usr/bin/env python3
"""camera_pub — publish a USB webcam as a ROS 2 image topic.

Use this to test the AI nodes on a laptop or a Jetson without the robot's camera.
Publishes: /image_raw (sensor_msgs/Image)
"""
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPub(Node):
    def __init__(self):
        super().__init__('camera_pub')
        self.declare_parameter('device', 0)            # 0 = /dev/video0
        self.declare_parameter('topic', '/image_raw')
        self.declare_parameter('fps', 15.0)

        device = self.get_parameter('device').value
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open camera {device} (try device:=1)')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.get_parameter('topic').value, 10)
        self.create_timer(1.0 / self.get_parameter('fps').value, self.tick)
        self.get_logger().info(f'Publishing camera {device} on {self.get_parameter("topic").value}')

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('No frame from camera', throttle_duration_sec=2.0)
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
