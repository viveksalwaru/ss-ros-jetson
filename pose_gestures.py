#!/usr/bin/env python3
"""pose_gestures — body-gesture commands with a YOLO pose model.

One hand raised above the shoulder -> "go"
Both hands raised                  -> "stop"
Hands down                         -> "idle"

Subscribes: image_topic     (sensor_msgs/Image)
Publishes:  /gesture        (std_msgs/String)   only when the gesture changes
            /gesture/image  (sensor_msgs/Image) skeleton drawn on the frame
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

# COCO body keypoint numbers used by YOLO pose models
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10


class PoseGestures(Node):
    def __init__(self):
        super().__init__('pose_gestures')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model', 'yolo11n-pose.pt')
        self.declare_parameter('hold_frames', 5)   # frames a gesture must be held before it counts

        self.model = YOLO(self.get_parameter('model').value)
        self.hold = self.get_parameter('hold_frames').value
        self.bridge = CvBridge()

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, image_qos)
        self.pub = self.create_publisher(String, '/gesture', 10)
        self.img_pub = self.create_publisher(Image, '/gesture/image', 10)

        self.candidate, self.count, self.current = None, 0, None
        self.get_logger().info('Raise one hand for GO, both hands for STOP')

    def read_gesture(self, result):
        if result.keypoints is None or len(result.boxes) == 0:
            return 'idle'
        # Several people? Use the biggest one (closest to the camera).
        areas = result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]
        i = int(areas.argmax())
        xy = result.keypoints.xy[i].tolist()          # 17 [x, y] points
        conf = result.keypoints.conf[i].tolist()      # how sure it is about each point

        def hand_up(wrist, shoulder):
            # Image y grows DOWNWARD, so "above" means a smaller y value
            return (conf[wrist] > 0.5 and conf[shoulder] > 0.5
                    and xy[wrist][1] < xy[shoulder][1] - 20)

        left, right = hand_up(L_WRIST, L_SHOULDER), hand_up(R_WRIST, R_SHOULDER)
        if left and right:
            return 'stop'
        if left or right:
            return 'go'
        return 'idle'

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.model(frame, verbose=False)[0]
        gesture = self.read_gesture(result)

        # Debounce: a gesture must be seen for `hold` frames in a row
        if gesture == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = gesture, 1
        if self.count == self.hold and gesture != self.current:
            self.current = gesture
            self.pub.publish(String(data=gesture))
            self.get_logger().info(f'Gesture: {gesture}')

        annotated = self.bridge.cv2_to_imgmsg(result.plot(), encoding='bgr8')
        annotated.header = msg.header
        self.img_pub.publish(annotated)


def main():
    rclpy.init()
    node = PoseGestures()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
