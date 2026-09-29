#!/usr/bin/env python3
"""yolo_detector — real-time object detection with YOLO, as a ROS 2 node.

Subscribes: image_topic        (sensor_msgs/Image)
Publishes:  /detections        (vision_msgs/Detection2DArray)  boxes + labels + scores
            /detections/image  (sensor_msgs/Image)             the frame with boxes drawn
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model', 'yolo11n.pt')   # yolo11n.engine after TensorRT export
        self.declare_parameter('conf', 0.5)             # ignore boxes below 50% confidence

        self.model = YOLO(self.get_parameter('model').value)
        self.conf = self.get_parameter('conf').value
        self.bridge = CvBridge()

        # depth=1 + best effort: always process the NEWEST frame, drop old ones
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, image_qos)
        self.det_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.img_pub = self.create_publisher(Image, '/detections/image', 10)

        self.frames, self.t0 = 0, time.time()
        self.get_logger().info('YOLO loaded, waiting for images...')

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.model(frame, conf=self.conf, verbose=False)[0]

        # 1. Turn YOLO's output into a standard ROS message
        out = Detection2DArray()
        out.header = msg.header
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = result.names[int(box.cls)]
            hyp.hypothesis.score = float(box.conf)
            det.results.append(hyp)
            out.detections.append(det)
        self.det_pub.publish(out)

        # 2. Publish a picture with the boxes drawn, for rqt_image_view
        annotated = self.bridge.cv2_to_imgmsg(result.plot(), encoding='bgr8')
        annotated.header = msg.header
        self.img_pub.publish(annotated)

        # 3. Every 5 s, log the frame rate and what we can see
        self.frames += 1
        if time.time() - self.t0 > 5:
            seen = sorted({d.results[0].hypothesis.class_id for d in out.detections})
            fps = self.frames / (time.time() - self.t0)
            self.get_logger().info(f'{fps:.1f} FPS | seeing: {", ".join(seen) or "nothing"}')
            self.frames, self.t0 = 0, time.time()


def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
