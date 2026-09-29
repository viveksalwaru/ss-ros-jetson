#!/usr/bin/env python3
"""open_vocab_detector — detect ANY object you describe in words (YOLO-World).

Normal YOLO only knows its 80 training classes. YOLO-World takes a text prompt,
so you can ask for "red cup" or "yellow toy car" without retraining anything.

Subscribes: image_topic        (sensor_msgs/Image)
Publishes:  /detections        (vision_msgs/Detection2DArray)  same format as yolo_detector
            /detections/image  (sensor_msgs/Image)

Change what it looks for while it runs:
  ros2 param set /open_vocab_detector classes "['blue ball', 'shoe']"
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLOWorld


class OpenVocabDetector(Node):
    def __init__(self):
        super().__init__('open_vocab_detector')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model', 'yolov8s-worldv2.pt')
        self.declare_parameter('classes', ['person', 'red cup', 'backpack'])
        self.declare_parameter('conf', 0.25)       # open-vocab scores run lower than normal YOLO

        self.model = YOLOWorld(self.get_parameter('model').value)
        self.set_classes(self.get_parameter('classes').value)
        self.add_on_set_parameters_callback(self.on_params)
        self.bridge = CvBridge()

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, image_qos)
        self.det_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.img_pub = self.create_publisher(Image, '/detections/image', 10)

    def set_classes(self, classes):
        self.model.set_classes(list(classes))
        self.get_logger().info(f'Now looking for: {list(classes)}')

    def on_params(self, params):
        for param in params:
            if param.name == 'classes':
                self.set_classes(param.value)
        return SetParametersResult(successful=True)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        conf = self.get_parameter('conf').value
        result = self.model.predict(frame, conf=conf, verbose=False)[0]

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

        annotated = self.bridge.cv2_to_imgmsg(result.plot(), encoding='bgr8')
        annotated.header = msg.header
        self.img_pub.publish(annotated)


def main():
    rclpy.init()
    node = OpenVocabDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
