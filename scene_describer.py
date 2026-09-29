#!/usr/bin/env python3
"""scene_describer — ask a vision-language model (VLM) about what the camera sees.

  ros2 topic pub --once /describe std_msgs/String "{data: 'Is there a cup on the table?'}"

Subscribes: image_topic         (sensor_msgs/Image)  keeps only the latest frame
            /describe           (std_msgs/String)    your question (empty = "describe the scene")
Publishes:  /scene_description  (std_msgs/String)
Needs:      Ollama running on the Jetson  (ollama pull moondream)
"""
import base64
import json
import threading
import time
import urllib.request

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class SceneDescriber(Node):
    def __init__(self):
        super().__init__('scene_describer')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model', 'moondream')
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/chat')

        self.bridge = CvBridge()
        self.latest = None
        self.busy = False

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, image_qos)
        self.create_subscription(String, '/describe', self.on_question, 10)
        self.pub = self.create_publisher(String, '/scene_description', 10)
        self.get_logger().info('Ask a question on /describe')

    def on_image(self, msg):
        self.latest = msg   # cheap: just keep a reference, convert only when asked

    def on_question(self, msg):
        if self.latest is None:
            self.get_logger().warn('No camera image yet - is the image_topic right?')
            return
        if self.busy:
            self.get_logger().warn('Still answering the last question')
            return
        question = msg.data.strip() or 'Describe what you see in one or two sentences.'
        frame = self.bridge.imgmsg_to_cv2(self.latest, desired_encoding='bgr8')
        self.busy = True
        threading.Thread(target=self.ask_vlm, args=(frame, question), daemon=True).start()

    def ask_vlm(self, frame, question):
        try:
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            body = {
                'model': self.get_parameter('model').value,
                'messages': [{'role': 'user', 'content': question,
                              'images': [base64.b64encode(jpg.tobytes()).decode()]}],
                'stream': False,
            }
            req = urllib.request.Request(self.get_parameter('ollama_url').value,
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                answer = json.loads(resp.read())['message']['content'].strip()
            self.get_logger().info(f'({time.time() - t0:.1f}s) {answer}')
            self.pub.publish(String(data=answer))
        except Exception as e:
            self.get_logger().error(f'VLM call failed: {e}')
        finally:
            self.busy = False


def main():
    rclpy.init()
    node = SceneDescriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
