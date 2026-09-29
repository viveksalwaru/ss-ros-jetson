#!/usr/bin/env python3
"""llm_commander — drive the robot with plain English, using a local LLM.

  ros2 topic pub --once /user_command std_msgs/String "{data: 'turn left a bit then go forward half a metre'}"

The LLM never touches the motors. It only returns a JSON plan; this node checks
and clamps every step before executing it. That separation is the key safety idea.

Subscribes: /user_command (std_msgs/String)
Publishes:  cmd_topic     (geometry_msgs/Twist)
            /llm_reply    (std_msgs/String)   what the robot "says" back
Needs:      Ollama running on the Jetson  (ollama pull qwen2.5:3b)
"""
import json
import math
import threading
import urllib.request

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

SYSTEM_PROMPT = """You control a small wheeled robot. Convert the user's request into JSON:
{"reply": "<one short sentence to the user>",
 "steps": [{"action": "forward" | "backward" | "turn_left" | "turn_right" | "stop", "value": <number>}]}
forward/backward value is in metres. turn value is in degrees. stop value is 0.
If the request is unsafe, unclear, or impossible, return an empty steps list and say why in reply.
Return only the JSON object."""

LIMITS = {'forward': 1.0, 'backward': 0.5, 'turn_left': 180, 'turn_right': 180, 'stop': 0}
LINEAR_SPEED = 0.10    # m/s
ANGULAR_SPEED = 0.50   # rad/s
MAX_STEPS = 6


class LlmCommander(Node):
    def __init__(self):
        super().__init__('llm_commander')
        self.declare_parameter('model', 'qwen2.5:3b')
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/chat')
        self.declare_parameter('cmd_topic', '/cmd_vel')

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)
        self.reply_pub = self.create_publisher(String, '/llm_reply', 10)
        self.create_subscription(String, '/user_command', self.on_command, 10)

        self.lock = threading.Lock()
        self.plan = []          # list of (Twist, seconds) still to run
        self.step_end = None    # when the current step finishes
        self.create_timer(0.1, self.run_plan)   # 10 Hz motion loop
        self.get_logger().info('Send commands on /user_command')

    # ---- 1. Ask the LLM (in a background thread so ROS keeps spinning) ----
    def on_command(self, msg):
        text = msg.data.strip()
        if text.lower() in ('stop', 'halt', 'freeze'):
            self.set_plan([])   # instant stop, no LLM round-trip
            return
        self.get_logger().info(f'Thinking about: "{text}"')
        threading.Thread(target=self.ask_llm, args=(text,), daemon=True).start()

    def ask_llm(self, text):
        body = {
            'model': self.get_parameter('model').value,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': text}],
            'format': 'json',
            'stream': False,
            'options': {'temperature': 0},
        }
        try:
            req = urllib.request.Request(self.get_parameter('ollama_url').value,
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                answer = json.loads(json.loads(resp.read())['message']['content'])
        except Exception as e:  # network error, bad JSON, Ollama not running...
            self.get_logger().error(f'LLM call failed: {e}')
            return

        reply = str(answer.get('reply', ''))
        steps = self.validate(answer.get('steps', []))
        self.get_logger().info(f'LLM: {reply} | plan: {steps}')
        self.reply_pub.publish(String(data=reply))
        self.set_plan(steps)

    # ---- 2. Never trust the model's output: check and clamp every step ----
    def validate(self, steps):
        safe = []
        if not isinstance(steps, list):
            return safe
        for step in steps[:MAX_STEPS]:
            try:
                action = step['action']
                value = abs(float(step.get('value', 0)))
            except (KeyError, TypeError, ValueError):
                continue
            if action not in LIMITS:
                continue
            safe.append((action, min(value, LIMITS[action])))
        return safe

    # ---- 3. Turn the plan into timed velocity commands ----
    def set_plan(self, steps):
        plan = []
        for action, value in steps:
            cmd = Twist()
            if action == 'forward':
                cmd.linear.x, secs = LINEAR_SPEED, value / LINEAR_SPEED
            elif action == 'backward':
                cmd.linear.x, secs = -LINEAR_SPEED, value / LINEAR_SPEED
            elif action == 'turn_left':
                cmd.angular.z, secs = ANGULAR_SPEED, math.radians(value) / ANGULAR_SPEED
            elif action == 'turn_right':
                cmd.angular.z, secs = -ANGULAR_SPEED, math.radians(value) / ANGULAR_SPEED
            else:  # stop
                secs = 0.5
            plan.append((cmd, secs))
        with self.lock:
            self.plan, self.step_end = plan, None
        if not plan:
            self.cmd_pub.publish(Twist())

    def run_plan(self):
        with self.lock:
            now = self.get_clock().now()
            if self.step_end is not None and now < self.step_end:
                self.cmd_pub.publish(self.current)   # keep sending the current step
                return
            if not self.plan:
                if self.step_end is not None:        # just finished the last step
                    self.cmd_pub.publish(Twist())
                    self.step_end = None
                return
            self.current, secs = self.plan.pop(0)
            self.step_end = now + Duration(seconds=secs)
            self.cmd_pub.publish(self.current)


def main():
    rclpy.init()
    node = LlmCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
