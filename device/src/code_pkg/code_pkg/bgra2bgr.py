import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class BGRA2BGRNode(Node):
	def __init__(self):
		super().__init__('bgra_to_bgr')
		self.bridge = CvBridge()
		self.sub = self.create_subscription(
			Image,
			'/camera/image_raw',
			self.image_callback,
			10)
		self.pub = self.create_publisher(Image, '/camera/image_bgr', 10)

	def image_callback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
			bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
			new_msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
			new_msg.header = msg.header
			self.pub.publish(new_msg)
		except Exception as e:
			self.get_logger().error(f"Image conversion failed: {e}")

def main():
		rclpy.init()
		node = BGRA2BGRNode()
		rclpy.spin(node)
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
		main()