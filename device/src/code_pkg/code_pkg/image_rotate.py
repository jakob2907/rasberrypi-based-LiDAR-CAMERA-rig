import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageRotateNode(Node):
    def __init__(self):
        super().__init__('image_rotate_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/camera/image_rotated', 10)

    def listener_callback(self, msg):
        # ROS Image -> OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Bild um 180 Grad drehen
        rotated = cv2.rotate(cv_image, cv2.ROTATE_180)

        # OpenCV Image -> ROS Image
        rotated_msg = self.bridge.cv2_to_imgmsg(rotated, encoding='bgr8')
        rotated_msg.header = msg.header

        self.publisher.publish(rotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImageRotateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()