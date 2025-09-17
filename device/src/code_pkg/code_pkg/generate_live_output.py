import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import cv2
import os
from PIL import Image as Image_PIL
from collections import deque, namedtuple
from datetime import datetime

SynchronizedPair = namedtuple('SynchronizedPair', ['timestamp_ns', 'pointcloud_xyz', 'image'])

K = [[968.691, 0, 309.268],
     [0, 965.205, 231.423],
     [0, 0, 1]]

T_cam_to_lidar = [[0.078630,  -0.996786,  -0.015351, -0.000833],
                  [-0.108505,   0.006750,  -0.994073, 0.291587],
                  [0.990982,   0.079830,  -0.107626, -0.375218],
                  [0, 0, 0, 1]]

T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)
image_shape = (800, 600, 3)

def bgra_2_bgr(image: np.ndarray) -> np.ndarray:
    if image.shape[2] != 4:
        raise ValueError("Eingabebild ist nicht im BGRA-Format")
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

class SyncNode(Node):
    def __init__(self):
        super().__init__('Live_sync_node')

        self.cloud_sub = self.create_subscription(PointCloud2, '/livox/lidar', self.cloud_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.cloud_buffer = deque()
        self.image_buffer = deque()
        self.time_tolerance_ns = 1_000_000_000  # 1000 ms

        self.synced_data = []

    def cloud_callback(self, msg):
        timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        self.cloud_buffer.append((timestamp, msg))
        self.try_sync()

    def image_callback(self, msg):
        timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        self.image_buffer.append((timestamp, msg))
        self.try_sync()

    def try_sync(self):
        while self.cloud_buffer and self.image_buffer:
            t_cloud, cloud_msg = self.cloud_buffer[0]
            t_img, img_msg = self.image_buffer[0]
            dt = abs(t_cloud - t_img)

            if dt < self.time_tolerance_ns:
                # LiDAR
                points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))

                # Kamera
                img_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, -1))

                self.synced_data.append(SynchronizedPair(timestamp_ns=t_img,
                                                         pointcloud_xyz=points,
                                                         image=img_data))

                self.get_logger().info(f"Synchronisiert! dt = {dt/1e6:.2f} ms | {len(self.synced_data)} Paare")

                self.cloud_buffer.popleft()
                self.image_buffer.popleft()

            elif t_cloud < t_img:
                self.cloud_buffer.popleft()
            else:
                self.image_buffer.popleft()

    def store_synced_data(self):
        timestamp = datetime.now().strftime("%d_%m_%H_%M_%S")
        output_root = f"synced_data_{timestamp}"
        os.makedirs(output_root, exist_ok=True)

        for i, pair in enumerate(self.synced_data, start=1):
            folder_name = f'scan_{i:03d}'
            folder_path = os.path.join(output_root, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # pointcloud
            np.save(os.path.join(folder_path, 'pointcloud.npy'), pair.pointcloud_xyz)

            with open(os.path.join(folder_path, 'pointcloud.pts'), 'w') as f:
                for point in pair.pointcloud_xyz:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

            # image
            img = bgra_2_bgr(pair.image)
            img = Image_PIL.fromarray(img)
            img = img.rotate(180.0, expand=True)
            img.save(os.path.join(folder_path, 'image.png'))

            with open(os.path.join(folder_path, 'timestamp.txt'), 'w') as f:
                f.write(str(pair.timestamp_ns))

        self.get_logger().info(f"{len(self.synced_data)} Paare gespeichert unter '{output_root}'")

def main():
    rclpy.init()
    node = SyncNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Beende Node, speichere Daten...")
        node.store_synced_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()

