#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
import time

class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_intrinsic_calibration')

        # Checkerboard settings
        self.chessboard_size = (7, 4)  # inner corners
        self.square_size = 0.055  # meters

        # Prepare object points
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Collected points
        self.objpoints = []
        self.imgpoints = []

        self.bridge = CvBridge()

        # Subscribe to raw image
        self.image_sub_raw = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback_raw, 10)

        # Auto-capture throttling
        self.last_capture_time = 0.0
        self.capture_interval = 1.0  # seconds

        # Output folder for saved images (with drawn chessboard)
        self.output_dir = "captured_chessboards"
        os.makedirs(self.output_dir, exist_ok=True)

        self.calibrated = False
        self.get_logger().info("Calibration node started. Listening only to /camera/image_raw.")

    def image_callback_raw(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # falls deine Kamera auf dem Kopf ist – sonst Zeile entfernen
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        self.process_frame(frame)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        display_frame = frame.copy()
        if ret:
            cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)

            # Auto-capture
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_interval:
                self.capture_frame(frame, gray, corners)
                self.last_capture_time = current_time

        cv2.imshow("Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # finish calibration manually
        if key == ord('q'):
            self.calibrate(gray.shape[::-1])

    def capture_frame(self, frame, gray, corners):
        # refine corners (subpixel)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # accumulate for calibration
        self.objpoints.append(self.objp)
        self.imgpoints.append(corners2)

        # draw refined corners into a copy and save that image
        annotated = frame.copy()
        cv2.drawChessboardCorners(annotated, self.chessboard_size, corners2, True)

        idx = len(self.objpoints)
        filename = os.path.join(self.output_dir, f"chessboard_{idx:03d}_annotated.png")
        cv2.imwrite(filename, annotated)
        self.get_logger().info(f"Captured frame {idx} (with drawn chessboard) -> {filename}")

    def calibrate(self, image_size):
        if len(self.objpoints) < 5:
            self.get_logger().warn("Not enough frames for calibration. Capture at least 5 good frames.")
            return

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )

        if ret:
            self.get_logger().info("Calibration successful.")
            calib_data = {
                "camera_matrix": mtx.tolist(),
                "dist_coeffs": dist.tolist()
            }
            with open("camera_intrinsics.json", "w") as f:
                json.dump(calib_data, f, indent=4)
            self.get_logger().info(f"Calibration saved to {os.path.abspath('camera_intrinsics.json')}")
            self.calibrated = True
        else:
            self.get_logger().error("Calibration failed.")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
