import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField

import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER
)

class BNO085Publisher(Node):
    def __init__(self):
        super().__init__('bno085_publisher')

        # I2C starten
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(i2c)
        self.get_logger().info("✅ BNO085 erkannt")

        # Sensor-Features aktivieren
        self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_MAGNETOMETER)

        # Publisher für IMU und Magnetometer
        self.imu_pub = self.create_publisher(Imu, 'imu/rawdata', 10)
        self.mag_pub = self.create_publisher(MagneticField, 'imu/mag', 10)

        # Timer: 10 Hz
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

    def publish_sensor_data(self):
        try:
            # Sensordaten auslesen
            accel = self.sensor.acceleration  # m/s²
            gyro = self.sensor.gyro          # rad/s
            mag = self.sensor.magnetic       # µT

            if accel is None or gyro is None:
                return  # auf nächsten Zyklus warten

            # IMU-Nachricht
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = "imu_link"

            imu_msg.linear_acceleration.x = accel[0]
            imu_msg.linear_acceleration.y = accel[1]
            imu_msg.linear_acceleration.z = accel[2]

            imu_msg.angular_velocity.x = gyro[0]
            imu_msg.angular_velocity.y = gyro[1]
            imu_msg.angular_velocity.z = gyro[2]

            # Orientierung leer lassen (nicht vom Sensor geliefert)
            imu_msg.orientation_covariance[0] = -1.0  # sagt: "nicht verfügbar"

            self.imu_pub.publish(imu_msg)

            # Magnetometer-Nachricht (optional)
            if mag:
                mag_msg = MagneticField()
                mag_msg.header = imu_msg.header
                mag_msg.magnetic_field.x = mag[0] * 1e-6  # µT → Tesla
                mag_msg.magnetic_field.y = mag[1] * 1e-6
                mag_msg.magnetic_field.z = mag[2] * 1e-6
                self.mag_pub.publish(mag_msg)

        except Exception as e:
            self.get_logger().error(f"❌ Fehler beim Lesen: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BNO085Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
