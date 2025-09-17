#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2, Image, Imu, MagneticField
from sensor_msgs_py import point_cloud2 as pc2
from collections import namedtuple
from PIL import Image as Image_PIL

import numpy as np
import cv2
import os
from datetime import datetime
import argparse
import json

# ===================== Kalibrierung (DEINE WERTE) ============================
# Intrinsics
K = np.array([
    [679.119,   0.0,   409.699],
    [  0.0,   677.096, 313.462],
    [  0.0,     0.0,     1.0  ],
], dtype=np.float64)

# Verzerrung: (k1, k2, p1, p2, k3) – k3 nicht gegeben -> 0.0
D = np.array([-0.150247, 0.468422, 0.00899281, 0.00482663, 0.0], dtype=np.float64)

# Extrinsics: Kamera -> LiDAR (R_cl | P_cl)
Rcl = np.array([
    [ 0.038044, -0.999249, -0.007335],
    [-0.017242,  0.006682, -0.999829],
    [ 0.999127,  0.038164, -0.016974],
], dtype=np.float64)
Pcl = np.array([-0.083912, -0.101645, 0.012768], dtype=np.float64)

T_cam_to_lidar = np.eye(4, dtype=np.float64)
T_cam_to_lidar[:3, :3] = Rcl
T_cam_to_lidar[:3,  3] = Pcl

T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)
# ============================================================================

SynchronizedPair = namedtuple('SynchronizedPair', ['timestamp_ns', 'pointcloud_xyz', 'image'])
IMUData = namedtuple('IMUData', ['mag', 'raw'])


def image_msg_to_rgb(img_msg: Image, *, undistort: bool = False) -> np.ndarray:
    """
    sensor_msgs/Image -> RGB (uint8, HxWx3). Optional: Entzerrung mit K,D.
    Unterstützt: bgra8, bgr8, rgba8, rgb8, mono8 – oder Fallback über Kanalzahl.
    """
    H, W = int(img_msg.height), int(img_msg.width)
    enc = (img_msg.encoding or "").lower()

    buf = np.frombuffer(img_msg.data, dtype=np.uint8)
    # Kanalzahl bestimmen
    if   enc in ("bgra8", "rgba8"): C = 4
    elif enc in ("bgr8", "rgb8"):   C = 3
    elif enc in ("mono8", "8uc1"):  C = 1
    else:
        C = buf.size // (H * W)
        if C not in (1, 3, 4):
            raise ValueError(f"Unerwartete Kanalzahl: {C} (encoding='{enc}')")

    img = buf.reshape(H, W, C)

    # -> RGB wandeln
    if C == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) if ('bgra' in enc or enc == "") else cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif C == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  if ('bgr'  in enc or enc == "") else img.copy()
    else:  # C == 1
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Optional: Entzerren (OpenCV arbeitet kanalunabhängig)
    if undistort:
        rgb = cv2.undistort(rgb, K, D, None, K)

    return rgb  # uint8, RGB


def read_synchronized_pairs(bag_path, cloud_topic, image_topic, imu_mag_topic, imu_raw_topic,
                            time_tolerance_ns=100_000_000, time_tolerance_imu_ns=100_000,
                            *, undistort_images: bool = False):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    cloud_msg_type = PointCloud2
    image_msg_type = Image
    imu_mag_msg_type = MagneticField
    imu_raw_msg_type = Imu

    cloud_buffer, image_buffer = [], []
    imu_mag_buffer, imu_raw_buffer = [], []
    synced_data = []

    print("Synchronisierte Verarbeitung gestartet...")

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == cloud_topic:
            msg = deserialize_message(data, cloud_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            cloud_buffer.append((timestamp, msg))

        elif topic == image_topic:
            msg = deserialize_message(data, image_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            image_buffer.append((timestamp, msg))

        elif topic == imu_mag_topic:
            msg = deserialize_message(data, imu_mag_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            imu_raw_buffer.append({"timestamp": timestamp,
                                   "x": msg.magnetic_field.x, "y": msg.magnetic_field.y, "z": msg.magnetic_field.z})

        elif topic == imu_raw_topic:
            msg = deserialize_message(data, imu_raw_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            imu_mag_buffer.append({
                "timestamp": timestamp,
                "orientation": {
                    "x": msg.orientation.x, "y": msg.orientation.y,
                    "z": msg.orientation.z, "w": msg.orientation.w
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x, "y": msg.angular_velocity.y, "z": msg.angular_velocity.z
                },
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x, "y": msg.linear_acceleration.y, "z": msg.linear_acceleration.z
                }
            })

        while cloud_buffer and image_buffer:
            t_cloud, cloud_msg = cloud_buffer[0]
            t_img, img_msg = image_buffer[0]
            dt = abs(t_cloud - t_img)

            if dt < time_tolerance_ns:
                # Punktwolke
                points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))
                # Bild -> RGB (optional undistort)
                img_rgb = image_msg_to_rgb(img_msg, undistort=undistort_images)

                synced_data.append(SynchronizedPair(timestamp_ns=t_img,
                                                    pointcloud_xyz=points,
                                                    image=img_rgb))
                cloud_buffer.pop(0)
                image_buffer.pop(0)
            elif t_cloud < t_img:
                cloud_buffer.pop(0)
            else:
                image_buffer.pop(0)

    return synced_data, IMUData(mag=imu_mag_buffer, raw=imu_raw_buffer)


def read_not_synchronized_pairs(bag_path, cloud_topic, image_topic, imu_raw_topic, imu_mag_topic,
                                time_tolerance_ns=100_000_000, time_tolerance_imu_ns=100_000,
                                *, undistort_images: bool = False):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    cloud_msg_type = PointCloud2
    image_msg_type = Image
    imu_mag_msg_type = MagneticField
    imu_raw_msg_type = Imu

    cloud_buffer, image_buffer = [], []
    imu_mag_buffer, imu_raw_buffer = [], []
    synced_data = []

    print("Verarbeitung gestartet...")

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == cloud_topic:
            msg = deserialize_message(data, cloud_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            cloud_buffer.append((timestamp, msg))

        elif topic == image_topic:
            msg = deserialize_message(data, image_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            image_buffer.append((timestamp, msg))

        elif topic == imu_mag_topic:
            msg = deserialize_message(data, imu_mag_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            imu_raw_buffer.append({"timestamp": timestamp,
                                   "x": msg.magnetic_field.x, "y": msg.magnetic_field.y, "z": msg.magnetic_field.z})

        elif topic == imu_raw_topic:
            msg = deserialize_message(data, imu_raw_msg_type)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            imu_mag_buffer.append({
                "timestamp": timestamp,
                "orientation": {
                    "x": msg.orientation.x, "y": msg.orientation.y,
                    "z": msg.orientation.z, "w": msg.orientation.w
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x, "y": msg.angular_velocity.y, "z": msg.angular_velocity.z
                },
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x, "y": msg.linear_acceleration.y, "z": msg.linear_acceleration.z
                }
            })

        while cloud_buffer and image_buffer:
            t_cloud, cloud_msg = cloud_buffer[0]
            t_img, img_msg = image_buffer[0]

            points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))
            img_rgb = image_msg_to_rgb(img_msg, undistort=undistort_images)

            synced_data.append(SynchronizedPair(timestamp_ns=t_img,
                                                pointcloud_xyz=points,
                                                image=img_rgb))
            cloud_buffer.pop(0)
            image_buffer.pop(0)

    print(f"length: {len(synced_data)}")
    return synced_data, IMUData(mag=imu_mag_buffer, raw=imu_raw_buffer)


def store_synced_data(bag_infos, imu_infos, *, undistort_used: bool):
    # Mainfolder
    timestamp = datetime.now().strftime("%d_%m_%H_%M_%S")
    output_root = f"Output_{timestamp}"
    os.makedirs(output_root, exist_ok=True)

    # Kalibrierung mitschreiben (praktisch für spätere Schritte)
    calib = {
        "K": K.tolist(),
        "D": D.tolist(),
        "T_cam_to_lidar": T_cam_to_lidar.tolist(),
        "T_lidar_to_cam": T_lidar_to_cam.tolist(),
        "undistort": bool(undistort_used),
        "note": "Images sind RGB gespeichert; falls undistort=True, bereits entzerrt."
    }
    with open(os.path.join(output_root, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)

    scan_counter = 1
    index_lines = []

    for bag_path, synced_data in bag_infos:
        bag_name = os.path.basename(bag_path)
        start_scan = scan_counter

        for pair in synced_data:
            folder_name = f'scan_{scan_counter:03d}'
            folder_path = os.path.join(output_root, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Pointcloud
            np.save(os.path.join(folder_path, 'pointcloud.npy'), pair.pointcloud_xyz)
            with open(os.path.join(folder_path, 'pointcloud.pts'), 'w') as f:
                for p in pair.pointcloud_xyz:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")

            # Bild als PNG (RGB!) – 180° drehen wie bisher
            img_rgb = pair.image
            img_pil = Image_PIL.fromarray(img_rgb)  # erwartet RGB
            img_pil = img_pil.rotate(180.0, expand=True)
            img_pil.save(os.path.join(folder_path, 'image.png'))

            # Zeitstempel
            with open(os.path.join(folder_path, 'timestamp.txt'), 'w') as f:
                f.write(str(pair.timestamp_ns))

            scan_counter += 1

        end_scan = scan_counter - 1
        index_lines.append(f"{bag_name}: scan_{start_scan:03d} – scan_{end_scan:03d}")

    # Index-Datei schreiben
    with open(os.path.join(output_root, "index.txt"), "w") as f:
        f.write("\n".join(index_lines))

    # IMU-Dateien pro Bag
    for bag_path, imu_data in imu_infos:
        bag_name = os.path.basename(bag_path)
        imu_path = os.path.join(output_root, f'imu_data_{bag_name}.json')
        if imu_data and (imu_data.mag or imu_data.raw):
            with open(imu_path, 'w') as f:
                json.dump({"mag": imu_data.mag, "raw": imu_data.raw}, f, indent=4)

    print(f"{scan_counter-1} Scans wurden in '{output_root}' gespeichert (inkl. .npy, .pts, imu_data, index.txt, calibration.json).")


def main():
    rclpy.init()

    parser = argparse
