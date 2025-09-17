#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
from collections import deque, namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as Image_PIL  # kept for optional paths (not used by default)

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image, Imu, MagneticField, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import shutil

# -------------------------- Calibration (editable) ---------------------------

K_DEFAULT = [[679.119, 0.0, 409.699],
             [0.0, 677.096, 313.462],
             [0.0, 0.0, 1.0]]

T_CAM_TO_LIDAR_DEFAULT = [
    [0.038044,  -0.999249,  -0.007335, -0.083912],
    [-0.017242,   0.006682,  -0.999829, -0.101645],
    [0.999127,   0.038164,  -0.016974, 0.012768],
    [0.0, 0.0, 0.0, 1.0],
]

# ------------------------------- Data types ----------------------------------

SynchronizedPair = namedtuple("SynchronizedPair", ["timestamp_ns", "pointcloud_xyz", "image_bgr"])

@dataclass
class IMUData:
    mag: List[dict]
    raw: List[dict]

# ------------------------------- Utilities -----------------------------------

def setup_logger(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3:4]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv.ravel()
    return T_inv

def parse_encoding_channels(encoding: str) -> int:
    enc = encoding.lower()
    if enc in ("bgr8", "rgb8"):
        return 3
    if enc in ("bgra8", "rgba8"):
        return 4
    if enc in ("mono8", ):
        return 1
    # conservative default (fallback path may try cv_bridge)
    return 3

def rosimg_to_bgr(img_msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs/Image to OpenCV BGR ndarray robustly.
    Tries common encodings; optionally falls back to cv_bridge if available.
    """
    enc = (img_msg.encoding or "").lower()
    channels = parse_encoding_channels(enc)

    # reshape using step to be safe about row stride
    data = np.frombuffer(img_msg.data, dtype=np.uint8)
    h, w, step = img_msg.height, img_msg.width, img_msg.step
    if len(data) != h * step:
        raise ValueError(f"Image data size ({len(data)}) != height*step ({h*step})")

    arr = np.reshape(data, (h, step))
    expected = w * channels
    if expected > step:
        raise ValueError(f"Expected row bytes {expected} > step {step}")

    arr = arr[:, :expected].copy()  # drop possible padding
    if channels == 1:
        gray = np.reshape(arr, (h, w))
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr

    img = np.reshape(arr, (h, w, channels))

    if enc == "bgr8":
        return img
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Fallback to cv_bridge if present
    try:
        from cv_bridge import CvBridge  # type: ignore
        bridge = CvBridge()
        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        return cv_img
    except Exception as e:
        raise ValueError(f"Unsupported encoding '{img_msg.encoding}', and cv_bridge failed: {e}")

def pointcloud2_to_xyz(msg) -> np.ndarray:
    """
    Liefert (N,3) float32-Array aus PointCloud2.
    Nutzt read_points_numpy, fällt bei Bedarf auf Generator zurück.
    """
    try:
        rec = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        # rec ist ein record array mit Feldern 'x','y','z'
        if rec.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = np.stack((rec["x"], rec["y"], rec["z"]), axis=-1).astype(np.float32, copy=False)
        return pts
    except Exception:
        # Fallback: Generator -> Liste von Tupeln -> ndarray
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts_list = [(x, y, z) for x, y, z in gen]
        if not pts_list:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(pts_list, dtype=np.float32)


def rotate180_inplace_bgr(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_bgr, cv2.ROTATE_180)

def write_scan(
    root: str,
    scan_idx: int,
    pair: SynchronizedPair,
    rotate180: bool,
    K: np.ndarray,
    T_cam_to_lidar: np.ndarray,
    topics: dict,
    image_encoding: str,
) -> None:
    folder_name = f"scan_{scan_idx:03d}"
    folder_path = os.path.join(root, folder_name)
    ensure_dir(folder_path)

    # Save point cloud
    np.save(os.path.join(folder_path, "pointcloud.npy"), pair.pointcloud_xyz)

    with open(os.path.join(folder_path, "pointcloud.pts"), "w") as f:
        for x, y, z in pair.pointcloud_xyz:
            f.write(f"{x} {y} {z}\n")

    # Save image
    img_bgr = pair.image_bgr
    if rotate180:
        img_bgr = rotate180_inplace_bgr(img_bgr)
    cv2.imwrite(os.path.join(folder_path, "image.png"), img_bgr)

    # Timestamp
    with open(os.path.join(folder_path, "timestamp.txt"), "w") as f:
        f.write(str(pair.timestamp_ns))

    # Metadata
    meta = {
        "timestamp_ns": int(pair.timestamp_ns),
        "camera_matrix_K": np.asarray(K, dtype=float).tolist(),
        "T_cam_to_lidar": np.asarray(T_cam_to_lidar, dtype=float).tolist(),
        "topics": topics,
        "image_encoding": image_encoding,
        "versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    with open(os.path.join(folder_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

# ----------------------- Synchronization (nearest) ---------------------------

def pop_nearest(
    target_ts: int,
    buffer: Deque[Tuple[int, object]],
    tolerance_ns: int,
) -> Optional[Tuple[int, object, int]]:
    """
    Find and remove the message in 'buffer' whose timestamp is nearest to target_ts,
    if within tolerance_ns. Returns (timestamp, msg, index) or None if not found.
    """
    if not buffer:
        return None
    # search linear (buffers are usually small). For large buffers, bisect or heap.
    best_idx = -1
    best_dt = None
    for i, (ts, _msg) in enumerate(buffer):
        dt = abs(ts - target_ts)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_idx = i
    if best_dt is None or best_dt > tolerance_ns:
        return None
    ts, msg = buffer[best_idx]
    del buffer[best_idx]
    return (ts, msg, best_idx)

# ------------------------------- Reader --------------------------------------

def read_synchronized_pairs(
    bag_path: str,
    cloud_topic: str,
    image_topic: str,
    imu_mag_topic: str,
    imu_raw_topic: str,
    storage_format: str,
    time_tolerance_ns: int,
    rotate180: bool,
    output_root: str,
    K: np.ndarray,
    T_cam_to_lidar: np.ndarray,
) -> Tuple[int, IMUData, str]:
    """
    Streams synchronized pairs directly to disk under 'output_root'.
    Returns (num_scans, imu_data, output_root).
    """
    logging.info(f"Opening bag: {bag_path} (storage: {storage_format})")
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id=storage_format)
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    logging.debug(f"Topic types: {type_map}")

    # Prepare buffers and counters
    cloud_buf: Deque[Tuple[int, PointCloud2]] = deque()
    image_buf: Deque[Tuple[int, Image]] = deque()
    imu_mag_buffer: List[dict] = []
    imu_raw_buffer: List[dict] = []
    scan_counter = 0

    # Topic meta for metadata.json
    topics_meta = {
        "cloud_topic": cloud_topic,
        "image_topic": image_topic,
        "imu_mag_topic": imu_mag_topic,
        "imu_raw_topic": imu_raw_topic,
    }

    # Stream loop
    while reader.has_next():
        topic, data, _t = reader.read_next()

        if topic == cloud_topic:
            msg = deserialize_message(data, PointCloud2)
            ts_cloud = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            cloud_buf.append((ts_cloud, msg))

            nearest = pop_nearest(ts_cloud, image_buf, time_tolerance_ns)
            if nearest is not None:
                ts_img, img_msg, _ = nearest
                dt_ns = abs(ts_cloud - ts_img)
                dt_ms = dt_ns / 1e6
                logging.debug(
                    f"Matched cloud↔image | Δt={dt_ms:.3f} ms "
                    f"(cloud_ts={ts_cloud}, image_ts={ts_img})"
                )
                try:
                    img_bgr = rosimg_to_bgr(img_msg)
                except Exception as e:
                    logging.warning(f"Image decode failed at ts={ts_img}: {e}")
                    continue

                pts = pointcloud2_to_xyz(msg)
                pair = SynchronizedPair(timestamp_ns=ts_img,
                                        pointcloud_xyz=pts,
                                        image_bgr=img_bgr)
                scan_counter += 1
                write_scan(
                    output_root, scan_counter, pair, rotate180,
                    K, T_cam_to_lidar, topics_meta,
                    image_encoding=(getattr(img_msg, "encoding", "") or "")
                )
            else:
                if image_buf:
                    # zeige z.B. das älteste Bild im Buffer als Referenz
                    ts_img, _ = image_buf[0]
                    dt_ns = abs(ts_cloud - ts_img)
                    dt_ms = dt_ns / 1e6
                    logging.debug(
                        f"No image within tolerance | Δt={dt_ms:.3f} ms "
                        f"(cloud_ts={ts_cloud}, nearest_image_ts={ts_img})"
                    )
                else:
                    logging.debug(f"No image in buffer for cloud_ts={ts_cloud}")


        elif topic == image_topic:
            # Bild nur puffern; Matching passiert im Cloud-Zweig
            img_msg = deserialize_message(data, Image)
            ts_img = img_msg.header.stamp.sec * 10**9 + img_msg.header.stamp.nanosec
            image_buf.append((ts_img, img_msg))

        elif topic == imu_mag_topic:
            msg = deserialize_message(data, MagneticField)
            ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            imu_mag_buffer.append({
                "timestamp": ts,
                "x": float(msg.magnetic_field.x),
                "y": float(msg.magnetic_field.y),
                "z": float(msg.magnetic_field.z),
            })

        elif topic == imu_raw_topic:
            msg = deserialize_message(data, Imu)
            ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            imu_raw_buffer.append({
                "timestamp": ts,
                "orientation": {
                    "x": float(msg.orientation.x),
                    "y": float(msg.orientation.y),
                    "z": float(msg.orientation.z),
                    "w": float(msg.orientation.w),
                },
                "angular_velocity": {
                    "x": float(msg.angular_velocity.x),
                    "y": float(msg.angular_velocity.y),
                    "z": float(msg.angular_velocity.z),
                },
                "linear_acceleration": {
                    "x": float(msg.linear_acceleration.x),
                    "y": float(msg.linear_acceleration.y),
                    "z": float(msg.linear_acceleration.z),
                },
            })

    # Write IMU for this bag (if any)
    return scan_counter, IMUData(mag=imu_mag_buffer, raw=imu_raw_buffer), output_root

# --------------------------------- Main --------------------------------------

def main():
    import shutil  # sicherstellen, dass verfügbar

    parser = argparse.ArgumentParser(
        description="Synchronize PointCloud2 and Image from ROS2 bags and write scan_* folders."
    )
    parser.add_argument("--bag_path", type=str, nargs="+", required=True,
                        help="List of bag files (e.g., .mcap, .db3)")
    parser.add_argument("--cloud_topic", type=str, default="/livox/lidar",
                        help="LiDAR topic (default: /livox/lidar)")
    parser.add_argument("--image_topic", type=str, default="/camera/image_raw",
                        help="Camera topic (default: /camera/image_raw)")
    parser.add_argument("--imu_raw_topic", type=str, default="/imu/rawdata",
                        help="IMU raw topic (default: /imu/rawdata)")
    parser.add_argument("--imu_mag_topic", type=str, default="/imu/mag",
                        help="Mag topic (default: /imu/mag)")
    parser.add_argument("--time_tolerance_ns", type=int, default=10_000_000,
                        help="Pairing tolerance in ns (default: 10ms)")
    parser.add_argument("--storage_format", type=str, default="mcap",
                        help="rosbag2 storage id (e.g., mcap, sqlite3)")
    parser.add_argument("--rotate180", action="store_true",
                        help="Rotate saved images by 180 degrees")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--output_root", type=str, default="",
                        help="Override output root folder (default: auto timestamp near bags)")
    # Optional: Kalibrierung überschreiben (falls in deinem Skript vorhanden)
    parser.add_argument("--K", type=float, nargs=9,
                        metavar=("k11","k12","k13","k21","k22","k23","k31","k32","k33"),
                        help="Override camera matrix K (row-major 3x3)")
    parser.add_argument("--T_cam_to_lidar", type=float, nargs=16,
                        help="Override 4x4 T_cam_to_lidar (row-major)")

    args = parser.parse_args()
    setup_logger(args.log_level)

    rclpy.init()

    # --- Bag-Pfade normieren (expand + absolut) ---
    bag_paths = [os.path.abspath(os.path.expanduser(p)) for p in args.bag_path]
    if not bag_paths:
        logging.error("No bag paths provided.")
        return

    # Gemeinsamer Basisordner aller Bags → Standard-Output liegt dort
    bag_dirs = [os.path.dirname(p) for p in bag_paths]
    base_dir = os.path.commonpath(bag_dirs) if len(bag_dirs) > 1 else bag_dirs[0]

    # --- Output-Root bestimmen ---
    if args.output_root:
        output_root = os.path.abspath(os.path.expanduser(args.output_root))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join(base_dir, f"Output_{stamp}")

    ensure_dir(output_root)
    logging.info(f"Output root: {output_root}")

    # --- Kalibrierung vorbereiten (optional) ---
    if hasattr(args, "K") and args.K is not None:
        K = np.array(args.K, dtype=float).reshape(3, 3)
    else:
        K = np.array(K_DEFAULT, dtype=float)

    if hasattr(args, "T_cam_to_lidar") and args.T_cam_to_lidar is not None:
        T_cam_to_lidar = np.array(args.T_cam_to_lidar, dtype=float).reshape(4, 4)
    else:
        T_cam_to_lidar = np.array(T_CAM_TO_LIDAR_DEFAULT, dtype=float)

    # --- Verarbeitung ---
    index_lines = []
    total_scans = 0
    imu_files_written = []

    for bag_path in bag_paths:
        bag_name = os.path.basename(bag_path)
        logging.info(f"Processing bag: {bag_name}")

        start_idx_before = total_scans + 1

        scans_written, imu_data, out_root = read_synchronized_pairs(
            bag_path=bag_path,
            cloud_topic=args.cloud_topic,
            image_topic=args.image_topic,
            imu_mag_topic=args.imu_mag_topic,
            imu_raw_topic=args.imu_raw_topic,
            storage_format=args.storage_format,
            time_tolerance_ns=args.time_tolerance_ns,
            rotate180=args.rotate180,
            output_root=output_root,  # absoluter Pfad verwenden
            K=K,
            T_cam_to_lidar=T_cam_to_lidar,
        )

        if scans_written > 0:
            first_scan = start_idx_before
            last_scan = start_idx_before + scans_written - 1
            index_lines.append(f"{bag_name}: scan_{first_scan:03d} – scan_{last_scan:03d}")
            total_scans += scans_written
        else:
            logging.warning(f"No scans written for {bag_name} (no paired data)")

        # IMU-JSON pro Bag (falls vorhanden)
        if imu_data.mag or imu_data.raw:
            imu_path = os.path.join(output_root, f"imu_data_{bag_name}.json")
            with open(imu_path, "w") as f:
                json.dump({"mag": imu_data.mag, "raw": imu_data.raw}, f, indent=2)
            imu_files_written.append(imu_path)

    # Index-Datei schreiben
    if index_lines:
        with open(os.path.join(output_root, "index.txt"), "w") as f:
            f.write("\n".join(index_lines))

    # ZIP-Archiv des Output-Ordners erzeugen (liegt neben dem Ordner)
    zip_file = shutil.make_archive(output_root, "zip", output_root)
    logging.info(f"ZIP archive created: {zip_file}")

    logging.info(
        f"Done. {total_scans} scans written into '{output_root}'. "
        f"Index entries: {len(index_lines)}. IMU files: {len(imu_files_written)}"
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()
