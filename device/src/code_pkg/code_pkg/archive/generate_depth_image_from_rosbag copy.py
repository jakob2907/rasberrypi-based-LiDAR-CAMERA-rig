import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from dataclasses import dataclass
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import Imu, MagneticField
import numpy as np
import cv2
import os
from collections import namedtuple
from PIL import Image as Image_PIL
from datetime import datetime
import argparse
import json

K = [[968.691, 0, 309.268],
	 [0, 965.205, 231.423],
	 [0, 0, 1]]

T_cam_to_lidar = [[0.078630,  -0.996786,  -0.015351, -0.000833],
				  [-0.108505,   0.006750,  -0.994073, 0.291587],
				  [0.990982,   0.079830,  -0.107626, -0.375218],
				  [0, 0, 0, 1]]

T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)

#height x width
image_shape = (800, 600, 3) 


#@dataclass
#class SynchronizedPair:
#	timestamp_ns: int
#	pointcloud_xyz: np.ndarray #(N, 3)
#	image: np.ndarray # (H, W, C)
	

SynchronizedPair = namedtuple('SynchronizedPair', ['timestamp_ns', 'pointcloud_xyz', 'image'])

IMUData = namedtuple('IMUData', ['mag', 'raw'])

def bgra_2_bgr(image:np.ndarray) -> np.ndarray:
	if image.shape[2] != 4:
		raise ValueError("eingabebild ist nicht in bgra Format")

	return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

#through rosbag
def read_synchronized_pairs(bag_path, cloud_topic, image_topic, imu_mag_topic, imu_raw_topic, time_tolerance_ns=100000000, time_tolerance_imu_ns = 100000):
#	rclpy.init()

	reader = SequentialReader()
	storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
	converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
	reader.open(storage_options, converter_options)

	topic_types = reader.get_all_topics_and_types()
	type_map = {topic.name: topic.type for topic in topic_types}

	cloud_msg_type = PointCloud2
	image_msg_type = Image
	imu_mag_msg_type = MagneticField
	imu_raw_msg_type = Imu

	cloud_buffer = []
	image_buffer = []
	imu_mag_buffer = []
	imu_raw_buffer = []
	synced_data = []
	imu_data = []

	print("Synchronisierte verarbeitung gestartet...")
	i = 0

	#print(f"image_msg_type: {image_msg_type}")

	while reader.has_next():
		#print(f"while schleifendurchlauf: {i}")		#1403 Durchläufe
		topic, data, t = reader.read_next()

		#print(f"Topic: {topic}")

		if topic == cloud_topic:
			#print(f"cloud_topic: {topic == cloud_topic}")
			msg = deserialize_message(data, cloud_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			#print(f"Timestamp pc: {timestamp / 1e9}")
			cloud_buffer.append((timestamp, msg))

		elif topic == image_topic:
			#print(f"image_topic: {topic}")
			msg = deserialize_message(data, image_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			#print(f"Timestamp img: {timestamp / 1e9}")
			image_buffer.append((timestamp, msg))

		elif topic == imu_mag_topic:
			msg = deserialize_message(data, imu_mag_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			imu_raw_buffer.append({
				"timestamp":timestamp,
				"x": msg.magnetic_field.x,
				"y": msg.magnetic_field.y,
				"z": msg.magnetic_field.z
			})

		elif topic == imu_raw_topic:
			msg = deserialize_message(data, imu_raw_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			imu_mag_buffer.append({
				"timestamp": timestamp,
				"orientation": {
					"x": msg.orientation.x,
					"y": msg.orientation.y,
					"z": msg.orientation.z,
					"w": msg.orientation.w
				},
				"angular_velocity": {
					"x": msg.angular_velocity.x,
					"y": msg.angular_velocity.y,
					"z": msg.angular_velocity.z
				},
				"linear_acceleration":{
					"x": msg.linear_acceleration.x,
					"y": msg.linear_acceleration.y,
					"z": msg.linear_acceleration.z
				}
			})

		while cloud_buffer and image_buffer:
			print("in while schleife")
			t_cloud, cloud_msg = cloud_buffer[0]
			t_img, img_msg = image_buffer[0]
			dt = abs(t_cloud - t_img)
			print(f"Delta_t: {dt}")

			if dt < time_tolerance_ns:
				print(f"Synchronisiert: dt={dt/1e6:.2f} ms")

				#Punktwolke verarbeiten
				points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))

				#Image verarbeiten
				img_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, -1))

				#depth_data = project_lidar_to_depth_image(points, K, T_lidar_to_cam, image_shape)


				synced_data.append(SynchronizedPair(timestamp_ns = t_img,
													pointcloud_xyz = points,
													image = img_data,
													))

				#synced data in eigenem ordner speichern -> Ordnerstruktur scan_001, scan_002,...
				#einzelne Dateien: pointcloud, bild, timestamp

				cloud_buffer.pop(0)
				image_buffer.pop(0)

			elif t_cloud < t_img:
				cloud_buffer.pop(0)
				print(f"t_cloud smaller")
			else:
				image_buffer.pop(0)
				print(f"t_image smaller")

		i += 1

	return synced_data, IMUData(mag=imu_mag_buffer, raw=imu_raw_buffer)
	rclpy.shutdown()


def read_not_synchronized_pairs(bag_path, cloud_topic, image_topic, imu_raw_topic, imu_mag_topic, time_tolerance_ns=100000000, time_tolerance_imu_ns = 100000):
	#rclpy.init()

	reader = SequentialReader()
	storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
	converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
	reader.open(storage_options, converter_options)

	topic_types = reader.get_all_topics_and_types()
	type_map = {topic.name: topic.type for topic in topic_types}

	cloud_msg_type = PointCloud2
	image_msg_type = Image
	imu_mag_msg_type = MagneticField
	imu_raw_msg_type = Imu

	cloud_buffer = []
	image_buffer = []
	imu_mag_buffer = []
	imu_raw_buffer = []
	synced_data = []
	imu_data = []

	print("Verarbeitung gestartet...")
	i = 0
	

	while reader.has_next():
		#print(f"while schleifendurchlauf: {i}")		#1403 Durchläufe
		topic, data, t = reader.read_next()

		print(f"Topic: {topic}")

		if topic == cloud_topic:
			#print(f"cloud_topic: {topic == cloud_topic}")
			msg = deserialize_message(data, cloud_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			#print(f"Timestamp pc: {timestamp / 1e9}")
			cloud_buffer.append((timestamp, msg))

		elif topic == image_topic:
			#print(f"image_topic: {topic}")
			msg = deserialize_message(data, image_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			#print(f"Timestamp img: {timestamp / 1e9}")
			image_buffer.append((timestamp, msg))

		elif topic == imu_mag_topic:
			msg = deserialize_message(data, imu_mag_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			imu_raw_buffer.append({
				"timestamp":timestamp,
				"x": msg.magnetic_field.x,
				"y": msg.magnetic_field.y,
				"z": msg.magnetic_field.z
			})

		elif topic == imu_raw_topic:
			msg = deserialize_message(data, imu_raw_msg_type)
			timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
			imu_mag_buffer.append({
				"timestamp": timestamp,
				"orientation": {
					"x": msg.orientation.x,
					"y": msg.orientation.y,
					"z": msg.orientation.z,
					"w": msg.orientation.w
				},
				"angular_velocity": {
					"x": msg.angular_velocity.x,
					"y": msg.angular_velocity.y,
					"z": msg.angular_velocity.z
				},
				"linear_acceleration":{
					"x": msg.linear_acceleration.x,
					"y": msg.linear_acceleration.y,
					"z": msg.linear_acceleration.z
				}
			})

		while cloud_buffer and image_buffer:
			print(f"Durchlauf: {i}")
			t_cloud, cloud_msg = cloud_buffer[0]
			t_img, img_msg = image_buffer[0]
			dt = abs(t_cloud - t_img)

			#if dt < time_tolerance_ns:
				#print(f"Synchronisiert: dt={dt/1e6:.2f} ms")

				#Punktwolke verarbeiten
			points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))

				#Image verarbeiten
			img_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((img_msg.height, img_msg.width, -1))

				#depth_data = project_lidar_to_depth_image(points, K, T_lidar_to_cam, image_shape)

			synced_data.append(SynchronizedPair(timestamp_ns = t_img,
													pointcloud_xyz = points,
													image = img_data
													))

			cloud_buffer.pop(0)
			image_buffer.pop(0)

			#elif t_cloud < t_img:
			#	cloud_buffer.pop(0)
			#	print(f"t_cloud smaller")
			#else:
			#	image_buffer.pop(0)
			#	print(f"t_image smaller")
			i += 1

	print(f"length: {len(synced_data)}")
	return synced_data, IMUData(mag=imu_mag_buffer, raw=imu_raw_buffer)
	rclpy.shutdown()

def store_synced_data(bag_infos, imu_infos):
    # Mainfolder
    timestamp = datetime.now().strftime("%d_%m_%H_%M_%S")
    output_root = f"Output_{timestamp}"
    os.makedirs(output_root, exist_ok=True)

    scan_counter = 1
    index_lines = []

    for bag_path, synced_data in bag_infos:
        bag_name = os.path.basename(bag_path)
        start_scan = scan_counter

        for pair in synced_data:
            folder_name = f'scan_{scan_counter:03d}'
            folder_path = os.path.join(output_root, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Pointcloud als .npy
            pointcloud_npy_path = os.path.join(folder_path, 'pointcloud.npy')
            np.save(pointcloud_npy_path, pair.pointcloud_xyz)

            # Pointcloud als .pts
            pointcloud_pts_path = os.path.join(folder_path, 'pointcloud.pts')
            with open(pointcloud_pts_path, 'w') as f:
                for point in pair.pointcloud_xyz:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

            # Bild als PNG
            image_path = os.path.join(folder_path, 'image.png')
            img_pair = bgra_2_bgr(pair.image)
            img = Image_PIL.fromarray(img_pair)
            img = img.rotate(180.0, expand=True)
            img.save(image_path)

            # Zeitstempel als Text
            timestamp_path = os.path.join(folder_path, 'timestamp.txt')
            with open(timestamp_path, 'w') as f:
                f.write(str(pair.timestamp_ns))

            scan_counter += 1

        end_scan = scan_counter - 1
        index_lines.append(f"{bag_name}: scan_{start_scan:03d} – scan_{end_scan:03d}")

    # Index-Datei schreiben
    index_path = os.path.join(output_root, "index.txt")
    with open(index_path, "w") as f:
        f.write("\n".join(index_lines))

    # IMU-Dateien schreiben (pro Bag)
    for bag_path, imu_data in imu_infos:
        bag_name = os.path.basename(bag_path)
        file_name = f'imu_data_{bag_name}.json'
        imu_path = os.path.join(output_root, file_name)
        os.makedirs(output_root, exist_ok=True)

        if imu_data and (imu_data.mag or imu_data.raw):
            with open(imu_path, 'w') as f:
                json.dump({"mag": imu_data.mag, "raw": imu_data.raw}, f, indent=4)

    print(f"{scan_counter-1} Scans wurden in '{output_root}' gespeichert (inkl. .npy, .pts, imu_data und index.txt).")


def main():

	rclpy.init()

	parser = argparse.ArgumentParser(description="Synchronisiere Pointclouds und Bilder aus einer ROS2-Bag")
	parser.add_argument(
        "--bag_path",
        type=str,
        nargs="+",
        required=True,
        help="Liste von .mcap Bag-Datei"
    )
	parser.add_argument(
        "--cloud_topic",
        type=str,
        default="/livox/lidar",
        help="Topic der Lidar-Daten (default: /livox/lidar)"
    )
	parser.add_argument(
        "--image_topic",
        type=str,
        default="/camera/image_raw",
        help="Topic der Kamera-Daten (default: /camera/image_raw)"
    )
	parser.add_argument(
        "--imu_raw_topic",
        type=str,
        default="/imu/rawdata",
        help="Topic der raw IMU Daten (default: /imu/raw_data)"
    )
	parser.add_argument(
        "--imu_mag_topic",
        type=str,
        default="/imu/mag",
        help="Topic der mag IMU Daten (default: /imu/mag)"
    )
	parser.add_argument(
        "--time_tolerance_ns",
        type=int,
        default=10000000,
        help="Toleranz für Synchronisation in ns (default: 10ms)"
    )
	parser.add_argument(
		"--storage_format",
		type=str,
		default="mcap",
		help="welches format die bag datei hat: mcap, sqlite, etc.."
	)

	args = parser.parse_args()

	bag_paths = [os.path.expanduser(p) for p in args.bag_path]
	cloud_topic = args.cloud_topic
	image_topic = args.image_topic
	imu_raw_topic = args.imu_raw_topic
	imu_mag_topic = args.imu_mag_topic
	time_tolerance = args.time_tolerance_ns
	storage_format = args.storage_format

	bag_infos = []
	imu_infos = []

	for bag_path in args.bag_path:
		print(f"Verarbeit bag: {bag_path}")
    
		synced, imu_data = read_not_synchronized_pairs(bag_path, cloud_topic, image_topic, imu_raw_topic, imu_mag_topic, time_tolerance_ns=10000000, time_tolerance_imu_ns=100000)
		print(f"synced-length({bag_path}): {len(synced)}")
		bag_infos.append((bag_path, synced))
		imu_infos.append((bag_path, imu_data))

		#wie werden die imu daten gespeichert
		#imu_data all imu data from one rosbag -> es gibt ein imu_file for every rosbag
		#imu
		# --imu_bag_name
		# --imu_bag_name

	total_scans = sum(len(sd) for _, sd in bag_infos)
	print(f"\n✅ Insgesamt {total_scans} synchronisierte Paare aus {len(bag_infos)} Bags")
	store_synced_data(bag_infos, imu_infos)
	rclpy.shutdown()
