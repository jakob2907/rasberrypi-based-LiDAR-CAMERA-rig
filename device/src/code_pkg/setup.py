from setuptools import find_packages, setup

package_name = 'code_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	('share/' + package_name + '/launch', ['launch/all_sensors.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi-mid70',
    maintainer_email='jakob.wehr@gmx.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bgra2bgr = code_pkg.bgra2bgr:main',
            'generate_depth_image = code_pkg.generate_depth_image_from_rosbag:main',
            'calibrate_intrinsics = code_pkg.camera_calibration_intrinsic:main',
            'image_rotate_node = code_pkg.image_rotate:main',
            'generate_output = code_pkg.generate_output_from_rosbag:main'
        ],
        'rclpy_components':[
            'ImageRotateComponent = code_pkg.image_rotate:ImageRotateNode',
        ],
    },
)
