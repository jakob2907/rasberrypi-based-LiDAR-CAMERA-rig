from setuptools import find_packages, setup

package_name = 'bno085'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
            'setuptools',
            'adafruit-circuitpython-bno08x',
            'adafruit-blinka'
    ],
    zip_safe=True,
    maintainer='pi-mid70',
    maintainer_email='pi-mid70@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bno085_test = bno085.test_bno085:main',
            'bno085_publisher = bno085.bno085_publisher:main'
        ],
    },
)
