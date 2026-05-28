from glob import glob
import os

from setuptools import find_packages
from setuptools import setup


package_name = 'robotx_safe_docking_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zjy',
    maintainer_email='zjy@example.com',
    description='State estimation nodes for the safe docking WAM-V autonomy stack.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_imu_ekf_node = robotx_safe_docking_estimation.gps_imu_ekf_node:main',
            'state_estimator_verifier = robotx_safe_docking_estimation.state_estimator_verifier:main',
        ],
    },
)
