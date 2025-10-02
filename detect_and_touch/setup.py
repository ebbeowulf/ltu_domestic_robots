from setuptools import find_packages, setup

package_name = 'detect_and_touch'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Example: Include all launch files
        # ('share/' + package_name + '/launch', ['launch/my_launch_file.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ebeowulf',
    maintainer_email='user@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    
    # This is how you create executable Python scripts in ROS 2
    entry_points={
        'console_scripts': [
             # <executable> = <package>.<file>:<function>
            'test_ros2 = test_ros2.test_ros2:main',
        ],
    },
)
