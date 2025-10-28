from setuptools import find_packages, setup

package_name = 'stretch_experiments'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hello-robot',
    maintainer_email='hello-robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'loop_drive = stretch_experiments.loop_drive:main',
            'test_ros2 = stretch_experiments.test_ros2:main',
            'test_grip = stretch_experiments.test_grip:main',
            'arm_base_move = stretch_experiments.arm_base_move:main',
            'move_and_pick_up = stretch_experiments.move_and_pick_up:main',
            'dry_run = stretch_experiments.dry_run:main',  
            'move_to_goal = stretch_experiments.move_to_goal:main',  
            'move_and_pick_up_odom = stretch_experiments.move_and_pick_up_odom:main',
            'detect_and_move = stretch_experiments.detect_and_move:main', 
            'test_controllers = stretch_experiments.test_controllers:main', 
            'detect_and_navigate = stretch_experiments.detect_and_navigate:main',
            'grasp_object = stretch_experiments.grasp_object:main',
        ],
    },
)
