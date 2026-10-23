"""What the autograder runs on Levine (levine_blocked, three laps, and the pose checks):

    ros2 launch mppi levine_launch.py

Everything the graded run needs goes here: the nodes to start, the waypoints
to follow and the parameter values that suit Levine (spielberg_launch.py is
Spielberg's). Start your own nodes only: the simulator is already running.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

# scripts/mppi_node.py
EXECUTABLE = 'mppi_node.py'

# `track` tells the skeleton which waypoints to load; add this track's values
# for the other parameters your node declares, e.g. 'max_speed': 6.0, or give
# it a full .yaml config file
PARAMETERS = {'track': 'levine'}


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mppi',
            executable=EXECUTABLE,
            output='screen',
            parameters=[PARAMETERS],
        ),
    ])
