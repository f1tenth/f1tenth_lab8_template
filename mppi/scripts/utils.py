# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Utility functions for the Kinematic MPPI waypoint tracker

Author: Hongrui Zheng, Johannes Betz, Ahmad Amine
"""
import numpy as np


def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.sum((point - trajectory[:-1, :]) * diffs, axis=1)
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + t[:, None] * diffs
    dists = np.linalg.norm(point - projections, axis=1)
    i = int(np.argmin(dists))
    return projections[i], dists[i], t[i], i


def calc_interpolated_ref_trajectory(x, y, cx, cy, cyaw, cv, dt, H):
    """
    Calculate the reference trajectory for the next H steps.

    Finds the nearest point on the waypoints, then moves forward along them at
    the waypoints' speed for H steps of dt. All states are linearly
    interpolated between waypoints.

    Args:
        x (float): Current x position
        y (float): Current y position
        cx (numpy.ndarray): x positions of waypoints
        cy (numpy.ndarray): y positions of waypoints
        cyaw (numpy.ndarray): yaw angles at waypoints
        cv (numpy.ndarray): velocities at waypoints
        dt (float): Time step
        H (int): Horizon (returns H+1 points)

    Returns:
        ref_traj (numpy.ndarray, (H+1, 4)): reference states [x, y, yaw, v],
            ref_traj[0] is the point nearest to the car
    """
    ncourse = len(cx)

    # Distance between waypoints (assumes uniform spacing)
    dl = np.sqrt((cx[1] - cx[0]) ** 2 + (cy[1] - cy[0]) ** 2)

    # Nearest point and its position t in [0, 1] on segment ind_current
    _, _, t_current, ind_current = nearest_point(np.array([x, y]), np.column_stack((cx, cy)))

    # Progress along the waypoints, in units of segments from ind_current
    t_list = np.zeros(H + 1)
    t_list[0] = t_current
    speed = (1.0 - t_current) * cv[ind_current] + t_current * cv[(ind_current + 1) % ncourse]
    for i in range(1, H + 1):
        t_list[i] = t_list[i - 1] + speed * dt / dl
        t_frac = t_list[i] % 1.0
        seg = (int(t_list[i]) + ind_current) % ncourse
        speed = (1.0 - t_frac) * cv[seg] + t_frac * cv[(seg + 1) % ncourse]

    # Interpolate all states between waypoints ind and ind + 1
    ind = (np.floor(t_list).astype(int) + ind_current) % ncourse
    ind_next = (ind + 1) % ncourse
    t_frac = t_list % 1.0
    yaw_diff = np.arctan2(np.sin(cyaw[ind_next] - cyaw[ind]), np.cos(cyaw[ind_next] - cyaw[ind]))

    return np.column_stack((
        (1.0 - t_frac) * cx[ind] + t_frac * cx[ind_next],
        (1.0 - t_frac) * cy[ind] + t_frac * cy[ind_next],
        cyaw[ind] + t_frac * yaw_diff,
        (1.0 - t_frac) * cv[ind] + t_frac * cv[ind_next],
    ))
