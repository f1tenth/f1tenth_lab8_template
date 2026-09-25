# Lab 8: Model Predictive Control

## I. Learning Goals

- Convex Optimization
- Linearization and Discretization
- Optimal Control

## II. Formulating the MPC problem

Before starting the lab, make sure you understand the formulation of the MPC problem from the lecture. We'll briefly go over it again here. The goal of the MPC is to generate valid input controls for $T$ steps ahead that controls the vehicle and follow the reference trajectory as close as possible.

### a. State and Dynamics Model

We use a kinematic model of the vehicle. The state space is $z=[x, y, \theta, v]$. And the input vector is $u=[a, \delta]$ where $a$ is acceleration and $\delta$ is steering angle. The kinematic model ODEs are:

$$
\dot{x}=v\cos(\theta)
$$

$$
\dot{y}=v\sin(\theta)
$$

$$
\dot{v}=a
$$

$$
\dot{\theta}=\frac{v\tan(\delta)}{L}
$$

Where $L$ is the wheelbase of the vehicle. In summary, we can write the continuous ODEs as:

$$
f(z, u)=A'z+B'u
$$

Here $A'$ and $B'$ are the original continuous system matrices. In the last step, you'll discretize and linearize the system to get new system matrices $A$, $B$ and $C$. Note that you won't have to implement these in code but you can still write out the matrix representation of $A'$ and $B'$. We highly recommend writing out the system of equation in the matrix form.

### b. Objective Function

First we formulate the objective function of the optimization. We want to minimize three objectives:
1. Deviation of the vehicle from the reference trajectory. Final state deviation weighted by Qf, other state deviations weighted by Q.
2. Influence of the control inputs. Weighted by R.
3. Difference between one control input and the next control input. Weighted by Rd.

<!-- $$\text{minimize}~~~u^TRu + (x-x_{\text{ref}})_{0,\ldots,T-1}^TQ(x-x_{\text{ref}})_{0,\ldots,T-1} + (x-x_{\text{ref}})_{T}^TQ_f(x-x_{\text{ref}})_{T} + (u_{1,\ldots,T}-u_{0,\ldots,T-1})^TR_d(u_{1,\ldots,T}-u_{0,\ldots,T-1})$$ -->

$$
\text{minimize}~~ Q_{f}\left(z_{T, r e f}-z_{T}\right)^{2}+Q \sum_{t=0}^{T-1}\left(z_{t, r e f}-z_{t}\right)^{2}+R \sum_{t=0}^{T} u_{t}^{2}+R_{d} \sum_{t=0}^{T-1}\left(u_{t+1}-u_{t}\right)^{2}
$$

### c. Constraints

We then formulate the constraints for the optimization problem. The constraints should inclue:
1. Future vehicle states must follow the linearized vehicle dynamic model.
   $$z_{t+1}=Az_t+Bu_t+C$$
2. Initial state in the plan for the current horizon must match current vehicle state.
   $$z_{0}=z_{\text{curr}}$$
3. Inputs generated must be within vehicle limits.
   $$a_{\text{min}} \leq a \leq a_{\text{max}}$$
   $$\delta_{\text{min}} \leq \delta \leq \delta_{\text{max}}$$

## III. Linearization and Discretization

In order to formulate the problem into a Quadratic Programming (QP), we need to first discretize the dynamical system, and also linearize it around some point.

### a. Discretization

Here we'll use Forward Euler discretization since it's the easiest. Other methods like RK4/6 should also work. We discretize with sampling time $dt$, which you can pick as a parameter to tune. Thus, we can express the system equation as:

$$z_{t+1} = z_t + f(z_t, u_t)dt$$

### b. Linearization
We'll use first order Taylor expansion of the two variable function around some $\bar{z}$ and $\bar{u}$:

$$
z_{t+1}=z_t + f(z_t, u_t)dt
$$

$$
z_{t+1}=z_t + (f(\bar{z_t}, \bar{u_t}) + f'_z(\bar{z_t}, \bar{u_t})(z_t - \bar{z_t}) + f'_u(\bar{z_t}, \bar{u_t})(u_t - \bar{u_t}))dt
$$

$$
z_{t+1}=z_t + (f(\bar{z_t}, \bar{u_t}) + A'(z_t - \bar{z_t}) + B'(u_t - \bar{u_t}))dt
$$

$$
z_{t+1}=z_t + (f(\bar{z_t}, \bar{u_t}) + A'z_t - A'\bar{z_t} + B'u_t - B'\bar{u_t})dt
$$

$$
z_{t+1}=(I+dtA')z_t + dtB'u_t + (f(\bar{z_t}, \bar{u_t})- A'\bar{z_t} - B'\bar{u_t})dt
$$

$$
z_{t+1} = Az_t + Bu_t + C
$$

You can then derive what are matrices $A$, $B$, and $C$.

## IV. Reference Trajectory

You'll need to create a reference trajectory that has velocity attached to each waypoint that you create. (You can follow instructions from the Pure Pursuit lab to create waypoints). For a smooth velocity profile, you can use the curvature information on the waypoint spline you've created to interpolate between a maximum and a minimum velocity. Make sure the reference has the same states as the vehicle model you've set up in the optimization problem. Please refer to the Pure Pursuit lab for instructions and hints on how to log and visualize waypoints.

## V. Setting up the Optimization

In Python, we'll be using CVXPY to set up the optimization problem with the OSQP solver. Most of the problem set up and potential code optimization that speeds up the MPC are already done for you. Your first task is to fill in the objective function and the constraints for the MPC in the function `mpc_prob_init()`. The second task is to fill in the `odom_callback`. You can find missing parts in the code by searching for `TODO` tags. There is also a C++ scaffold in `mpc/src/mpc_node.cpp` that mirrors the Python structure and uses [OSQP-Eigen](https://github.com/gbionics/osqp-eigen), with the state/input indexing provided in `mpc_utils.hpp` for your convenience.

**Dependencies.** The Python skeleton needs cvxpy with the OSQP solver, which are not ROS packages: `pip install cvxpy osqp` (add `--break-system-packages` if pip refuses on Ubuntu 24.04). The autograder installs them before it builds your package. The C++ scaffold needs [OSQP-Eigen](https://github.com/robotology/osqp-eigen), built from source together with [OSQP](https://github.com/osqp/osqp). **The grading image has no OSQP-Eigen**: the skeleton's `CMakeLists.txt` skips the C++ node when it is not found (so a Python team's package builds anywhere), which means a C++ team must vendor osqp and osqp-eigen inside the package, or wait for the course to announce a grading image that has them.

## VI. Visualization

It might be helpful to visualize the current selected segment of reference path and the predicted trajectory from MPC to debug.

## VII. What the autograder runs

In the [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros/tree/dev-jazzy) simulator your node drives Levine. Set these in `config/sim.yaml` and your laptop run is the autograder's run: `map_path: 'maps/levine_blocked'`, `sx: -12.0`, `sy: 0.0`, `stheta: 0.0` (the stock start pose), driving counter-clockwise. The map comes with a centerline, so the simulator counts your laps (`/ego_racecar/lap_count`, and a `completed lap N, last lap X s` line in the bridge log). Those lap times are exactly what the autograder reports and what the leaderboard ranks: a lap runs from the finish line back to it, the 12 m from the start pose to the line are a run-up, so every lap is a flying lap.

**One launch file.** The autograder starts your code with `mpc/launch/levine_launch.py`, and nothing else:

```bash
ros2 launch mpc levine_launch.py
```

It is yours to edit: set `EXECUTABLE` to the node you wrote (`mpc_node.py` for Python, `mpc_node` for C++), put your tuned values in `PARAMETERS` (or a `.yaml` config file), and start as many nodes as you like. Start your own nodes only: the autograder runs the simulator. Without the launch file the autograder falls back to `ros2 run mpc <executable>` with no parameter file, so tuned values must then be your node's defaults. Line F of your result tells you how your code was started.

**Ship your waypoints with your package.** Put your CSV files in `mpc/waypoints/`; the skeleton's `CMakeLists.txt` installs that folder, and your node finds it with `get_package_share_directory('mpc')` (Python, `ament_index_python.packages`) or `ament_index_cpp::get_package_share_directory("mpc")` (C++). A path like `/home/you/sim_ws/...` only exists on your laptop: on the autograder your node would die at start-up.

The autograder then does two things:

1. **Checks your MPC without the simulator.** It plays the simulator's localisation for a car held still at a pose on the Levine loop (`/ego_racecar/odom`, the `map -> ego_racecar/base_link` transform, an empty `/scan`) and records what your node publishes on `/drive`. Five checks, one per part of the problem:
   - standing 0.6 m left of the middle of the south hallway, the car must steer further right than from 0.6 m right of it;
   - turned 25° left, it must steer further right than turned 25° right;
   - half way round the south-east corner (a left turn: the loop runs counter-clockwise) it must steer further left than in the middle of the hallway;
   - standing still (the odometry reports 0 m/s), the speed it commands must be above 0 and at most 1.5 m/s: with the acceleration bounded, the first step of the plan can only add `a_max × dt`;
   - turned 60° left at 2 m/s, it must steer back to the right, and never command more than the car's steering limit, 0.4189 rad.

   Each comparison only asks which way the steering changes, so your path may be a centerline or a racing line.
2. **Drives three laps in a row** of `levine_blocked`, counter-clockwise, from the stock start pose, without touching a wall. A run that ends early earns partial credit for the fraction covered; the fastest of the three laps goes to the leaderboard.

## VIII. Deliverables and Submission

**This lab is done in teams**, the same teams as lab 4. Your team is already formed — you do not create one or invite anyone. **Every member of the team runs the same command**:

```bash
gh student accept RoboRacer-Class ese-6150 lab-8-model-predictive-control
```

Whoever runs it first creates the team's shared repository, `ese-6150-lab-8-model-predictive-control-group-<n>`; everyone else gets `Repository already exists` and the same URL. All of you push to that one repository, so **pull before you push**. One submission is the whole team's submission, and every member gets the same grade.

- **Deliverable 1**: Commit your `mpc` package to your team's repository, waypoints included. Your commited code should run smoothly in simulation: three laps of Levine in a row without touching a wall. The autograder watches the run, and the leaderboard keeps your team's fastest lap.
- **Deliverable 2**: Submit links to two videos in **`SUBMISSION.md`** (YouTube unlisted, or Google Drive shared as **"Anyone with the link can view"**): the car tracking waypoints with MPC in Levine in the simulator, and on the real car in Levine hallway.

### Submitting

You can commit and push your work as often as you need, but a plain push does **not** count as a submission. When your team is ready to submit, any one of you pushes a tag named `submission` — it counts for the whole team, so agree on the commit first:

```bash
# Make sure you've pulled before or switch branches
git push                            # your commits
git tag submission
git push origin submission          # this triggers the autograder
```

The autograder builds your package, checks your MPC and drives it around Levine in the simulator, then posts your score as a **Release** on your repo (check the Releases page or the commit's status check a few minutes after you tag). To resubmit, move the tag to a new commit:

```bash
git tag -f submission
git push --force origin submission
```

The best scored `submission` push is counted as your team's final submission, and its grade is every member's grade for the lab. The leaderboard keeps your team's fastest clean lap.

**The autograder finds your work by name.** Package `mpc`, launch file `levine_launch.py` (or, without it, an executable it can start with `ros2 run mpc <executable>`, the skeleton's `mpc_node.py`), taking its pose from `/ego_racecar/odom` and publishing `AckermannDriveStamped` on `/drive`. Otherwise, the autograder will not be able to grade your work and your submission may get the wrong grade.

**Only the topics the lab needs.** Your nodes may read the simulator's localisation (`/ego_racecar/odom`, `/tf`), `/scan` and `/initialpose`, and publish on `/drive`, `/tf` and topics only your own nodes use (path and waypoint markers). The autograder watches the ROS graph while your code runs: a node that reads the lap counter or the collision flag, or publishes on a topic the simulator or the autograder listens to (`/initialpose`, which teleports the car, `/ego_racecar/odom`, the lap counter, ...) gets every line that ran your code scored 0.

## IX: Grading Rubric
- Compilation: **10** Points (autograded)
- Correct objectives and constraints: **50** Points (autograded without the simulator, 10 per check: your node is given the pose of a car standing to the left and to the right of your path, turned to the left and to the right, half way round a corner, standing still, and turned 60° off its path, and must steer the right way each time, pull away no faster than its acceleration limit allows, and keep within the steering limit)
- Working path tracker: **20** Points (autograded in simulation: three counter-clockwise laps of `levine_blocked` in a row without touching a wall; a run that ends early earns partial credit for the fraction covered; the fastest of the three laps goes to the leaderboard)
- Videos (TA-graded from the links in `SUBMISSION.md`):
   - In sim: **10** Points
   - On Car: **10** Points
