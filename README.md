# Lab 8: Model Predictive Path Integral Control

## I. Learning Goals

- Discretization: Forward Euler and Runge-Kutta 4
- Model Predictive Path Integral (MPPI) Control

## II. Vehicle Dynamics

### a. State and Dynamics Model

We use the same kinematic model as in the MPC lab. The state is $x=[p_x, p_y, \theta, v]$, and the input vector is $u=[a, \delta]$ where $a$ is acceleration and $\delta$ is steering angle. The kinematic model ODEs are:

$$
\dot{p_x}=v\cos(\theta)
$$

$$
\dot{p_y}=v\sin(\theta)
$$

$$
\dot{\theta}=\frac{v\tan(\delta)}{L}
$$

$$
\dot{v}=a
$$

Where $L$ is the wheelbase of the vehicle. We can then represent the dynamics of the vehicle as:

$$
\dot{x}=f(x, u)
$$

In the MPC lab you linearized $f$ around a reference to turn the problem into a Quadratic Program. MPPI does not need to linearize since it only needs to simulate the dynamics $f$, so you can use the nonlinear model as is.

### b. Discretization

MPPI is still a discrete-time controller which works in discrete steps of $\Delta t$. As such, we need to discretize the dynamics $f$ into a function $F$ that takes the current state and input and returns the next state after $\Delta t$. We will use sample-and-hold to discretize the dynamics, where we assume the input is held constant over $\Delta t$. There are multiple ways to discretize nonlinear dynamics which we will cover next.

#### i. Forward Euler

The simplest discretization you can use is Forward Euler. It is a first-order method that approximates the next state by taking a step along the derivative at the current state. Forward Euler assumes the derivative stays constant over $\Delta t$:

$$
x_{k+1} = x_k + f(x_k, u_k)\,dt
$$

This is cheap because you only need one evaluation of $f$ per step, but it is also only first-order accurate: the error grows linearly with $\Delta t$. At high speeds (i.e > 5 m/s) and large $\Delta t$ (e.g. 0.1 s), the error can be significant. Smaller $\Delta t$ can help, but this would require more steps to cover the same seconds in your horizon, which increases the computation time.

#### ii. Runge-Kutta 4 (Bonus)

A better discretization for nonlinear dynamics is Runge-Kutta 4 (RK4). It is a fourth-order method that evaluates the derivative at multiple points in the interval and combines them to get a better estimate of the next state. The RK4 update is:

$$
k_1 = f(x_k, u_k)
$$

$$
k_2 = f\left(x_k + \tfrac{dt}{2} k_1,\, u_k\right)
$$

$$
k_3 = f\left(x_k + \tfrac{dt}{2} k_2,\, u_k\right)
$$

$$
k_4 = f\left(x_k + dt\, k_3,\, u_k\right)
$$

$$
x_{k+1} = x_k + \frac{dt}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
$$

RK4 is fourth-order accurate: halving $\Delta t$ cuts the error by about $16\times$, against $2\times$ for Euler. Each step costs four evaluations of $f$, but RK4 stays accurate at a larger $\Delta t$, so it can be faster overall.

**Bonus:** Implement RK4 as an alternative to Euler integration in your MPPI implementation and add a parameter to switch between the two. Report the difference between the two methods in `SUBMISSION.md` by rolling out the same control sequence (i.e max steering at 5 m/s for 1 s) using Euler and RK4 at your chosen $\Delta t$, and compare with Euler with a smaller $\Delta t$, and plot the three paths. Comment on the difference between the two, which one you chose, and why.

## II. Recap: the MPC problem

Before starting the lab, make sure you understand the MPC formulation from the previous lab and the MPPI algorithm from the lecture. We'll briefly go over both here. You'll have MPPI solve the same problem as MPC: generate valid control inputs for $H$ steps ahead that control the vehicle and follow the reference trajectory as closely as possible. What changes is *how* the problem is solved.

### b. Objective Function

We'll use the same objective from the MPC lab. We want to minimize:
1. Deviation of the vehicle from the reference trajectory. The final state deviation is weighted by $Q_f$, and other state deviations are weighted by $Q$.
2. Influence of the control inputs. Weighted by $R$.
3. Difference between one control input and the next control input. Weighted by $R_d$.

$$
J = \left(x_{H}-x_{H, ref}\right)^{\top} Q_{f}\left(x_{H}-x_{H, ref}\right)+\sum_{t=0}^{H-1}\left(x_{t}-x_{t, ref}\right)^{\top} Q\left(x_{t}-x_{t, ref}\right)+\sum_{t=0}^{H-1} u_{t}^{\top} R\, u_{t}+\sum_{t=0}^{H-2}\left(u_{t+1}-u_{t}\right)^{\top} R_{d}\left(u_{t+1}-u_{t}\right)
$$

Because MPPI only evaluates the cost, it does not need to be quadratic, convex or even differentiable. You are free to make your own custom cost function, such as a large penalty for going beyond the track bounds or similar.

Note that you will need to handle heading error wrap-around when computing the cost. For example, if the reference heading is 179° and the vehicle heading is -179°, the heading error is 2°, not 358°. You can achieve this either by wrapping the heading error to [-180°, 180°] or by squaring the angular distance $\Delta \theta = \text{atan2}(\sin(\alpha - \beta), \cos(\alpha - \beta))$

### c. Constraints

The controller needs to satisfy the same constraints as in the MPC lab. The constraints inclue:
1. Future vehicle states must follow the vehicle's dynamic model.
   $$\dot{x_t}=f(x_t, u_t)$$
2. Initial state in the plan for the current horizon must match current vehicle state.
   $$x_{0}=x_{\text{curr}}$$
3. Inputs generated must be within vehicle limits.
   $$a_{\text{min}} \leq a \leq a_{\text{max}}$$
   $$\delta_{\text{min}} \leq \delta \leq \delta_{\text{max}}$$

Because you can not enforce constraints with a solver for MPPI, we can enforce them by construction:

1. **Future vehicle states must follow the vehicle dynamics model.** MPPI simualtes the vehicle dynamics to generate future states, so every rollout obeys the dynamics by construction:
   $$x_{k+1}=F(x_k, u_k)$$
   where $F$ is the discretized dynamics of Section II.
2. **The initial state in the plan must match the current vehicle state.** Every rollout starts from the current vehicle state, so the first state in every rollout is the current state:
   $$x_{0}=x_{\text{curr}}$$
3. **Inputs generated must be within vehicle limits.** Clip every sampled input to the bounds:
   $$a_{\text{min}} \leq a \leq a_{\text{max}}$$
   $$\delta_{\text{min}} \leq \delta \leq \delta_{\text{max}}$$

   **Bonus:** Use a truncated sampling distribution instead of clipping to avoid sampling outside the bounds. For example, if the acceleration is limited to [-1, 1], you can sample from a truncated normal distribution with mean 0 and standard deviation 0.5, truncated to [-1, 1]. Experiment with different truncated sampling distributions like uniform, beta, etc. and report your findings in `SUBMISSION.md`.


For the rest of the constraints (e.g max speed, staying on the track), we will leave it as an exercise for you to come up with a way to enforce them. The simplest way is to add a large penalty to the cost function for violating the constraints. Feel free to experiment with different ways to enforce constraints and report your findings in `SUBMISSION.md`.

## IV. The MPPI Algorithm

MPPI approximate a solution to the following stochastic finite-time optimal control problem (FTOCP):

$$
\begin{aligned}
& \underset{u_0, \ldots, u_{T-1}}{\text{minimize}}
& & \mathbb{E}\left[J\left(x_{0}, u_0, \ldots, u_{H-1}\right)\right] \\
& \text{subject to}
& & x_{k+1}=F\left(x_k, u_k\right), \quad k=0, \ldots, H-1 \\
& & & x_0=x_{\text{curr}} \\
& & & u_t \in \mathcal{U} \subseteq \mathbb{R}^{m}, \quad k=0, \ldots, H-1
\end{aligned}
$$

The expectation is approximated via Monte Carlo sampling. MPPI *samples* many control sequences from a proposal control distribution $\mathbb{Q}_{U,\Sigma}$, simulates the resulting trajectories, and computes the cost of each trajectory. Then, the KL-divergence between the proposal distribution and the optimal distribution $\mathbb{Q}^*$ is minimized using importance sampling. The resulting update is a weighted average of the sampled control sequences, where the weights are computed from the cost of each trajectory and vary per timestep.

Each timestep, starting from the current state $x_{\text{curr}}$, the MPPI algorithm performs the following steps:

1. **Sample** $T$ control sequences of length $H$ from the proposal distribution $\mathbb{Q}_{U,\Sigma}$, where $T$ is the number of samples, $H$ is the horizon, $U$ is the mean of the propsal distribution, and $\Sigma$ is the covariance of the proposal distribution. See section II.c for which propsal distribution to use.
3. **Simulate** each control sequence using the discretized dynamics $F$ to generate a trajectory of states.
4. **Score** each trajectory using the cost function $J$.
5. **Weight** each trajectory from the cost of the trajectory and the temperature parameter $\lambda$ using importance sampling. The weight of each trajectory is given by:
$$
w^k = \frac{\exp\left(-\frac{1}{\lambda} J^k\right)}{\sum_{j=1}^{T} \exp\left(-\frac{1}{\lambda} J^j\right)}
$$
6. **Update** the mean of the proposal distribution $U$ using the weighted average of the sampled control sequences:
$$
U = \sum_{k=1}^{T} w^k u^k
$$
   **Bonus** You can also update the covariance of the proposal distribution $\Sigma$ using the weighted covariance of the sampled control sequences:
$$
\Sigma = \sum_{k=1}^{T} w^k (u^k - U)(u^k - U)^\top
$$
7. **Apply** $u_0$ to the car, then repeat the process at the next timestep.

### Parameters to tune

- **$T$, the number of samples.** More samples give a better approximation of the expectation, but take longer to compute. Fewer samples are faster, but may not find a better control sequence.
- **$H$, the horizon.** The number of timesteps to look ahead.
- **$\Delta t$, the time step.** The time between each control update.
- **$\Sigma$** The covariance of the proposal distribution. It sets how much exploration is done in the control space. A larger $\Sigma$ means more exploration, but also more noise in the control sequence.
- **$\lambda$, the temperature.** It sets how greedy the average is. As $\lambda \to 0$ the update takes only the best sample and the control gets noisy; as $\lambda$ increases, the weighting becomes more uniform. A useful check is the effective sample size, $1 / \sum_k (w^k)^2$: if it is close to $1$, $\lambda$ is too small; if it is close to $T$, it is too large.
- **$Q$, $Q_f$, $R$, $R_d$, the cost weights,** as in the MPC lab.

### Vectorize the rollouts

Simulating $T \times H$ calls to $F$ every timestep would take too long to compute if done sequentially using a for loop. Instead, you will need to vectorize the rollouts. Break down the simulation into three functions: a `step` function that takes in a state and a control input and returns the next state, a `simulate` function that takes in an initial state and a sequence of control inputs and returns the resulting trajectory of states, and a `rollout` function that takes in an initial state and a batch of control sequences and returns the resulting batch of trajectories. The `rollout` function should be vectorized over the $T$ samples, so that it can simulate all $T$ trajectories in parallel.
 
## V. Reference Trajectory

You can reuse the waypoints you made in the MPC lab as your reference trajectory. We recall the instructions here for convenience. 
   
> You'll need to create a reference trajectory that has velocity attached to each waypoint that you create. (You can follow instructions from the Pure Pursuit lab to create waypoints). For a smooth velocity profile, you can use the curvature information on the waypoint spline you've created to interpolate between a maximum and a minimum velocity. Make sure the reference has the same states as the vehicle model you've set up in the optimization problem. Please refer to the Pure Pursuit lab for instructions and hints on how to log and visualize waypoints.

## VI. Implementing MPPI

The skeleton is in `mppi/scripts/mppi_node.py`. You can find missing parts in the code by searching for `TODO` tags. Your tasks are:

1. **Dynamics.** Fill in `dynamics()` with the kinematic model $f(x, u)$, and `step_euler()` with the Euler step $F$, both vectorized over the $K$ samples. The bonus is `step_rk4()`.
2. **Rollouts and cost.** Fill in `rollout()`, which simulates all $T$ sampled sequences from the current state, and `cost()`, which scores them with the objective of Section II.
3. **MPPI update.** Fill in `mppi_step()` with steps 1 to 7 of Section IV.
4. **Callback.** Fill in `odom_callback()`: read the vehicle state, compute the reference for the next $T$ steps, run `mppi_step()` and publish the first input as an `AckermannDriveStamped`, with speed $v + a_0\,dt$ and steering $\delta_0$.

**Dependencies.** The Python skeleton only needs JAX; there is no solver to install. There is also a C++ scaffold in `mppi/src/mppi_node.cpp` that mirrors the Python structure and uses only Eigen, so you can implement MPPI in C++ if you prefer. The autograder runs your code with either language.

## VII. Visualization

It is very helpful to visualize, on top of the current segment of the reference path:
- a subset of the sampled rollouts (say 50), colored by their weight, to see where MPPI is searching;
   **WARNING:** This is very expensive to compute and render, especially on the real car, so only do it for debugging and not in your final submission.
- the trajectory of the updated nominal sequence, to see what the car plans to do.

## VIII. What the autograder runs

In the [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros/tree/dev-jazzy) simulator your node drives the **two tracks** of the Pure Pursuit lab. Set the map and start pose in `config/sim.yaml` to run the autograder's tests. The autograder runs your code with the following parameters:

| Track | `map_path` | `sx`, `sy`, `stheta` | Graded run |
| --- | --- | --- | --- |
| Levine | `'maps/levine_blocked'` | `-12.0`, `0.0`, `0.0` (default) | three laps in a row, counter-clockwise |
| Spielberg | `'maps/Spielberg'` | `14.59`, `3.92`, `-2.877` | one lap, in the direction of its centerline |

Both maps come with a centerline, so the simulator counts your laps (`/ego_racecar/lap_count`, and `completed lap N, last lap X s` line in the bridge log). 

**One launch file per track.** The autograder starts your code with the two launch files in `mppi/launch/`:

```bash
ros2 launch mppi levine_launch.py      # three laps of Levine (and the pose checks)
ros2 launch mppi spielberg_launch.py   # one lap of Spielberg
```

They are yours to edit: set `EXECUTABLE` to the node you wrote (`mppi_node.py` for Python, `mppi_node` for C++), put each track's tuned values in its `PARAMETERS` (or a `.yaml` config file), and start as many nodes as you like. Start your own nodes only: the autograder runs the simulator. Each file passes `track` (`levine` or `spielberg`), which the skeleton declares so your node knows which waypoints to load. Without a track's launch file the autograder falls back to `ros2 run mppi <executable> --ros-args -p track:=<track>` with no parameter file, so tuned values must then be your node's defaults. Line F of your result tells you how your code was started. For Spielberg the simulator ships a centerline and an optimised raceline, `maps/Spielberg_centerline.csv` and `maps/Spielberg_raceline.csv`: you may use either, reshape them, or make your own, with a speed profile that suits the track.

**Ship your waypoints with your package.** Put your CSV files in `mppi/waypoints/`; the skeleton's `CMakeLists.txt` installs that folder, and your node finds it with `get_package_share_directory('mppi')` (Python, `ament_index_python.packages`) or `ament_index_cpp::get_package_share_directory("mppi")` (C++). A path like `/home/you/sim_ws/...` only exists on your laptop: on the autograder your node would die at start-up.

The autograder then does three things:

1. **Checks your MPPI without the simulator.** It plays the simulator's localisation for a car held still at a pose on the Levine loop (`/ego_racecar/odom`, the `map -> ego_racecar/base_link` transform, an empty `/scan`) and records what your node publishes on `/drive`. Five checks, one per part of the problem:
   - standing 0.6 m left of the middle of the south hallway, the car must steer further right than from 0.6 m right of it;
   - turned 25° left, it must steer further right than turned 25° right;
   - half way round the south-east corner (a left turn: the loop runs counter-clockwise) it must steer further left than in the middle of the hallway;
   - standing still (the odometry reports 0 m/s), the speed it commands must be above 0 and at most 1.5 m/s: with the acceleration clipped to its limit, the first step of the plan can only add `a_max × dt`;
   - turned 60° left at 2 m/s, it must steer back to the right, and never command more than the car's steering limit, 0.4189 rad.

   Each comparison only asks which way the steering changes, so your path may be a centerline or a racing line. MPPI samples at random, so make sure your node gives the same answer every time it sees the same pose: with too few samples or too small a $\lambda$, the first command is mostly noise.
2. **Drives three laps in a row** of `levine_blocked`, counter-clockwise, from the stock start pose, without touching a wall. A run that ends early earns partial credit for the fraction covered; the fastest of the three laps goes to the Levine leaderboard.
3. **Drives one lap of Spielberg** from the start pose above, without touching a wall, with partial credit the same way; the lap time goes to the Spielberg leaderboard.

## IX. Deliverables and Submission

**This lab is done in teams**, the same teams as lab 7. Your team is already formed — you do not create one or invite anyone. **Every member of the team runs the same command**:

```bash
gh student accept RoboRacer-Class ese-6150 lab-8-model-predictive-path-integral
```

Whoever runs it first creates the team's shared repository, `ese-6150-lab-8-model-predictive-path-integral-group-<n>`; everyone else gets `Repository already exists` and the same URL. All of you push to that one repository, so **pull before you push**. One submission is the whole team's submission, and every member gets the same grade.

- **Deliverable 1**: Commit your `mppi` package to your team's repository, waypoints included. Your committed code should run smoothly in simulation: three laps of Levine in a row and one lap of Spielberg, without touching a wall. The autograder watches both runs, and the leaderboards keep your team's fastest lap on each track.
- **Deliverable 2**: Submit links to two videos in **`SUBMISSION.md`** (YouTube unlisted, or Google Drive shared as **"Anyone with the link can view"**): the car tracking waypoints with MPPI in Levine in the simulator, with the sampled rollouts visible in RViz, and on the real car in Levine hallway.
- **Bonus**: The Euler and RK4 comparison of Section III.b in **`SUBMISSION.md`**.

### Submitting

You can commit and push your work as often as you need, but a plain push does **not** count as a submission. When your team is ready to submit, any one of you pushes a tag named `submission` — it counts for the whole team, so agree on the commit first:

```bash
# Make sure you've pulled before or switch branches
git push                            # your commits
git tag submission
git push origin submission          # this triggers the autograder
```

The autograder builds your package, checks your MPPI and drives it around Levine and Spielberg in the simulator, then posts your score as a **Release** on your repo (check the Releases page or the commit's status check a few minutes after you tag). To resubmit, move the tag to a new commit:

```bash
git tag -f submission
git push --force origin submission
```

The best scored `submission` push is counted as your team's final submission, and its grade is every member's grade for the lab. The two leaderboards are independent: each keeps your team's fastest clean lap on its track, and each track has its own launch file, so one submission carries a tuning for Levine and another for Spielberg.

**The autograder finds your work by name.** Package `mppi`, launch files `levine_launch.py` and `spielberg_launch.py` (or, without them, an executable it can start with `ros2 run mppi <executable> --ros-args -p track:=<track>`, the skeleton's `mppi_node.py`), taking its pose from `/ego_racecar/odom` and publishing `AckermannDriveStamped` on `/drive`. Otherwise, the autograder will not be able to grade your work and your submission may get the wrong grade.

**Only the topics the lab needs.** Your nodes may read the simulator's localisation (`/ego_racecar/odom`, `/tf`), `/scan` and `/initialpose`, and publish on `/drive`, `/tf` and topics only your own nodes use (path, rollout and waypoint markers). The autograder watches the ROS graph while your code runs: a node that reads the lap counter or the collision flag, or publishes on a topic the simulator or the autograder listens to (`/initialpose`, which teleports the car, `/ego_racecar/odom`, the lap counter, ...) gets every line that ran your code scored 0.

## X: Grading Rubric
- Compilation: **10** Points (autograded)
- Correct dynamics, cost and update: **50** Points (autograded without the simulator, 10 per check: your node is given the pose of a car standing to the left and to the right of your path, turned to the left and to the right, half way round a corner, standing still, and turned 60° off its path, and must steer the right way each time, pull away no faster than its acceleration limit allows, and keep within the steering limit)
- Working path tracker: **30** Points
  - **20** Points (autograded in simulation: three counter-clockwise laps of `levine_blocked` in a row without touching a wall; a run that ends early earns partial credit for the fraction covered; the fastest of the three laps goes to the Levine leaderboard)
  - **10** Points (autograded in simulation: one lap of `Spielberg` without touching a wall, with partial credit; the lap time goes to the Spielberg leaderboard)
- Videos (TA-graded from the links in `SUBMISSION.md`):
   - In sim: **5** Points
   - On Car: **5** Points
- Bonus: RK4 discretization: **5** Points (TA-graded from the Euler and RK4 comparison in `SUBMISSION.md`)
