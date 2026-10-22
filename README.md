# Lab 8: Model Predictive Path Integral Control

## I. Learning Goals

- Sampling-based Optimal Control
- Discretization: Forward Euler and Runge-Kutta 4
- Model Predictive Path Integral (MPPI) Control

## II. Recap: the MPC problem

Before starting the lab, make sure you understand the MPC formulation from the previous lab and the MPPI algorithm from the lecture. We'll briefly go over both here. MPPI solves the same problem as MPC: generate valid input controls for $T$ steps ahead that control the vehicle and follow the reference trajectory as closely as possible. What changes is *how* the problem is solved.

### a. State and Dynamics Model

We use the same kinematic model of the vehicle as in the MPC lab. The state space is $z=[x, y, \theta, v]$, and the input vector is $u=[a, \delta]$ where $a$ is acceleration and $\delta$ is steering angle. The kinematic model ODEs are:

$$
\dot{x}=v\cos(\theta)
$$

$$
\dot{y}=v\sin(\theta)
$$

$$
\dot{\theta}=\frac{v\tan(\delta)}{L}
$$

$$
\dot{v}=a
$$

Where $L$ is the wheelbase of the vehicle. In summary, we can write the continuous ODEs as:

$$
\dot{z}=f(z, u)
$$

In the MPC lab you linearized $f$ around a reference to turn the problem into a Quadratic Program. MPPI does not need to linearize since it only needs to simulate the dynamics $f$, so you can use the nonlinear model as is.

### b. Objective Function

We'll use the same objective from the MPC lab. We want to minimize:
1. Deviation of the vehicle from the reference trajectory. The final state deviation is weighted by $Q_f$, and other state deviations are weighted by $Q$.
2. Influence of the control inputs. Weighted by $R$.
3. Difference between one control input and the next control input. Weighted by $R_d$.

$$
J = \left(z_{T}-z_{T, ref}\right)^{\top} Q_{f}\left(z_{T}-z_{T, ref}\right)+\sum_{t=0}^{T-1}\left(z_{t}-z_{t, ref}\right)^{\top} Q\left(z_{t}-z_{t, ref}\right)+\sum_{t=0}^{T-1} u_{t}^{\top} R\, u_{t}+\sum_{t=0}^{T-2}\left(u_{t+1}-u_{t}\right)^{\top} R_{d}\left(u_{t+1}-u_{t}\right)
$$

Because MPPI only evaluates the cost, it does not need to be quadratic, convex or even differentiable. You are free to make your own custom cost function, such as a large penalty for going beyond the track bounds or similar.

**Wrap the heading error.** $\theta - \theta_{ref}$ must be wrapped to $[-\pi, \pi]$ before it is squared, otherwise a car heading at $179°$ with a reference at $-179°$ looks $358°$ off.

### c. Constraints

The MPC lab had three constraints. Because you can not enforce constraints with a solver for MPPI, we can enforce them by construction:
1. **Future vehicle states must follow the vehicle dynamics model.** Every candidate trajectory is produced by simulating the model forward, so it obeys the dynamics exactly. This time it is the nonlinear model, not a linearization:
   $$z_{t+1}=F(z_t, u_t)$$
   where $F$ is the discretized dynamics of Section III.
2. **The initial state in the plan must match the current vehicle state.** Every rollout starts from it:
   $$z_{0}=z_{\text{curr}}$$
3. **Inputs generated must be within vehicle limits.** Clip every sampled input:
   $$a_{\text{min}} \leq a \leq a_{\text{max}}$$
   $$\delta_{\text{min}} \leq \delta \leq \delta_{\text{max}}$$

State limits (maximum speed, staying on the track) become penalties in the cost.

## III. Discretization

The controller works in discrete steps of $dt$, which you can pick as a parameter to tune, so we need a function $F$ that takes the state $z_t$ and the input $u_t$ and returns the state $z_{t+1}$ one step later. The input is held constant over the step. In the MPC lab this step was folded into the linearization; here you implement $F$ yourself, and it is the function your rollouts call $K \times T$ times per control cycle.

### a. Forward Euler

Forward Euler assumes the derivative stays constant over the whole step:

$$
z_{t+1} = z_t + f(z_t, u_t)\,dt
$$

It is cheap (one evaluation of $f$ per step) but only first-order accurate: the error over the horizon shrinks linearly with $dt$. At racing speeds the error is visible: the heading changes during the step, but Euler moves the car along the heading it had at the start, so predicted trajectories cut corners and drift from what the car actually does. Smaller $dt$ helps, but then you need more steps to see the same distance ahead.

### b. Runge-Kutta 4 (Bonus)

RK4 evaluates $f$ four times per step, at the start, twice at the midpoint and at the end, and averages the slopes:

$$
k_1 = f(z_t, u_t)
$$

$$
k_2 = f\left(z_t + \tfrac{dt}{2} k_1,\, u_t\right)
$$

$$
k_3 = f\left(z_t + \tfrac{dt}{2} k_2,\, u_t\right)
$$

$$
k_4 = f\left(z_t + dt\, k_3,\, u_t\right)
$$

$$
z_{t+1} = z_t + \frac{dt}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
$$

RK4 is fourth-order accurate: halving $dt$ cuts the error by about $16\times$, against $2\times$ for Euler. Each step costs four evaluations of $f$, but RK4 stays accurate at a larger $dt$, so you can see further ahead with fewer steps.

**For the bonus**, implement RK4 next to Euler, make it selectable by a parameter, and show the difference in `SUBMISSION.md`: roll the same control sequence (for example full steering at 5 m/s for 1 s) through Euler and RK4 at your $dt$, and through Euler at a very small $dt$ (say $0.001$ s) as ground truth, and plot the three paths. A sentence on which one you raced with, and why, completes it.

## IV. The MPPI Algorithm

MPC asks a solver for the best control sequence. MPPI instead *samples* many control sequences around its current guess, simulates each one, and averages them, giving more weight to the ones with a lower cost. Repeat every control cycle and the average moves toward the optimum.

MPPI keeps a nominal control sequence $U = (u_0, \ldots, u_{T-1})$ between cycles. Each cycle, starting from the current state $z_{\text{curr}}$:

1. **Sample** $K$ noise sequences, $\epsilon^k_t \sim \mathcal{N}(0, \Sigma)$ for $k = 1, \ldots, K$ and $t = 0, \ldots, T-1$.
2. **Perturb and clip** the nominal sequence, $v^k_t = \text{clip}(u_t + \epsilon^k_t)$, then set $\epsilon^k_t = v^k_t - u_t$ so the noise is the perturbation that was actually applied.
3. **Roll out** each sequence through the discretized dynamics, $z^k_0 = z_{\text{curr}}$, $z^k_{t+1} = F(z^k_t, v^k_t)$.
4. **Score** each rollout with the objective of Section II, plus the MPPI control-cost term that keeps the update unbiased:
   $$S^k = J(z^k, v^k) + \lambda \sum_{t=0}^{T-1} u_t^{\top} \Sigma^{-1} \epsilon^k_t$$
5. **Weight** each rollout by its exponentiated cost. Subtract the minimum cost $\rho = \min_k S^k$ first, or the exponentials underflow to zero:
   $$w^k = \frac{\exp\left(-\frac{1}{\lambda}(S^k - \rho)\right)}{\sum_{j=1}^{K} \exp\left(-\frac{1}{\lambda}(S^j - \rho)\right)}$$
6. **Update** the nominal sequence with the weighted average of the noise:
   $$u_t \leftarrow u_t + \sum_{k=1}^{K} w^k \epsilon^k_t$$
7. **Apply** $u_0$ to the car, then **shift** the sequence one step to warm-start the next cycle: $u_t \leftarrow u_{t+1}$, and repeat the last input at the end.

### Parameters to tune

- **$K$, the number of samples.** More samples give a better estimate and a smoother control, at a higher cost per cycle. Expect hundreds to a few thousand.
- **$T$ and $dt$, the horizon.** $T \cdot dt$ seconds ahead; at speed you need to see the next corner.
- **$\Sigma$, the sampling noise.** One standard deviation per input. Too small and MPPI cannot find a better sequence; too large and most samples are wasted hitting the input limits.
- **$\lambda$, the temperature.** It sets how greedy the average is. As $\lambda \to 0$ the update takes only the best sample and the control gets noisy; as $\lambda$ grows every sample counts equally and the update stops improving anything. A useful check is the effective sample size, $1 / \sum_k (w^k)^2$: if it is close to $1$, $\lambda$ is too small; if it is close to $K$, it is too large.
- **$Q$, $Q_f$, $R$, $R_d$, the cost weights,** as in the MPC lab.

### Vectorize the rollouts

$K \times T$ calls to $F$ every cycle will not run at control rate as a Python loop. Write $F$ so it takes arrays of states and inputs and steps all $K$ rollouts at once: a state array of shape `(K, 4)`, an input array of shape `(K, 2)`, one NumPy expression per state. The loop over $t$ stays, the loop over $k$ disappears. The same goes for the cost: compute $S^k$ for all $k$ in one expression.

## V. Reference Trajectory

You'll need a reference trajectory that has velocity attached to each waypoint, exactly as in the MPC lab, and you can reuse the waypoints you made there. (You can follow instructions from the Pure Pursuit lab to create waypoints.) For a smooth velocity profile, you can use the curvature information on the waypoint spline you've created to interpolate between a maximum and a minimum velocity. Make sure the reference has the same states as the vehicle model, and that it is sampled at the same $dt$ as your rollouts so $z_{t,ref}$ lines up with $z_t$.

## VI. Implementing MPPI

The skeleton is in `mppi/scripts/mppi_node.py`. You can find missing parts in the code by searching for `TODO` tags. Your tasks are:

1. **The dynamics.** Fill in `dynamics()` with the kinematic model $f(z, u)$, and `step_euler()` with the Euler step $F$, both vectorized over the $K$ samples. The bonus is `step_rk4()`.
2. **The rollouts and the cost.** Fill in `rollout()`, which simulates all $K$ sampled sequences from the current state, and `cost()`, which scores them with the objective of Section II.
3. **The MPPI update.** Fill in `mppi_step()` with steps 1 to 7 of Section IV.
4. **The callback.** Fill in `odom_callback()`: read the vehicle state, compute the reference for the next $T$ steps, run `mppi_step()` and publish the first input as an `AckermannDriveStamped`, with speed $v + a_0\,dt$ and steering $\delta_0$.

**Dependencies.** The Python skeleton only needs NumPy and SciPy; there is no solver to install. There is also a C++ scaffold in `mppi/src/mppi_node.cpp` that mirrors the Python structure and uses only Eigen, so a C++ team builds in the grading image as is.

## VII. Visualization

It is very helpful to visualize, on top of the current segment of the reference path:
- a subset of the sampled rollouts (say 50), colored by their weight, to see where MPPI is searching;
- the trajectory of the updated nominal sequence, to see what the car plans to do.

If the samples all bunch on one side of the reference, or they all leave the track, it is easy to see here and hard to see anywhere else.

## VIII. What the autograder runs

In the [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros/tree/dev-jazzy) simulator your node drives the **two tracks** of the Pure Pursuit lab. Set the map and start pose in `config/sim.yaml` and your laptop run is the autograder's run:

| Track | `map_path` | `sx`, `sy`, `stheta` | Graded run |
| --- | --- | --- | --- |
| Levine | `'maps/levine_blocked'` | `-12.0`, `0.0`, `0.0` (the stock start pose) | three laps in a row, counter-clockwise |
| Spielberg | `'maps/Spielberg'` | `14.59`, `3.92`, `-2.877` | one lap, in the direction of its centerline |

Both maps come with a centerline, so the simulator counts your laps (`/ego_racecar/lap_count`, and a `completed lap N, last lap X s` line in the bridge log). Those lap times are exactly what the autograder reports and what the leaderboards rank: a lap runs from the finish line back to it, and both start poses sit before the line (12 m on Levine, 15 m on Spielberg), so every lap is a flying lap.

**One launch file per track.** The autograder starts your code with the two launch files in `mppi/launch/`, and nothing else:

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
