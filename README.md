# Virtual RobotX (VRX)
This repository is the home to the source code and software documentation for the VRX simulation environment, which supports simulation of unmanned surface vehicles in marine environments.
* Designed in coordination with RobotX organizers, this project provides arenas and tasks similar to those featured in past and future RobotX competitions, as well as a description of the WAM-V platform.
* For RobotX competitors this simulation environment is intended as a first step toward developing tools prototyping solutions in advance of physical on-water testing.
* We also welcome users with simulation needs beyond RobotX. As we continue to improve the environment, we hope to offer support to a wide range of potential applications.

## Now supporting Gazebo Sim and ROS 2 by default
We're happy to announce with release 2.0 VRX has transitioned from Gazebo Classic to the newer Gazebo simulator (formerly [Ignition Gazebo](https://www.openrobotics.org/blog/2022/4/6/a-new-era-for-gazebo)). 
* Gazebo Garden and ROS 2 are now default prerequisites for VRX.
* This is the recommended configuration for new users.
* Users who wish to continue running Gazebo Classic and ROS 1 can still do so using the `gazebo_classic` branch of this repository. 
  * Tutorials for VRX Classic will remain available on our Wiki.
  * VRX Classic will transition from an officially supported branch to a community supported branch by Spring 2023.

## The VRX Competition
The VRX environment is also the "virtual venue" for the [VRX Competition](https://github.com/osrf/vrx/wiki). Please see our Wiki for tutorials and links to registration and documentation relevant to the virtual competition. 

![VRX](images/sydney_regatta_gzsim.png)
![Ubuntu CI](https://github.com/osrf/vrx/workflows/Ubuntu%20CI/badge.svg)

## Getting Started

 * Watch the [Release 2.3 Highlight Video](https://vimeo.com/851696025).
 * The [VRX Wiki](https://github.com/osrf/vrx/wiki) provides documentation and tutorials.
 * The instructions assume a basic familiarity with the ROS environment and Gazebo.  If these tools are new to you, we recommend starting with the excellent [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)
 * For technical problems, please use the [project issue tracker](https://github.com/osrf/vrx/issues) to describe your problem or request support. 

## Safe Docking Quick Start

This branch adds a first safe-bay docking simulation milestone. The launch file
`vrx_gz/launch/safe_docking.launch.py` starts the `safe_docking_task` world,
spawns a WAM-V from `vrx_gz/config/safe_docking_wamv.yaml`, and can optionally
start RViz with `vrx_gz/config/safe_docking.rviz`.

On ROS 2 Humble, use the Gazebo Garden bridge packages. If these are not
available on the machine, ask an administrator to install:

```bash
sudo apt update
sudo apt install \
  ros-humble-ros-gzgarden-interfaces \
  ros-humble-ros-gzgarden-bridge \
  ros-humble-ros-gzgarden-sim \
  ros-humble-rviz2 \
  ros-humble-xacro \
  python3-scipy \
  python3-sdformat13
```

Build from a shell using system Python, not a Conda Python environment. If the
prompt shows `(base)`, run `conda deactivate` first.

```bash
cd /home/zjy/vrx_docking
source /opt/ros/humble/setup.bash
colcon build --packages-up-to vrx_gz \
  --cmake-clean-cache \
  --event-handlers console_direct+ \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

Launch the visual simulation. The extra `GZ_SIM_RESOURCE_PATH` entries let
Gazebo resolve WAM-V meshes from `model://wamv_description/...` and
`model://wamv_gazebo/...`.

```bash
cd /home/zjy/vrx_docking
source /opt/ros/humble/setup.bash
source install/setup.bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(pwd)/install/wamv_description/share:$(pwd)/install/wamv_gazebo/share

ros2 launch vrx_gz safe_docking.launch.py
```

To launch Gazebo and RViz together, enable the RViz launch argument. RViz starts
after a short delay and subscribes to the WAM-V camera and LiDAR topics.

```bash
ros2 launch vrx_gz safe_docking.launch.py launch_rviz:=True
```

For machines without working OpenGL / render-backed sensors, use the no-render
smoke-test configuration:

```bash
ros2 launch vrx_gz safe_docking.launch.py \
  headless:=True \
  config_file:=$(pwd)/install/vrx_gz/share/vrx_gz/config/safe_docking_wamv_norender.yaml
```

Inspect sensor topics from another sourced terminal:

```bash
ros2 topic list | grep sensor
ros2 topic hz /wamv/sensors/cameras/front_camera_sensor/optical/image_raw
ros2 topic hz /wamv/sensors/lidars/lidar_wamv_sensor/scan
ros2 topic hz /wamv/sensors/lidars/lidar_wamv_sensor/points
```

Use `Ctrl-C` to stop launches cleanly. Avoid `Ctrl-Z`, which only suspends the
launch and can leave old Gazebo / bridge processes around.

### Safe Docking Notes From 2026-05-28

- Verified the branch builds on Ubuntu 22.04 / ROS 2 Humble with Gazebo Garden
  bridge packages and `python3-sdformat13`.
- Confirmed the safe docking world reaches `ScoringPlugin::OnRunning`, starts
  dock bay detector topics, and spawns WAM-V with camera, LiDAR, GPS, IMU, and
  thruster bridges.
- Extended the safe docking task running duration from 300 seconds to 3600
  seconds.
- Added `launch_rviz:=True` and a safe docking RViz config using the active ROS
  sensor topics under `/wamv/sensors/...`.
- Added and verified a GPS/IMU planar inertial EKF state estimator. State
  estimation is achieved for the current simulation baseline.
- Replaced the first path-guided controller with a SciPy / SLSQP cubic
  B-spline trajectory tracker for the two-thruster WAM-V. Perception and
  planning are still intentionally not implemented.
- The controller now fails safe with zero thrust if the B-spline trajectory is
  infeasible or SciPy is unavailable.

## USV Dynamics Model For Safe Docking

For the first autonomy baseline, model the WAM-V as a planar 3-DOF surface
vehicle in the local ENU frame. The local frame origin should initially be the
first valid GPS fix:

```text
x: East position in meters
y: North position in meters
psi: yaw angle in radians
```

Use the local-frame state:

```math
x_{\mathrm{state}} =
\begin{bmatrix}
x & y & \psi & \dot{x} & \dot{y} & \dot{\psi}
\end{bmatrix}^{T}
```

The control input is the left and right thruster force command. In the current
Gazebo setup these commands are published directly as thrust forces in Newtons:

```math
T =
\begin{bmatrix}
T_L & T_R
\end{bmatrix}^{T}
```

### Kinematics

The first three equations are directly from the state definition:

```math
\dot{x}_{\mathrm{state}} =
\begin{bmatrix}
\dot{x} & \dot{y} & \dot{\psi} & \ddot{x} & \ddot{y} & \ddot{\psi}
\end{bmatrix}^{T}
```

Define the body-to-local yaw rotation:

```math
R(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}
```

The local linear velocity can be converted into body-frame surge and sway
velocity for hydrodynamic damping:

```math
\begin{aligned}
\left[
\begin{array}{c}
u \\
v
\end{array}
\right]
&=
R(\psi)^{T}
\left[
\begin{array}{c}
\dot{x} \\
\dot{y}
\end{array}
\right]
\end{aligned}
```

```math
u = \dot{x}\cos\psi + \dot{y}\sin\psi
```

```math
v = -\dot{x}\sin\psi + \dot{y}\cos\psi
```

### Thruster Allocation

Assume two fixed aft thrusters, no lateral thruster, and symmetric spacing.
Let `b` be the lateral distance between the two thrusters and `l = b / 2`.

```math
\begin{aligned}
\left[
\begin{array}{c}
F_x \\
F_y \\
\tau_z
\end{array}
\right]
&=
B(\psi)
\left[
\begin{array}{c}
T_L \\
T_R
\end{array}
\right]
\end{aligned}
```

```math
B(\psi) =
\begin{bmatrix}
\cos\psi & \cos\psi \\
\sin\psi & \sin\psi \\
-l & l
\end{bmatrix}
```

Therefore:

```math
F_x = (T_L + T_R)\cos\psi
```

```math
F_y = (T_L + T_R)\sin\psi
```

```math
\tau_z = l(T_R - T_L)
```

`F_x` and `F_y` are local-frame force components. `tau_z` is the yaw torque
about the local up axis. Because `B(psi)` is `3 x 2`, the WAM-V with two fixed
aft thrusters is underactuated: it cannot command arbitrary `F_x`, `F_y`, and
`tau_z` independently.

### Damping

Hydrodynamic damping is most naturally modeled in the vessel body frame because
water resistance depends on motion relative to the hull:

```math
F_{D,\mathrm{body}} =
\begin{bmatrix}
-(d_u + d_{uu}|u|)u \\
-(d_v + d_{vv}|v|)v
\end{bmatrix}
```

Rotate it back into the local frame:

```math
F_{D,\mathrm{local}} = R(\psi)F_{D,\mathrm{body}}
```

Expanded:

```math
\begin{aligned}
F_{D,x}
&=
-\cos\psi(d_u + d_{uu}|u|)u
+\sin\psi(d_v + d_{vv}|v|)v
\end{aligned}
```

```math
\begin{aligned}
F_{D,y}
&=
-\sin\psi(d_u + d_{uu}|u|)u
-\cos\psi(d_v + d_{vv}|v|)v
\end{aligned}
```

Yaw damping is:

```math
\tau_{D,z} = -(d_r + d_{rr}|\dot{\psi}|)\dot{\psi}
```

### Rotation Simplification

The body angular velocity relative to the local frame can be written as:

```math
\omega_{BW} = p\,x_b + q\,y_b + r\,z_b
```

For planar docking:

```math
p = 0,\quad q = 0,\quad r = \dot{\psi}
```

```math
\omega_{BW} = \dot{\psi}\,z_b
```

The full rigid-body rotational equation is:

```math
\tau = I\dot{\omega} + \omega \times (I\omega)
```

For yaw-only motion and a body inertia tensor aligned with the principal axes,
`omega_BW` and `I * omega_BW` are parallel, so:

```math
\omega_{BW} \times (I\omega_{BW}) = 0
```

Therefore the yaw equation reduces to:

```math
I_z\ddot{\psi} = \tau_z + \tau_{D,z}
```

### Local-Frame Dynamics

The complete dynamics are:

```math
\frac{dx}{dt} = \dot{x}
```

```math
\frac{dy}{dt} = \dot{y}
```

```math
\frac{d\psi}{dt} = \dot{\psi}
```

```math
\begin{aligned}
\ddot{x}
&=
\frac{
(T_L + T_R)\cos\psi
-\cos\psi(d_u + d_{uu}|u|)u
+\sin\psi(d_v + d_{vv}|v|)v
}{m}
\end{aligned}
```

```math
\begin{aligned}
\ddot{y}
&=
\frac{
(T_L + T_R)\sin\psi
-\sin\psi(d_u + d_{uu}|u|)u
-\cos\psi(d_v + d_{vv}|v|)v
}{m}
\end{aligned}
```

```math
\begin{aligned}
\ddot{\psi}
&=
\frac{
l(T_R - T_L)
-(d_r + d_{rr}|\dot{\psi}|)\dot{\psi}
}{I_z}
\end{aligned}
```

with:

```math
u = \dot{x}\cos\psi + \dot{y}\sin\psi
```

```math
v = -\dot{x}\sin\psi + \dot{y}\cos\psi
```

The corresponding matrix form is:

```math
\begin{aligned}
\frac{d}{dt}
\left[
\begin{array}{c}
x \\
y \\
\psi \\
\dot{x} \\
\dot{y} \\
\dot{\psi}
\end{array}
\right]
&=
\left[
\begin{array}{c}
\dot{x} \\
\dot{y} \\
\dot{\psi} \\
(F_x + F_{D,x})/m \\
(F_y + F_{D,y})/m \\
(\tau_z + \tau_{D,z})/I_z
\end{array}
\right]
\end{aligned}
```

For the current VRX WAM-V simulation, reasonable baseline parameters from the
model files are:

| Parameter | Value |
| --- | ---: |
| $m$ | 180.0 |
| $I_z$ | 446.0 |
| $d_u$ | 100.0 |
| $d_{uu}$ | 150.0 |
| $d_v$ | 100.0 |
| $d_{vv}$ | 100.0 |
| $d_r$ | 800.0 |
| $d_{rr}$ | 800.0 |

These values are suitable for first controller and estimator development, but
payloads and fitted simulator behavior may shift the effective mass, inertia,
and damping. Re-identify the parameters from simulation data before relying on
high-accuracy prediction.

## GPS / IMU EKF State Estimator

The initial state estimator lives in `robotx_safe_docking_estimation`. It uses a
planar inertial EKF with GPS position updates and IMU yaw as a compass / AHRS
placeholder. The internal EKF state is:

```math
x_{\mathrm{ekf}} =
\begin{bmatrix}
x & y & \psi & \dot{x} & \dot{y} & b_{ax} & b_{ay} & b_{gz}
\end{bmatrix}^{T}
```

where `b_ax` and `b_ay` are body-frame accelerometer biases and `b_gz` is the
gyro-z bias. The published controller-facing state is:

```math
x_{\mathrm{pub}} =
\begin{bmatrix}
x & y & \psi & \dot{x} & \dot{y} & \dot{\psi}
\end{bmatrix}^{T}
```

with $\dot{\psi} = \omega_{z,m} - b_{gz}$.

At startup, the estimator holds initialization while it collects a short batch
of GPS fixes. The averaged GPS fix defines the local ENU origin. For the small
VRX task area, GPS is converted to local meters with an equirectangular WGS84
approximation:

```math
x = (\mathrm{lon} - \mathrm{lon}_0)\cos(\mathrm{lat}_0)R_e
```

```math
y = (\mathrm{lat} - \mathrm{lat}_0)R_e
```

where `R_e = 6378137.0 m`.

The estimator publishes the WAM-V reference point, not the GPS antenna point.
The safe docking WAM-V GPS is mounted at `[-0.85, 0.0]` m in the body frame, so
GPS position measurements are corrected for the yaw-dependent lever arm:

```math
\begin{aligned}
r_{\mathrm{gpsB}}
&=
\left[
\begin{array}{c}
x_{\mathrm{gpsBody}} \\
y_{\mathrm{gpsBody}}
\end{array}
\right]
\end{aligned}
```

```math
\begin{aligned}
z_{\mathrm{base}}
&=
z_{\mathrm{gpsDelta}}
- R(\psi)r_{\mathrm{gpsB}}
+ R(\psi_0)r_{\mathrm{gpsB}}
\end{aligned}
```

where `psi_0` is the first IMU yaw used by the estimator, and
$x_{\mathrm{gpsBody}}$, $y_{\mathrm{gpsBody}}$ are the configured
`gps_body_x`, `gps_body_y` offsets.

The EKF prediction uses body-frame IMU linear acceleration and yaw rate:

```math
a_B =
\begin{bmatrix}
a_{x,m} - b_{ax} \\
a_{y,m} - b_{ay}
\end{bmatrix}
```

```math
R(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}
```

```math
a_W = R(\psi)a_B
```

```math
\omega = \omega_{z,m} - b_{gz}
```

```math
x_{k+1} = x_k + \dot{x}_k\Delta t + \frac{1}{2}a_{W,x}\Delta t^2
```

```math
y_{k+1} = y_k + \dot{y}_k\Delta t + \frac{1}{2}a_{W,y}\Delta t^2
```

```math
\psi_{k+1} = \mathrm{wrap}(\psi_k + \omega\Delta t)
```

```math
\dot{x}_{k+1} = \dot{x}_k + a_{W,x}\Delta t
```

```math
\dot{y}_{k+1} = \dot{y}_k + a_{W,y}\Delta t
```

```math
b_{ax,k+1} = b_{ax,k},\quad
b_{ay,k+1} = b_{ay,k},\quad
b_{gz,k+1} = b_{gz,k}
```

GPS measurement update:

```math
z_{\mathrm{gps}} =
\begin{bmatrix}
x_{\mathrm{gps}} \\
y_{\mathrm{gps}}
\end{bmatrix}
```

```math
h_{\mathrm{gps}}(x_{\mathrm{ekf}}) =
\begin{bmatrix}
x \\
y
\end{bmatrix}
```

IMU yaw measurement update. In simulation this comes from
`sensor_msgs/msg/Imu.orientation`; in the real system it is intended to stand in
for a future compass, AHRS, INS, or dual-RTK heading source:

```math
z_{\mathrm{imu,yaw}} = \psi_{\mathrm{imu}}
```

```math
h_{\mathrm{imu,yaw}}(x_{\mathrm{ekf}}) = \psi
```

The gyro-z value is not double-counted as a separate measurement in the current
filter. It is used as the prediction input, and the published yaw rate is
$\dot{\psi} = \omega_{z,m} - b_{gz}$.

Current EKF parameters are:

```text
use_imu_orientation: true
use_gps_velocity_measurement: true

gps_body_x: -0.85
gps_body_y: 0.0

origin_init_duration_sec: 5.0
origin_min_samples: 20
origin_max_std_m: 0.5

gps_position_variance: 0.01
gps_velocity_variance: 0.05
imu_yaw_variance: 0.02
acceleration_noise_variance: 0.25
gyro_noise_variance: 0.0004
yaw_process_variance: 0.01
accel_bias_random_walk_variance: 0.0004
gyro_bias_random_walk_variance: 0.000001
unmodeled_velocity_process_variance: 0.05
```

Launch the estimator after the safe docking simulation is running:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch robotx_safe_docking_estimation estimator.launch.py
```

For development-only RMSE reporting against Gazebo pose ground truth:

```bash
ros2 launch robotx_safe_docking_estimation estimator.launch.py verify:=True
```

The verifier expects the debug ground-truth odometry topic:

```text
/wamv/sensors/position/ground_truth_odometry
```

That topic is only available when the WAM-V xacro argument
`ground_truth_enabled:=true` is used. Keep this disabled for normal autonomy
runs; enable it only in temporary verification configs or evaluator code.

The estimator publishes:

```text
/safe_docking/state     std_msgs/msg/Float64MultiArray
/safe_docking/odometry  nav_msgs/msg/Odometry
```

`/safe_docking/state` publishes the six controller-facing states followed by the
three bias estimates:

```text
[x, y, psi, x_dot, y_dot, psi_dot, b_ax, b_ay, b_gz]
```

The verifier subscribes to Gazebo pose ground truth only for offline/debug
performance reporting. Autonomy nodes must consume `/safe_docking/state` or
`/safe_docking/odometry`, not Gazebo pose or evaluator topics.

The current verified baseline used a short open-loop maneuver: about 10 seconds
of straight thrust (`T_L = 80 N`, `T_R = 80 N`) followed by about 10 seconds of
turning thrust (`T_L = 40 N`, `T_R = 120 N`). Against debug ground truth, the
lever-arm-compensated EKF achieved:

```text
samples: 4190
duration_s: 35.196

metric,rmse,mae,max_abs
x,0.033476,0.030465,0.089608
y,0.040380,0.031415,0.092828
psi,0.002228,0.001464,0.007272
x_dot,0.020328,0.017419,0.049187
y_dot,0.011758,0.009380,0.033439
psi_dot,0.009176,0.007324,0.032530
```

## RK4 Dynamics NMPC Trajectory Controller

The current controller in `robotx_safe_docking_control` is an acados nonlinear
MPC for the two fixed aft thrusters. It consumes EKF odometry, tracks feasible
synthetic references, and publishes direct left / right thrust commands. The
controller does not consume Gazebo ground truth; ground truth is used only by
recorder scripts for offline metrics and plots.

This NMPC is the active controller path. The previous B-spline lookahead /
PID-style controller remains in the branch history but is not the current
baseline.

The implemented data flow is:

```text
EKF odometry
  -> feasible synthetic reference generator
  -> acados RK4 dynamics NMPC
  -> tau_u, tau_v, tau_r prediction
  -> constrained differential-thrust allocation
  -> WAM-V left/right thrust topics
```

### Reference Generation

Synthetic references are generated by rolling out the nominal differential
thrust USV model under feasible thrust commands. This is intentional: MPC tests
should not start from arbitrary holonomic paths that already require lateral
force. For each reference the generator stores:

```text
z_ref(t), z_dot_ref(t), z_ddot_ref(t), nu_ref(t), tau_ref(t)
```

where:

```math
z =
\begin{bmatrix}
x & y & \psi
\end{bmatrix}^{T}
```

and:

```math
\nu =
\begin{bmatrix}
u & v & r
\end{bmatrix}^{T}
=
R_3(\psi)^T \dot{z}
```

The available reference names are:

```text
hold, straight, arc, stop, figure8, spiral, yaw
```

### NMPC State And Control

The physical prediction state is:

```math
\eta_k =
\begin{bmatrix}
x_k &
y_k &
\psi_k &
\dot{x}_k &
\dot{y}_k &
\dot{\psi}_k
\end{bmatrix}^{T}
```

The optimizer control input is the generalized force:

```math
\tau_k =
\begin{bmatrix}
\tau_{u,k} &
\tau_{v,k} &
\tau_{r,k}
\end{bmatrix}^{T}
```

The acados solver uses an augmented bookkeeping state:

```math
\chi_k =
\begin{bmatrix}
\eta_k^{T} &
s_v &
\tau_{u,k-1} &
\tau_{r,k-1}
\end{bmatrix}^{T}
```

`s_v` is a nonnegative horizon-wide lateral-force slack variable. The
previous-input terms are not physical states and are not current control inputs;
they are carried only so the discrete NLP can penalize command steps:

```math
\tau_{u,k}-\tau_{u,k-1},\qquad
\tau_{r,k}-\tau_{r,k-1}
```

The discrete augmented dynamics keep `s_v` constant and copy the current
optimized command into the previous-command memory states at the next shooting
node.

The receding horizon uses `N = 12` and this nonuniform time grid:

```text
[0.08, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.70, 0.97] s
```

The total prediction horizon is 4.0 seconds.

### Continuous Model

Define the body-frame velocities from local-frame velocity:

```math
\begin{aligned}
u &= \dot{x}\cos\psi + \dot{y}\sin\psi \\
v &= -\dot{x}\sin\psi + \dot{y}\cos\psi \\
r &= \dot{\psi}
\end{aligned}
```

The nominal damping terms are:

```math
\begin{aligned}
D_u &= d_u u + d_{uu}|u|u \\
D_v &= d_v v + d_{vv}|v|v \\
D_r &= d_r r + d_{rr}|r|r
\end{aligned}
```

The body-frame dynamics are:

```math
\begin{aligned}
\dot{u} &= \frac{\tau_u + mvr - D_u}{m} \\
\dot{v} &= \frac{\tau_v - mur - D_v}{m} \\
\dot{r} &= \frac{\tau_r - D_r}{I_z}
\end{aligned}
```

The local-frame acceleration used by the NMPC is:

```math
\begin{aligned}
\ddot{x} &=
\cos\psi\,\dot{u}
- \sin\psi\,\dot{v}
- r\dot{y} \\
\ddot{y} &=
\sin\psi\,\dot{u}
+ \cos\psi\,\dot{v}
+ r\dot{x} \\
\ddot{\psi} &= \dot{r}
\end{aligned}
```

The discrete prediction model is fourth-order Runge-Kutta:

```math
x_{k+1} = f_{\mathrm{RK4}}(x_k,u_k,\Delta t_k)
```

### Continuous-Time OCP Form

For paper notation, the corresponding continuous-time optimal control problem can
be written before discretization. Define the physical NMPC state and generalized
force input as:

```math
\eta(t) =
\begin{bmatrix}
x(t) &
y(t) &
\psi(t) &
\dot{x}(t) &
\dot{y}(t) &
\dot{\psi}(t)
\end{bmatrix}^{T}
```

```math
\tau(t) =
\begin{bmatrix}
\tau_u(t) &
\tau_v(t) &
\tau_r(t)
\end{bmatrix}^{T}
```

The continuous-time problem is:

```math
\begin{aligned}
\min_{\eta(\cdot),\,\tau(\cdot),\,s_v}
\quad
&\Phi(\eta(t_f),\eta_{\mathrm{ref}}(t_f))
+ \int_{t_0}^{t_f}
L(\eta(t),\tau(t),s_v,\eta_{\mathrm{ref}}(t))\,dt \\
\mathrm{s.t.}\quad
&\dot{\eta}(t)=f(\eta(t),\tau(t)) \\
&\eta(t_0)=\hat{\eta}(t_0) \\
&T_{\min}\le T_L(\tau(t))\le T_{\max} \\
&T_{\min}\le T_R(\tau(t))\le T_{\max} \\
&-\tau_{r,\max}\le \tau_r(t)\le \tau_{r,\max} \\
&-s_v\le \tau_v(t)\le s_v \\
&0\le s_v\le s_{v,\max}
\end{aligned}
```

with:

```math
\begin{aligned}
T_L(\tau) &=
\frac{1}{2}
\left(
\tau_u-\frac{\tau_r}{l}
\right) \\
T_R(\tau) &=
\frac{1}{2}
\left(
\tau_u+\frac{\tau_r}{l}
\right)
\end{aligned}
```

The continuous running cost matching the current implementation is:

```math
\begin{aligned}
L =\;&
q_p\|p-p_{\mathrm{ref}}\|^2
+q_{\psi}e_{\psi}^{2}
+q_v\|\dot{p}-\dot{p}_{\mathrm{ref}}\|^2
+q_r(\dot{\psi}-\dot{\psi}_{\mathrm{ref}})^2 \\
&+q_u\tau_u^2
+q_{\tau r}\tau_r^2
+q_s s_v^2
\end{aligned}
```

and the terminal cost is:

```math
\begin{aligned}
\Phi =\;&
q_{p,N}\|p(t_f)-p_{\mathrm{ref}}(t_f)\|^2
+ q_{\psi,N}e_{\psi}(t_f)^2 \\
&+ q_{v,N}\|\dot{p}(t_f)-\dot{p}_{\mathrm{ref}}(t_f)\|^2
+ q_{r,N}(\dot{\psi}(t_f)-\dot{\psi}_{\mathrm{ref}}(t_f))^2
\end{aligned}
```

The implemented controller solves a nonuniform RK4 transcription of this OCP.
The acados implementation augments the physical state with the constant `s_v`,
`tau_u,k-1`, and `tau_r,k-1` only for the lateral-force soft constraint and
command-step regularization. These augmented entries are bookkeeping variables,
not extra USV states.

### Cost Function

Let:

```math
e_{\psi,k} =
\mathrm{wrap}(\psi_k-\psi_{\mathrm{ref},k})
```

The stage cost is:

```math
\begin{aligned}
J_k = \Delta t_k (&
q_p\left((x_k-x_{\mathrm{ref},k})^2+(y_k-y_{\mathrm{ref},k})^2\right)
+ q_{\psi}e_{\psi,k}^2 \\
&+ q_v\left((\dot{x}_k-\dot{x}_{\mathrm{ref},k})^2
+(\dot{y}_k-\dot{y}_{\mathrm{ref},k})^2\right) \\
&+ q_r(\dot{\psi}_k-\dot{\psi}_{\mathrm{ref},k})^2
+ q_u\tau_{u,k}^2
+ q_{\tau r}\tau_{r,k}^2 \\
&+ q_{\Delta}\left((\tau_{u,k}-\tau_{u,k-1})^2
+(\tau_{r,k}-\tau_{r,k-1})^2\right)
+ q_s s_v^2 )
\end{aligned}
```

The terminal cost uses the same pose and velocity errors with terminal weights
and no control effort terms:

```math
J_N =
q_{p,N}\|p_N-p_{\mathrm{ref},N}\|^2
+ q_{\psi,N}e_{\psi,N}^2
+ q_{v,N}\|\dot{p}_N-\dot{p}_{\mathrm{ref},N}\|^2
+ q_{r,N}(\dot{\psi}_N-\dot{\psi}_{\mathrm{ref},N})^2
```

The total problem is:

```math
\min_{\chi_0,\ldots,\chi_N,\tau_0,\ldots,\tau_{N-1}}
J_N + \sum_{k=0}^{N-1} J_k
```

subject to the RK4 augmented dynamics, physical initial state from the EKF, the
previous-command memory initialized from the last applied command, actuator
limits, and the tightened lateral-force slack constraints below.

### Constraints And Allocation

The differential-thrust allocation inside the NMPC is:

```math
\begin{aligned}
T_L &=
\frac{1}{2}
\left(
\tau_u-\frac{\tau_r}{l}
\right) \\
T_R &=
\frac{1}{2}
\left(
\tau_u+\frac{\tau_r}{l}
\right)
\end{aligned}
```

The current hard actuator constraints are:

```math
\begin{aligned}
-100 &\le T_L \le 100 \\
-100 &\le T_R \le 100 \\
-100 &\le \tau_r \le 100
\end{aligned}
```

The underactuated lateral-force condition is enforced as a tight soft
constraint:

```math
\begin{aligned}
-s_v &\le \tau_{v,k} \le s_v \\
0 &\le s_v \le 0.05
\end{aligned}
```

The controller outputs only the first optimized `tau_u` and `tau_r` through the
allocation above. `tau_v` is never sent to an actuator; it is kept in the NLP to
discourage infeasible lateral correction trajectories.

### Current Parameters

| Parameter | Value |
| --- | ---: |
| `solver_backend` | `acados` |
| `control_rate_hz` | 20 |
| `horizon_steps` | 12 |
| total horizon | 4.0 s |
| `q_position` | 4.0 |
| `q_yaw` | 3.0 |
| `q_velocity` | 2.0 |
| `q_yaw_rate` | 1.5 |
| `q_terminal_position` | 700.0 |
| `q_terminal_yaw` | 320.0 |
| `q_terminal_velocity` | 200.0 |
| `q_terminal_yaw_rate` | 100.0 |
| `q_tau_u` | 0.001 |
| `q_tau_r` | 0.0004 |
| `q_delta_tau` | 0.004 |
| `q_tau_v_slack` | 20.0 |
| `tau_v_slack_max` | 0.05 N |
| `min_thrust` | -100 N |
| `max_thrust` | 100 N |
| `max_yaw_moment` | 100 N m |

### Debug And Plots

The controller debug topic `/safe_docking/flatness_mpc/debug` includes the
active reference, predicted state/control values, `tau_v` slack, thrust
commands, solver status, solve time, terminal prediction errors, and allocation
residuals.

The command topic `/safe_docking/flatness_mpc/command` publishes:

```text
left_cmd, right_cmd, tau_u_cmd, tau_v_cmd, tau_r_cmd
```

Recorder output includes:

```text
01_trajectory_xy.png
02_states.png
03_velocities.png
04_forces.png
05_errors_solver.png
flatness_mpc_performance.csv
metrics.json
```

### Running The Controller

Launch the controller after the safe docking simulation and estimator are
running:

```bash
ros2 launch robotx_safe_docking_control flatness_mpc_controller.launch.py \
  reference_name:=figure8 \
  use_sim_time:=True
```

It subscribes to EKF odometry on `/safe_docking/odometry` and publishes direct
thrust commands on:

```text
/wamv/thrusters/left/thrust
/wamv/thrusters/right/thrust
```

### Verification

Static checks used for this controller:

```bash
PYTHONPATH=/home/zjy/vrx_docking/robotx_safe_docking_control \
  /usr/bin/python3 -m py_compile \
  robotx_safe_docking_control/robotx_safe_docking_control/flatness_mpc_controller.py \
  robotx_safe_docking_control/robotx_safe_docking_control/feasible_references.py \
  robotx_safe_docking_control/robotx_safe_docking_control/usv_flatness.py \
  scripts/record_flatness_mpc_performance.py

colcon build --packages-select robotx_safe_docking_control \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

The pure model tests pass:

```bash
PYTHONPATH=/home/zjy/vrx_docking/robotx_safe_docking_control \
  /usr/bin/python3 -m pytest \
  robotx_safe_docking_control/test/test_usv_flatness.py
```

Headless Gazebo / VRX verification was run with the no-render WAM-V config and
recorder-only ground truth odometry. Before each scenario the test harness
explicitly stops ROS launch processes, Gazebo, bridges, the EKF, the controller,
the recorder, and the ROS 2 daemon. This avoids stale EKF / simulator
publishers contaminating the plots.

The clean rerun artifacts are saved under
`output/nmpc_rk4_tight005_all_refs_clean_rerun_20260604/`.

| Reference | Truth final dist (m) | Truth final yaw (rad) | Truth final speed (m/s) | RMS ref dist (m) | Max abs tau_v (N) | Saturation frac | Solver success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hold | 0.0428 | 0.00345 | 0.0052 | 0.0321 | 0.0298 | 0.000 | 1.000 |
| straight | 0.0122 | 0.00016 | 0.0110 | 0.1824 | 0.0154 | 0.000 | 1.000 |
| arc | 0.0439 | 0.00190 | 0.0165 | 0.1230 | 0.0311 | 0.000 | 1.000 |
| stop | 0.0202 | 0.00230 | 0.0118 | 0.1220 | 0.0294 | 0.001 | 1.000 |
| spiral | 0.0448 | 0.00077 | 0.0047 | 0.2161 | 0.0358 | 0.001 | 1.000 |
| yaw | 0.0727 | 0.01022 | 0.0222 | 0.1008 | 0.0353 | 0.006 | 1.000 |
| figure8 | 0.0239 | 0.00045 | 0.0091 | 0.2078 | 0.0442 | 0.003 | 1.000 |

The clean nominal synthetic-reference runs are successful. The tight
lateral-force condition works numerically: all recorded runs have zero `tau_v`
constraint violation and `|tau_v| <= 0.05 N`. Earlier runs that showed large
terminal errors were traced to stale processes / topic state between tests, not
the NMPC formulation.

### Work Since The Previous Commit

- Added common USV flatness / model utilities and unit tests.
- Added feasible synthetic reference generation for `hold`, `straight`, `arc`,
  `stop`, `figure8`, `spiral`, and `yaw`.
- Added the acados RK4 dynamics NMPC controller and launch/config files.
- Added recorder tooling for EKF-vs-ground-truth MPC plots and metrics.
- Added a no-render WAM-V config with recorder-only ground truth odometry.
- Tightened the lateral-force slack cap to `0.05 N` and verified it in Gazebo.

### Next Controller Improvements

The most urgent issue is not `tau_v` feasibility; that part is working. The
weakness is now validation hygiene and robustness beyond nominal clean starts.
The next iterations should:

1. Revisit reference speed profiles against the `+-100 N` actuator limits.
2. Keep the strict cleanup / fresh EKF-origin workflow for every closed-loop
   test.
3. Improve terminal logic with a docking/hold phase that remains inside the
   same NMPC formulation instead of relying on a separate PID fallback.
4. Investigate solver status `2` from acados and tune scaling / globalization
   so successful closed-loop behavior also has cleaner solver convergence.
5. Test with wind/current, initial pose offsets, and estimator noise before
   calling the controller robust.
6. Later, replace the synthetic references with a MINCO or obstacle-aware
   spline planner that directly optimizes curvature, clearance, and actuator
   feasibility.

## Reference

If you use the VRX simulation in your work, please cite our summary publication, [Toward Maritime Robotic Simulation in Gazebo](https://wiki.nps.edu/display/BB/Publications?preview=/1173263776/1173263778/PID6131719.pdf): 

```
@InProceedings{bingham19toward,
  Title                    = {Toward Maritime Robotic Simulation in Gazebo},
  Author                   = {Brian Bingham and Carlos Aguero and Michael McCarrin and Joseph Klamo and Joshua Malia and Kevin Allen and Tyler Lum and Marshall Rawson and Rumman Waqar},
  Booktitle                = {Proceedings of MTS/IEEE OCEANS Conference},
  Year                     = {2019},
  Address                  = {Seattle, WA},
  Month                    = {October}
}
```

## Contributing
This project is under active development to support the VRX and RobotX teams. We are adding and improving things all the time. Our primary focus is to provide the fundamental aspects of the robot and environment, but we rely on the community to develop additional functionality around their particular use cases.

If you have any questions about these topics, or would like to work on other aspects, please contribute.  You can contact us directly (see below), submit an [issue](https://github.com/osrf/vrx/issues) or, better yet, submit a [pull request](https://github.com/osrf/vrx/pulls/)!

## Contributors

We continue to receive important improvements from the community.  We have done our best to document this on our [Contributors Wiki](https://github.com/osrf/vrx/wiki/Contributors).

## Contacts

 * Carlos Agüero <caguero@openrobotics.org>
 * Michael McCarrin <mrmccarr@nps.edu>
 * Brian Bingham <bbingham@nps.edu>
