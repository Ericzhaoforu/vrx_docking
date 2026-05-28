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
  estimation is achieved for the current simulation baseline; perception,
  planning, and control are still intentionally not implemented.

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

```text
x_state =
[
  x
  y
  psi
  x_dot
  y_dot
  psi_dot
]^T
```

The control input is the left and right thruster force command. In the current
Gazebo setup these commands are published directly as thrust forces in Newtons:

```text
T =
[
  T_L
  T_R
]^T
```

### Kinematics

The first three equations are directly from the state definition:

```text
x_state_dot =
[
  x_dot
  y_dot
  psi_dot
  x_ddot
  y_ddot
  psi_ddot
]
```

Define the body-to-local yaw rotation:

```text
R(psi) =
[
  cos(psi)  -sin(psi)
  sin(psi)   cos(psi)
]
```

The local linear velocity can be converted into body-frame surge and sway
velocity for hydrodynamic damping:

```text
[
  u
  v
]
=
R(psi)^T
[
  x_dot
  y_dot
]

u = x_dot * cos(psi) + y_dot * sin(psi)
v = -x_dot * sin(psi) + y_dot * cos(psi)
```

### Thruster Allocation

Assume two fixed aft thrusters, no lateral thruster, and symmetric spacing.
Let `b` be the lateral distance between the two thrusters and `l = b / 2`.

```text
[
  F_x
  F_y
  tau_z
]
=
B(psi)
[
  T_L
  T_R
]

B(psi) =
[
  cos(psi)   cos(psi)
  sin(psi)   sin(psi)
     -l          l
]
```

Therefore:

```text
F_x = (T_L + T_R) * cos(psi)
F_y = (T_L + T_R) * sin(psi)
tau_z = l * (T_R - T_L)
```

`F_x` and `F_y` are local-frame force components. `tau_z` is the yaw torque
about the local up axis. Because `B(psi)` is `3 x 2`, the WAM-V with two fixed
aft thrusters is underactuated: it cannot command arbitrary `F_x`, `F_y`, and
`tau_z` independently.

### Damping

Hydrodynamic damping is most naturally modeled in the vessel body frame because
water resistance depends on motion relative to the hull:

```text
F_D_body =
[
  -(d_u + d_uu * |u|) * u
  -(d_v + d_vv * |v|) * v
]
```

Rotate it back into the local frame:

```text
F_D_local = R(psi) * F_D_body
```

Expanded:

```text
F_D_x =
  -cos(psi) * (d_u + d_uu * |u|) * u
  +sin(psi) * (d_v + d_vv * |v|) * v

F_D_y =
  -sin(psi) * (d_u + d_uu * |u|) * u
  -cos(psi) * (d_v + d_vv * |v|) * v
```

Yaw damping is:

```text
tau_D_z = -(d_r + d_rr * |psi_dot|) * psi_dot
```

### Rotation Simplification

The body angular velocity relative to the local frame can be written as:

```text
omega_BW = p * x_b + q * y_b + r * z_b
```

For planar docking:

```text
p = 0
q = 0
r = psi_dot

omega_BW = psi_dot * z_b
```

The full rigid-body rotational equation is:

```text
tau = I * omega_dot + omega x (I * omega)
```

For yaw-only motion and a body inertia tensor aligned with the principal axes,
`omega_BW` and `I * omega_BW` are parallel, so:

```text
omega_BW x (I * omega_BW) = 0
```

Therefore the yaw equation reduces to:

```text
I_z * psi_ddot = tau_z + tau_D_z
```

### Local-Frame Dynamics

The complete dynamics are:

```text
d(x) / dt = x_dot

d(y) / dt = y_dot

d(psi) / dt = psi_dot

x_ddot =
[
  (T_L + T_R) * cos(psi)
  -cos(psi) * (d_u + d_uu * |u|) * u
  +sin(psi) * (d_v + d_vv * |v|) * v
] / m

y_ddot =
[
  (T_L + T_R) * sin(psi)
  -sin(psi) * (d_u + d_uu * |u|) * u
  -cos(psi) * (d_v + d_vv * |v|) * v
] / m

psi_ddot =
[
  l * (T_R - T_L)
  -(d_r + d_rr * |psi_dot|) * psi_dot
] / I_z
```

with:

```text
u = x_dot * cos(psi) + y_dot * sin(psi)
v = -x_dot * sin(psi) + y_dot * cos(psi)
```

The corresponding matrix form is:

```text
d/dt
[
  x
  y
  psi
  x_dot
  y_dot
  psi_dot
]
=
[
  x_dot
  y_dot
  psi_dot
  (F_x + F_D_x) / m
  (F_y + F_D_y) / m
  (tau_z + tau_D_z) / I_z
]
```

For the current VRX WAM-V simulation, reasonable baseline parameters from the
model files are:

```text
m = 180.0
I_z = 446.0

d_u = 100.0
d_uu = 150.0
d_v = 100.0
d_vv = 100.0
d_r = 800.0
d_rr = 800.0
```

These values are suitable for first controller and estimator development, but
payloads and fitted simulator behavior may shift the effective mass, inertia,
and damping. Re-identify the parameters from simulation data before relying on
high-accuracy prediction.

## GPS / IMU EKF State Estimator

The initial state estimator lives in `robotx_safe_docking_estimation`. It uses a
planar inertial EKF with GPS position updates and IMU yaw as a compass / AHRS
placeholder. The internal EKF state is:

```text
x_ekf =
[
  x
  y
  psi
  x_dot
  y_dot
  b_ax
  b_ay
  b_gz
]^T
```

where `b_ax` and `b_ay` are body-frame accelerometer biases and `b_gz` is the
gyro-z bias. The published controller-facing state is:

```text
x_pub =
[
  x
  y
  psi
  x_dot
  y_dot
  psi_dot
]^T
```

with `psi_dot = omega_z,m - b_gz`.

The first valid GPS fix defines the local ENU origin. For the small VRX task
area, GPS is converted to local meters with an equirectangular WGS84
approximation:

```text
x = (lon - lon_0) * cos(lat_0) * R_e
y = (lat - lat_0) * R_e
```

where `R_e = 6378137.0 m`.

The estimator publishes the WAM-V reference point, not the GPS antenna point.
The safe docking WAM-V GPS is mounted at `[-0.85, 0.0]` m in the body frame, so
GPS position measurements are corrected for the yaw-dependent lever arm:

```text
r_gps_B = [gps_body_x, gps_body_y]^T
z_base = z_gps_delta - R(psi) * r_gps_B + R(psi_0) * r_gps_B
```

where `psi_0` is the first IMU yaw used by the estimator.

The EKF prediction uses body-frame IMU linear acceleration and yaw rate:

```text
a_B =
[
  a_x,m - b_ax
  a_y,m - b_ay
]

R(psi)
=
[
  cos(psi)  -sin(psi)
  sin(psi)   cos(psi)
]

a_W = R(psi) * a_B
omega = omega_z,m - b_gz

x_k+1 = x_k + x_dot_k * dt + 0.5 * a_W,x * dt^2
y_k+1 = y_k + y_dot_k * dt + 0.5 * a_W,y * dt^2
psi_k+1 = wrap(psi_k + omega * dt)
x_dot_k+1 = x_dot_k + a_W,x * dt
y_dot_k+1 = y_dot_k + a_W,y * dt
b_ax,k+1 = b_ax,k
b_ay,k+1 = b_ay,k
b_gz,k+1 = b_gz,k
```

GPS measurement update:

```text
z_gps =
[
  x_gps
  y_gps
]

h_gps(x_state) =
[
  x
  y
]
```

IMU yaw measurement update. In simulation this comes from
`sensor_msgs/msg/Imu.orientation`; in the real system it is intended to stand in
for a future compass, AHRS, INS, or dual-RTK heading source:

```text
z_imu_yaw = psi_imu
h_imu_yaw(x_ekf) = psi
```

The gyro-z value is not double-counted as a separate measurement in the current
filter. It is used as the prediction input, and the published yaw rate is
`psi_dot = omega_z,m - b_gz`.

Current EKF parameters are:

```text
use_imu_orientation: true
use_gps_velocity_measurement: true

gps_body_x: -0.85
gps_body_y: 0.0

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
