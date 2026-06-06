# Codex Handoff Context: `vrx_docking` MINCO Work

This note summarizes the work Codex did in the real target repository:

```text
/home/zjy/vrx_docking
```

The earlier UAV-state answer came from the wrong repository:

```text
/home/zjy/isaac-catch/Dynamic-Catching
```

For `vrx_docking`, the current algorithm is USV / RobotX safe docking focused, not UAV focused.

## Repository And Framework

The `vrx_docking` repo is a ROS 2 Humble safe docking stack for a RobotX-style USV in VRX/Gazebo simulation.

Main framework pieces:

- ROS 2 Python packages and launch files
- Gazebo / VRX simulation
- Odometry-based state extraction
- Flatness-based reference handling
- acados-based nonlinear MPC controller
- USV dynamics / allocation model
- Optional MINCO-based feasible reference generation

The main control package touched by this work is:

```text
/home/zjy/vrx_docking/robotx_safe_docking_control
```

## Main Work Completed

Implemented a MINCO-based feasible reference generation path for the USV docking controller.

The new MINCO path generates smooth fifth-order trajectories in the flat output space:

```text
z = [x, y, psi]
```

The optimizer only tunes sparse outer variables:

```text
intermediate points
segment times
```

Polynomial coefficients are regenerated from boundary constraints and continuity constraints. They are not directly optimized.

The implementation follows a strict fifth-order / S3NU MINCO form with:

- position / velocity / acceleration boundary constraints
- continuity of velocity, acceleration, jerk, and snap at segment boundaries
- fixed-time and variable-time support
- energy objective
- adjoint gradient propagation

## New Files Added

```text
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_s3nu.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_usv_penalties.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_usv_optimizer.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_feasible_reference.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_offline_validation.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_reference_generation.py
```

New tests were also added:

```text
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_s3nu.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_usv_penalties.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_usv_optimizer.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_feasible_reference.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_offline_validation.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_reference_generation.py
```

## Existing Files Modified

```text
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/flatness_mpc_controller.py
/home/zjy/vrx_docking/robotx_safe_docking_control/config/flatness_mpc_controller.yaml
/home/zjy/vrx_docking/robotx_safe_docking_control/launch/flatness_mpc_controller.launch.py
/home/zjy/vrx_docking/docs/minco_implementation_status.md
```

## Controller State Used In `vrx_docking`

The current controller uses USV flat state:

```text
z = [x, y, psi]
z_dot = [x_dot, y_dot, psi_dot]
```

The controller reads odometry pose and twist, then computes body-frame velocity:

```text
nu = [u, v, r]
```

The acados / NMPC state is:

```text
[x, y, psi, x_dot, y_dot, psi_dot, tau_v_slack, tau_u_prev, tau_r_prev]
```

The NMPC control is:

```text
[tau_u, tau_v, tau_r]
```

Only `tau_u` and `tau_r` are mapped to left and right thrust commands. `tau_v` is constrained/penalized as an underactuation-consistency term.

## MINCO USV Penalties

The new USV-specific penalty module includes dense trajectory penalties for:

- bounded lateral generalized force `tau_v`
- body-frame velocity limits
- body-frame acceleration limits
- actuator thrust bounds

The left/right thrust mapping used in the penalties is:

```text
T_L = 0.5 * (tau_u - tau_r / l)
T_R = 0.5 * (tau_u + tau_r / l)
```

No obstacle, corridor, guide-path, neural-network, PID, or effort-shaping terms were added.

## MINCO Reference Integration

The flatness MPC controller now supports:

```text
reference_source:=synthetic
reference_source:=minco
```

Synthetic reference generation remains available. MINCO can be selected from launch/config for local feasible reference generation.

Example launch form used during validation:

```bash
ros2 launch robotx_safe_docking_control flatness_mpc_controller.launch.py reference_source:=minco reference_name:=local_offset
```

The launch file now exposes `reference_source`, and the YAML config contains MINCO-related parameters.

## Boundary Acceleration Policy

MINCO needs boundary acceleration, but current odometry/estimation does not directly provide trusted acceleration.

A fallback boundary acceleration policy was added in:

```text
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_reference_generation.py
```

Priority order:

1. previous accepted MINCO reference acceleration
2. previous NMPC predicted acceleration
3. nominal dynamics from last applied command
4. zero acceleration

The controller does not read acceleration directly from odometry.

Diagnostic source IDs:

```text
previous_minco = 1
previous_mpc = 2
last_command = 3
zero = 4
```

## Live Default Parameters After Repair

The online MINCO defaults were made conservative enough for the 25-second smoke test:

```text
minco_terminal_forward = 2.0
minco_terminal_lateral = 0.4
minco_terminal_yaw = 0.25
minco_piece_count = 2
minco_fixed_total_time = 8.0
minco_lambda_actuator = 10.0
minco_max_iterations = 20
```

The multi-piece/full reference setup is still configurable.

## Tests And Build

The phase gate test command passed:

```bash
PYTHONPATH=/home/zjy/vrx_docking/robotx_safe_docking_control \
python3 -m pytest -q \
  robotx_safe_docking_control/test/test_usv_flatness.py \
  robotx_safe_docking_control/test/test_minco_s3nu.py \
  robotx_safe_docking_control/test/test_minco_usv_penalties.py \
  robotx_safe_docking_control/test/test_minco_usv_optimizer.py \
  robotx_safe_docking_control/test/test_minco_feasible_reference.py \
  robotx_safe_docking_control/test/test_minco_offline_validation.py \
  robotx_safe_docking_control/test/test_minco_reference_generation.py
```

Result:

```text
27 passed
```

The package build passed:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select robotx_safe_docking_control --symlink-install
```

Only a pre-existing setuptools warning about `tests_require` appeared.

## Offline Validation

Offline validation was regenerated:

```bash
PYTHONPATH=/home/zjy/vrx_docking/robotx_safe_docking_control \
python3 -m robotx_safe_docking_control.minco_offline_validation
```

Metrics file:

```text
/home/zjy/vrx_docking/output/minco_offline_validation/metrics.json
```

Important metrics:

Straight scenario:

```text
optimizer_success = 1.0
objective_before = 11.069859205193236
objective_after = 1.5897600000000012
pva_residual = 6.661338147750939e-16
continuity_residual = 5.551115123125783e-16
gradient_check_error = 6.129713875322317e-10
max_abs_tau_v = 2.817438963630209e-14
rms_tau_v = 1.457714193887478e-14
velocity_violation = 0.0
acceleration_violation = 0.0
actuator_bound_violation = 0.0
feasible_reference_exported = 1.0
```

Gentle arc scenario:

```text
optimizer_success = 1.0
objective_before = 4.942075384207857
objective_after = 0.7072685187553539
pva_residual = 5.907440274120147e-16
continuity_residual = 3.510833468576701e-16
gradient_check_error = 3.226349243926798e-10
max_abs_tau_v = 43.50963633036301
rms_tau_v = 24.267875732047347
velocity_violation = 0.0
acceleration_violation = 0.0
actuator_bound_violation = 0.0
feasible_reference_exported = 1.0
```

## Closed-Loop Smoke Validation

A 25-second headless VRX/Gazebo smoke test was run with MINCO enabled.

The run used an isolated ROS domain:

```text
ROS_DOMAIN_ID = 88
```

Main launch pieces:

```bash
ros2 launch vrx_gz safe_docking.launch.py headless:=True launch_rviz:=False
ros2 launch robotx_safe_docking_estimation estimator.launch.py verify:=False
ros2 launch robotx_safe_docking_control flatness_mpc_controller.launch.py reference_source:=minco reference_name:=local_offset
python3 scripts/record_flatness_mpc_performance.py \
  --output-dir /home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606 \
  --duration 25
```

Output artifacts:

```text
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/metrics.json
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/flatness_mpc_performance.csv
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/01_trajectory_xy.png
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/02_states.png
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/03_velocities.png
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/04_forces.png
/home/zjy/vrx_docking/output/minco_phase8_smoke_repair2_20260606/05_errors_solver.png
```

Important closed-loop metrics:

```text
solver_success_rate = 1.0
solve_time_mean_ms = 22.14966060942406
solve_time_max_ms = 32.40162692964077
estimate_rms_ref_distance_m = 0.0426964709616615
estimate_rms_ref_yaw_error_rad = 0.003107906029522406
estimate_final_terminal_distance_m = 0.04921113359065198
estimate_final_terminal_yaw_error_rad = 0.0007488149066783302
estimate_final_speed_mps = 0.013391224191147324
max_abs_tau_v_n = 0.02952576032130968
rms_tau_v_n = 0.01763068226133377
max_pred_tau_v_violation_n = 0.0
max_abs_thrust_cmd_n = 8.747150916849924
thrust_saturation_fraction = 0.0
```

The controller reported MINCO startup diagnostics similar to:

```text
duration = 8.00 s
boundary_acceleration = zero
feasible_reference_exported = 1.0
minco_piece_count = 2.0
minco_optimizer_success = 1.0
minco_optimizer_iterations = 5.0
minco_objective_before = 0.099885913280714
minco_objective_after = 0.049948578756232966
max_abs_tau_v = 5.919248304388293
rms_tau_v = 3.45125342717628
velocity_violation = 0.0
acceleration_violation = 0.0
actuator_bound_violation = 0.0
boundary_source_id = 4.0
```

## Repairs Made During Live Validation

Initial live validation exposed an L-BFGS line-search failure:

```text
MINCO optimizer failed: ABNORMAL_TERMINATION_IN_LNSRCH
```

Likely cause:

```text
Live EKF state plus tight actuator constraints let L-BFGS try extreme intermediate/time variables,
which overflowed dense penalties and caused line-search failure.
```

Repairs made:

- clipped extreme smooth-positive penalty inputs
- added conservative L-BFGS-B bounds on sparse outer variables
- improved fixed-time allocation with minimum segment time reserve
- increased default fixed total time to 8 seconds
- reduced online actuator penalty aggressiveness
- reduced online default target offset and piece count
- reduced online max iterations to keep startup time reasonable

After repair, live-state repro became fast and stable:

```text
wall time approximately 1.48 s
optimizer iterations = 4
zero violations
```

The final 25-second smoke test then passed.

## Caveats

The repo worktree was dirty before and after this work. It includes both MINCO-related changes and unrelated pre-existing changes. Do not blindly revert the repo.

The 25-second VRX/Gazebo run was a smoke test, not a long endurance validation.

During shutdown after artifact collection, Gazebo printed a segfault in cleanup code. This happened after the validation window and after metrics/artifacts had been collected. It should be tracked separately if long-running simulation stability matters.

## Git Status Notes

Branch at the time of work:

```text
safe-docking-dev...origin/safe-docking-dev
```

Tracked modified files included both MINCO-related files and unrelated estimation/documentation files. Be careful when staging or committing.

MINCO-related files to consider if making a focused commit:

```text
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_s3nu.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_usv_penalties.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_usv_optimizer.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_feasible_reference.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_offline_validation.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/minco_reference_generation.py
/home/zjy/vrx_docking/robotx_safe_docking_control/robotx_safe_docking_control/flatness_mpc_controller.py
/home/zjy/vrx_docking/robotx_safe_docking_control/config/flatness_mpc_controller.yaml
/home/zjy/vrx_docking/robotx_safe_docking_control/launch/flatness_mpc_controller.launch.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_s3nu.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_usv_penalties.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_usv_optimizer.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_feasible_reference.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_offline_validation.py
/home/zjy/vrx_docking/robotx_safe_docking_control/test/test_minco_reference_generation.py
/home/zjy/vrx_docking/docs/minco_implementation_status.md
/home/zjy/vrx_docking/docs/codex_context_minco_vrx_docking_handoff.md
```
