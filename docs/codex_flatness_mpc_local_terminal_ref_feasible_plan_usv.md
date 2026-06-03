# Codex Plan: Paper-Faithful Flatness-MPC Tracking with Feasible Synthetic References and Local Terminal Cost

## Purpose

This plan defines the implementation sequence for the safe-docking-dev branch. The controller must follow the flatness-based MPC design inspired by:

- *Optimal Trajectory Planning and Model Predictive Control of Underactuated Marine Surface Vessels using a Flatness-Based Approach*.

The short-term goal is to implement and validate the tracking NMPC in Gazebo/VRX before implementing MINCO and front-end planning.

This plan supersedes previous instructions that suggested a thrust-space dynamics NMPC as the main controller. The main controller is a flatness-MPC trajectory tracking controller.

---

## Non-negotiable design requirements

1. The tracking controller must be a flatness-based MPC.
2. The MPC predicted variables must parameterize the flat output trajectory.
3. The flat output is

   z = [x, y, psi]^T.

4. The MPC must compute the nominal generalized force

   tau = theta_tau(z, z_dot, z_ddot)

   at prediction nodes.

5. The MPC must include a soft lateral generalized-force condition on tau_v.
6. The MPC output is tau_u and tau_r from the first predicted control node.
7. tau_u and tau_r are mapped to left/right thrust commands after the MPC solve.
8. Do not implement PID fallback for the new pipeline.
9. Do not implement NN residual compensation yet.
10. Do not use ground-truth dock pose, evaluator state, target bay index, Gazebo entity pose, or simulator oracle topics in autonomy code.

---

## Important conceptual correction

The upper-layer trajectory optimizer, eventually MINCO, should generate a reference trajectory that already satisfies, or approximately satisfies, the underactuated condition tau_v = 0.

However, the tracking MPC must still care about tau_v. The reason is that the real measured state may deviate from the reference because of disturbances, estimator errors, or modeling mismatch. The MPC then predicts a local correction trajectory. Without a tau_v penalty or soft constraint, that correction trajectory may require lateral actuator force, even if the reference trajectory itself was feasible.

Therefore the tracking MPC should include a soft lateral-force condition:

- It should penalize tau_v.
- It may use a slack variable to prevent infeasibility.
- It should not hard-enforce tau_v = 0 in the first implementation.

---

## Local terminal cost correction

The terminal cost should be local, not global.

At each tracking solve, the terminal target is the terminal state of the currently active local reference trajectory. Denote it as:

- z_term = local reference terminal pose.
- z_dot_term = local reference terminal flat velocity.
- nu_term = R_3(psi_term)^T z_dot_term.

The terminal cost is always active:

J_terminal = e_N^T Q_N e_N.

where

- e_N contains position error, yaw error, and body-velocity error relative to the local terminal target.

When the local trajectory terminal is the docking pose, the same local terminal cost naturally becomes the docking terminal cost. No separate global terminal cost is required in the first implementation.

Use stage-dependent terminal weights only if necessary:

- moderate terminal weights during normal local tracking;
- stronger yaw, position, and velocity-stop weights when the local terminal target is the docking pose or an approach pose.

Do not add a competing global-goal terminal penalty while tracking an intermediate local trajectory. That can pull the MPC away from the safe local reference.

---

## Phase 0: Common USV flatness/model utilities

### Goal

Create reusable math/model utilities used by the flatness-MPC, synthetic reference generator, and future MINCO optimizer.

### Required functions

Implement utilities for:

1. angle wrapping;
2. R_3(psi);
3. flat-to-body velocity mapping;
4. body-to-flat velocity mapping;
5. theta_tau_nominal(z, z_dot, z_ddot, params);
6. allocation from tau_u, tau_r to T_L, T_R;
7. allocation feasibility checks;
8. basic nominal damping model.

### Mathematical definitions

Flat output:

z = [x, y, psi]^T.

Flat velocity:

z_dot = [x_dot, y_dot, psi_dot]^T.

Body velocity:

nu = [u, v, r]^T = R_3(psi)^T z_dot.

Nominal acceleration:

nu_dot = R_3(psi)^T z_ddot + d/dt(R_3(psi)^T) z_dot.

Nominal generalized force:

tau_nom = M nu_dot + C_nom(nu) nu + D_nom(nu) nu.

Allocation:

T_L = 0.5 * (tau_u - tau_r / l).

T_R = 0.5 * (tau_u + tau_r / l).

### Unit tests

Add pure unit tests for:

- flat-to-body and body-to-flat inverse consistency;
- r = psi_dot;
- allocation sign convention;
- T_L = T_R gives nonzero surge and zero yaw;
- T_R > T_L gives the expected yaw sign;
- theta_tau produces finite outputs for representative z, z_dot, z_ddot.

### Phase gate

Proceed only if unit tests pass and allocation signs are verified.

If tests fail:

1. inspect frame convention;
2. inspect yaw sign;
3. inspect lever-arm sign;
4. stop after three focused repair attempts and report.

---

## Phase 1: Feasible synthetic reference generation for NMPC testing

### Goal

Before MINCO exists, generate synthetic reference trajectories that are nominally feasible for the underactuated vessel. Do not use arbitrary holonomic trajectories.

### Rationale

If the reference trajectory itself violates the underactuated condition, then an NMPC tracking failure is ambiguous. It may be caused by an infeasible reference rather than a bad controller.

### Preferred approach

Generate synthetic references by rolling out a nominal vessel model under feasible actuator commands. This guarantees actuator lateral force is zero by construction.

Use commanded generalized force or thrust sequences such as:

1. straight surge with T_L = T_R;
2. low-speed large-radius turn with T_L != T_R but both within bounds;
3. stop-at-goal with smooth thrust reduction;
4. differential-thrust yaw maneuver only if negative thrust is allowed.

Do not generate the first NMPC tests by prescribing arbitrary x(t), y(t), psi(t) curves that may require tau_v != 0.

### Reference output

For each synthetic rollout, publish or store:

- z_ref(t);
- z_dot_ref(t);
- z_ddot_ref(t) if available;
- nu_ref(t) = R_3(psi_ref)^T z_dot_ref(t);
- tau_ref(t) for diagnostics.

### Diagnostics

For each synthetic reference, compute:

- max |tau_v_ref|;
- RMS tau_v_ref;
- max |tau_u_ref|;
- max |tau_r_ref|;
- max |T_L_ref| and max |T_R_ref|;
- max speed;
- max yaw rate.

### Required reference scenarios

1. Hold current pose.
2. Straight surge along current heading.
3. Low-speed large-radius arc turn.
4. Stop at goal.
5. Differential-thrust yaw maneuver only if T_min < 0.

### Phase gate

Proceed only if at least straight surge and low-speed arc references have:

- finite states;
- bounded thrust;
- small numerical tau_v_ref;
- no yaw wrapping discontinuity.

If differential-thrust yaw is not physically allowed by thrust bounds, skip it and report why.

---

## Phase 2: Flatness-MPC trajectory tracking in Gazebo

### Goal

Implement the paper-faithful flatness-MPC tracking controller and test it in Gazebo/VRX against feasible synthetic references from Phase 1.

### MPC prediction variables

The MPC should parameterize predicted flat trajectories over the horizon:

- z_k;
- z_dot_k;
- z_ddot_k.

A simple first implementation may use a discrete double-integrator chain:

z_{k+1} = z_k + dt z_dot_k + 0.5 dt^2 z_ddot_k.

z_dot_{k+1} = z_dot_k + dt z_ddot_k.

The initial condition must be set from EKF feedback:

z_0 = [x_meas, y_meas, psi_meas]^T.

z_dot_0 = [x_dot_meas, y_dot_meas, psi_dot_meas]^T.

### Compute tau at each node

For every prediction node:

nu_k = R_3(psi_k)^T z_dot_k.

nu_dot_k = R_3(psi_k)^T z_ddot_k + d/dt(R_3(psi_k)^T) z_dot_k.

tau_k = M nu_dot_k + C_nom(nu_k) nu_k + D_nom(nu_k) nu_k.

where

tau_k = [tau_u,k, tau_v,k, tau_r,k]^T.

### Cost function

Use:

J = J_track + J_input + J_delta_input + J_sway + J_terminal.

Tracking cost:

- penalize z_k - z_ref,k;
- penalize nu_k - nu_ref,k.

Input cost:

- penalize tau_u,k and tau_r,k.

Input-rate cost:

- penalize tau_u,k+1 - tau_u,k;
- penalize tau_r,k+1 - tau_r,k.

Sway-force soft condition:

J_sway = q_v sum_k tau_v,k^2 + q_sigma sum_k sigma_v,k^2.

Optional soft inequality:

-epsilon_v - sigma_v,k <= tau_v,k <= epsilon_v + sigma_v,k.

sigma_v,k >= 0.

Terminal cost:

Use the local terminal target from the active reference trajectory:

z_term = z_ref,N.

nu_term = nu_ref,N.

J_terminal = e_N^T Q_N e_N.

where e_N includes:

- x_N - x_term;
- y_N - y_term;
- wrap(psi_N - psi_term);
- nu_N - nu_term.

No separate global terminal cost is required in this phase.

### Allocation after solve

After solving the MPC, take tau_u,0 and tau_r,0 and allocate:

T_L = 0.5 * (tau_u - tau_r / l).

T_R = 0.5 * (tau_u + tau_r / l).

Publish T_L and T_R to the existing thrust topics.

### Allocation constraints inside the MPC

Add constraints or penalties ensuring the allocated thrusts remain feasible:

T_L,min <= 0.5 * (tau_u,k - tau_r,k / l) <= T_L,max.

T_R,min <= 0.5 * (tau_u,k + tau_r,k / l) <= T_R,max.

Also include rate constraints if available.

### Gazebo tests

Run closed-loop Gazebo/VRX tests for:

1. straight surge along current heading;
2. low-speed large-radius arc;
3. stop-at-goal;
4. differential-thrust yaw only if negative thrust is allowed.

### Metrics

For every test, report:

- RMS position error;
- final position error;
- RMS yaw error;
- final yaw error;
- RMS velocity error;
- final speed;
- max |tau_v| in prediction;
- RMS tau_v in prediction;
- max |T_L| and max |T_R|;
- solver success rate;
- average solve time;
- maximum solve time.

### Phase gate

Proceed only if:

- solver success rate is acceptable for the scenario;
- controls are finite and not persistently saturated;
- tracking errors decrease or remain bounded;
- tau_v penalty behaves as expected;
- no hidden PID fallback is used.

If the controller fails:

1. test open-loop allocation sign;
2. reduce reference speed and curvature;
3. reduce horizon or simplify cost;
4. check theta_tau sign and frame convention;
5. check yaw wrapping;
6. stop after three focused repair attempts and report.

---

## Phase 3: MINCO-like trajectory optimization with soft sway-force loss

### Goal

Implement standalone trajectory optimization over z(t) = [x(t), y(t), psi(t)]^T.

Do this only after Phase 2 passes.

### Initial condition

Use either:

- manually specified start and local terminal target;
- a synthetic feasible reference from Phase 1;
- later, a front-end-generated initial trajectory.

### Optimization objective

J_traj = J_smooth + J_goal + J_input + J_sway.

Smoothness:

J_smooth = w_s integral ||z^(q)(t)||^2 dt.

Use q = 3 for minimum jerk first.

Goal/local terminal:

J_goal = ||z(T) - z_local_term||^2 with yaw wrapping.

Input:

J_input = w_tau sum_j (tau_u(t_j)^2 + tau_r(t_j)^2).

Sway force:

J_sway = w_v sum_j tau_v(t_j)^2.

### First implementation constraints

Do not use hard tau_v = 0 initially.

Do not optimize total time initially.

Do not include obstacle constraints initially.

After stable operation:

- add allocation feasibility bounds;
- add soft obstacle/corridor cost;
- gradually add |tau_v| <= epsilon_v if needed.

### Tests

Compare w_v = 0 and w_v > 0.

Report:

- terminal error;
- smoothness cost;
- sum tau_v^2 before and after;
- max |tau_v|;
- max |tau_u| and max |tau_r|;
- allocated thrust bounds.

### Phase gate

Proceed only if increasing w_v reduces the sway-force loss without producing invalid or excessively distorted trajectories.

---

## Phase 4: Track MINCO reference with flatness-MPC in Gazebo

### Goal

Connect MINCO output to the flatness-MPC tracker.

### Requirements

- MINCO publishes z_ref, z_dot_ref, and optionally z_ddot_ref.
- Tracker converts z_dot_ref to nu_ref.
- Tracker uses local terminal target from the active reference endpoint.
- Tracker keeps soft tau_v condition active.

### Metrics

Report the same metrics as Phase 2, plus:

- MINCO tau_v RMS;
- tracking-MPC predicted tau_v RMS;
- final local terminal error;
- final speed.

### Phase gate

Proceed only if a MINCO-generated local trajectory can be tracked in Gazebo with bounded errors and feasible thrusts.

---

## Phase 5: Front-end initial trajectory generation

### Goal

Implement Fast-Planner-style initial trajectory generation in flat-output integrator-chain state:

s = [z, z_dot].

This phase starts only after the NMPC and MINCO local trajectory pipeline are verified.

### Important

The front-end is only an initial-guess generator. Do not enforce tau_v = 0 here in the first version.

Check only:

- known-free region;
- collision/footprint;
- velocity/yaw-rate sanity;
- local horizon or local terminal condition.

The MINCO optimizer is responsible for reducing tau_v and input magnitudes.

---

## Phase 6: NN residual compensation

Do not begin until the nominal flatness-MPC + MINCO pipeline is stable.

The residual model should estimate environment disturbance only, independent of control input in the first version.

Do not add Neural-Lander-style fixed-point control unless a later data analysis shows strong control-dependent residuals.

---

## Final instruction for Codex

Proceed phase by phase.

Run Gazebo/VRX tests where required.

Do not claim success without metrics.

Do not proceed after a failed phase.

For each failure, provide:

- hypothesis;
- exact code/config change;
- rerun command;
- metric change;
- conclusion.

Stop after three focused repair attempts per phase.
