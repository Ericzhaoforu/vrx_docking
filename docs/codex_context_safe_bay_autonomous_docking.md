# Codex Context: VRX-based Safe-Bay Autonomous Docking

## 0. Purpose of This File

This file is intended to be used as the project context / coding prompt for Codex.

The project is based on the official `osrf/vrx` repository and aims to create a RobotX-2026-style safe-bay autonomous docking task using the existing VRX simulation infrastructure.

Official repository:

```text
https://github.com/osrf/vrx
```

Relevant existing VRX task:

```text
scan_dock_deliver_task.sdf
```

Relevant existing dock model:

```text
dock_2022
```

The goal is not to build a simulator from scratch. The goal is to reuse the official VRX environment, WAM-V model, marine simulation, sensors, existing docking world/model assets, and launch infrastructure, then add a new safe-bay docking task and our own autonomy packages.

---

## 1. Project Goal

Develop a VRX-based USV autonomous docking project.

The specific target is:

> In a VRX simulation environment, create or adapt a three-bay floating dock scene. Each docking bay has a RED/GREEN safety indicator. The WAM-V/USV starts from a randomized initial pose. Without using ground-truth dock pose or target bay information, the USV detects the dock and the bay indicators using camera/LiDAR, selects the GREEN bay, approaches it, aligns with its centerline, enters the bay, stabilizes inside it, and finally reports `CONFIRM_DOCKED`.

The project stops at `CONFIRM_DOCKED`.

Out of scope for now:

```text
- Window/fire target detection after docking
- Water spraying / firefighting
- UAV resource delivery
- USV-UAV coordination
- Full RoboCommand protocol integration
- Full RobotX 2026 Task 3 completion
- Real-world deployment
```

---

## 2. Background and Motivation

The project is inspired by RobotX 2026 Task 3 / Coordinated Logistics, but only implements the initial safe docking stage.

The relevant task logic is:

```text
Locate floating docking platform
-> Identify which docking bay is safe using RED/GREEN indicator
-> Dock into GREEN bay
-> Report successful docking
```

This differs from the older VRX scan-dock-deliver task, which uses:

```text
Scan-the-code buoy
-> color sequence
-> placard color/shape matching
-> correct dock bay
-> projectile delivery
-> exit
```

For this project, the correct bay is determined directly from the RED/GREEN bay safety indicator, not from a scan-code buoy or placard shape/color sequence.

---

## 3. Development Strategy

Use the official `osrf/vrx` repo as the simulation base.

Recommended workspace layout:

```text
robotx_docking_ws/
└── src/
    ├── vrx/                           # forked official osrf/vrx repo
    ├── robotx_safe_docking_sim/        # custom world/model/evaluator/launch
    ├── robotx_safe_docking_msgs/       # custom messages/services/actions
    ├── robotx_safe_docking_perception/ # camera + LiDAR perception
    ├── robotx_safe_docking_mapping/    # semantic dock map
    ├── robotx_safe_docking_planning/   # search + approach + docking planner
    ├── robotx_safe_docking_control/    # dock-relative controller
    └── robotx_safe_docking_manager/    # FSM / mission manager
```

Short-term fast implementation is also acceptable:

```text
- Fork osrf/vrx directly.
- Copy/modify the existing scan_dock_deliver_task.sdf.
- Copy/modify dock_2022 into dock_2026_safe.
- Add custom packages inside the same workspace.
- Keep original VRX tasks intact.
```

Important: avoid breaking the original VRX examples.

---

## 4. Existing VRX Assets to Reuse

The existing VRX `scan_dock_deliver_task.sdf` is a good starting point because it already contains a docking-oriented scene.

Reuse:

```text
- WAM-V platform
- Gazebo / VRX marine environment
- Existing sensor support
- Existing dock model include
- Three-bay dock geometry from dock_2022
- Dock collision geometry
- Bay containment / detector regions if available
- Existing launch structure
```

Remove, ignore, or replace:

```text
- scan-the-code buoy logic
- color sequence topic/service
- placard shape/color matching logic
- projectile target logic
- projectile shooting behavior
- original scan-dock-deliver scoring semantics
```

Add:

```text
- New safe_docking_task.sdf world
- New dock_2026_safe model derived from dock_2022
- RED/GREEN safety indicator panels, one per bay
- Configurable GREEN bay index
- Safe docking evaluator
- Autonomy packages for perception, planning, control, and mission management
```

---

## 5. First Minimal Simulation Milestone

The first Codex task should focus only on simulation setup, not autonomy.

### Target

Create a new world/model pair derived from the existing VRX scan-dock-deliver resources.

### Required changes

1. Inspect the repository structure.
2. Locate the official `scan_dock_deliver_task.sdf`.
3. Locate the official `dock_2022` model.
4. Create a new world named:

```text
safe_docking_task.sdf
```

5. Create a new dock model named:

```text
dock_2026_safe
```

6. The new `dock_2026_safe` model should be derived from `dock_2022`.
7. Keep the three docking bays and the bay geometry.
8. Keep bay containment / detection regions if they already exist.
9. Add simple visual RED/GREEN safety indicator panels, one per bay.
10. For the first version:

```text
bay_1 = RED
bay_2 = GREEN
bay_3 = RED
```

11. The autonomy stack should not be implemented yet.
12. The original VRX worlds and models should remain usable.
13. Add or update launch/config support so the new world can be launched easily.
14. Document how to launch and manually verify the new world.

### Expected result

After the first milestone, I should be able to launch a VRX world containing:

```text
- WAM-V
- Three-bay floating dock
- RED/GREEN safety indicators
- bay_2 as the only GREEN bay
- bay_1 and bay_3 as RED bays
- sensors available for later autonomy development
```

Example desired launch command, depending on VRX branch/package structure:

```bash
ros2 launch vrx_gz competition.launch.py world:=safe_docking_task
```

or a custom launch file:

```bash
ros2 launch robotx_safe_docking_sim safe_docking.launch.py
```

---

## 6. Autonomy Must Not Use Ground Truth

The autonomy stack must not directly subscribe to ground-truth information such as:

```text
- dock true pose
- target GREEN bay index
- Gazebo entity pose of dock
- bay containment detector topics
- evaluator result
- scoring plugin state
```

The autonomy stack should only use onboard-like sensor inputs:

```text
- camera image
- LiDAR point cloud or laser scan
- IMU
- GPS / odometry
```

However, an evaluator/scoring node may use ground truth to determine whether the task is successful.

Keep these separated:

```text
/autonomy/status      # state estimated by autonomy stack
/eval/status          # ground-truth-based evaluation state
```

---

## 7. Safe Docking Evaluator Design

After the simulation world is created, implement a lightweight evaluator.

The evaluator can use ground-truth or existing bay containment/detector topics if available.

Evaluator should publish:

```text
/safe_docking/eval_status
/safe_docking/confirm_docked
```

Recommended ground-truth success condition:

```text
USV center is inside target GREEN bay polygon
AND target bay is GREEN
AND heading error relative to bay direction < 10~15 deg
AND lateral error to bay centerline < 0.5~1.0 m
AND speed < 0.3~0.5 m/s
AND no severe collision
AND all conditions hold for 3~5 seconds
```

Initial simplified version:

```text
USV is inside bay_2 contain region
AND bay_2 is configured as GREEN
AND condition holds for 3~5 seconds
=> publish CONFIRM_DOCKED
```

Manual test:

```text
Teleoperate WAM-V into bay_2.
Evaluator should publish CONFIRM_DOCKED.
Teleoperate WAM-V into bay_1 or bay_3.
Evaluator should not publish CONFIRM_DOCKED.
```

---

## 8. Full Autonomy Architecture

Eventually, the autonomy stack should look like this:

```text
VRX / Gazebo
  ├── WAM-V
  ├── 3-bay floating dock
  ├── RED/GREEN indicators
  └── evaluator

ROS 2 Autonomy Stack
  ├── perception
  │   ├── dock_detector_lidar
  │   ├── bay_geometry_estimator
  │   ├── indicator_detector_camera
  │   └── bay_indicator_association
  │
  ├── semantic_dock_map
  │   ├── dock pose
  │   ├── bay entrance poses
  │   ├── bay centerlines
  │   ├── bay RED/GREEN status
  │   └── target bay id
  │
  ├── mission_manager
  │   ├── SEARCH_DOCK_PLATFORM
  │   ├── LOCALIZE_DOCK
  │   ├── DETECT_BAY_INDICATORS
  │   ├── SELECT_GREEN_BAY
  │   ├── APPROACH_PRE_DOCK_POSE
  │   ├── ALIGN_WITH_BAY_CENTERLINE
  │   ├── ENTER_DOCK
  │   └── CONFIRM_DOCKED
  │
  ├── planner
  │   ├── search planner
  │   ├── pre-dock pose generator
  │   ├── approach planner
  │   └── final docking path generator
  │
  ├── controller
  │   ├── waypoint / LOS controller
  │   ├── dock-relative alignment controller
  │   ├── low-speed entry controller
  │   └── hold controller
  │
  └── thruster allocation
```

---

## 9. Mission FSM

The mission manager should eventually implement this finite-state machine:

```text
INIT
  -> SEARCH_DOCK_PLATFORM
  -> LOCALIZE_DOCK
  -> DETECT_BAY_INDICATORS
  -> SELECT_GREEN_BAY
  -> APPROACH_PRE_DOCK_POSE
  -> ALIGN_WITH_BAY_CENTERLINE
  -> ENTER_DOCK
  -> HOLD_IN_BAY
  -> CONFIRM_DOCKED
  -> FINISH
```

State responsibilities:

```text
SEARCH_DOCK_PLATFORM:
  Move according to search pattern until dock candidate is detected.

LOCALIZE_DOCK:
  Estimate dock pose, bay entrance poses, and bay centerlines from LiDAR/camera.

DETECT_BAY_INDICATORS:
  Detect RED/GREEN indicator for each bay using camera.

SELECT_GREEN_BAY:
  Select bay with GREEN indicator and sufficient confidence.

APPROACH_PRE_DOCK_POSE:
  Move to a pre-dock pose outside target bay.

ALIGN_WITH_BAY_CENTERLINE:
  Align USV heading and lateral position with target bay centerline.

ENTER_DOCK:
  Move forward slowly into target bay while maintaining small lateral/yaw error.

HOLD_IN_BAY:
  Stop or hold low velocity inside target bay for several seconds.

CONFIRM_DOCKED:
  Publish success state once stable docking conditions are met.
```

---

## 10. Perception Baseline Design

### 10.1 LiDAR Dock Geometry

Input:

```text
/points or /scan
```

Output concept:

```yaml
DockGeometry:
  dock_pose
  bays:
    - id
      entrance_pose
      centerline_pose
      bay_polygon
      confidence
```

First baseline method:

```text
point cloud filtering
-> clustering
-> line fitting / edge extraction
-> infer three bay entrances
-> estimate dock frame
```

### 10.2 Camera RED/GREEN Indicator Detection

Input:

```text
/camera/image
```

Output concept:

```yaml
BayIndicatorArray:
  bay_1: RED / GREEN / UNKNOWN
  bay_2: RED / GREEN / UNKNOWN
  bay_3: RED / GREEN / UNKNOWN
```

First baseline method:

```text
HSV or Lab color threshold
-> morphology filtering
-> contour detection
-> bbox filtering
-> temporal voting
-> associate detection to bay ROI
```

Do not implement YOLO in the first version. Keep the detector interface modular so YOLO or segmentation can replace it later.

---

## 11. Semantic Dock Map

The system should maintain a structured semantic dock map:

```yaml
dock:
  pose_world: [x, y, yaw]
  confidence: 0.90

bays:
  - id: 1
    entrance_pose_world: [x, y, yaw]
    centerline_pose_world: [x, y, yaw]
    indicator_color: RED
    indicator_confidence: 0.91
    status: unsafe

  - id: 2
    entrance_pose_world: [x, y, yaw]
    centerline_pose_world: [x, y, yaw]
    indicator_color: GREEN
    indicator_confidence: 0.88
    status: safe

  - id: 3
    entrance_pose_world: [x, y, yaw]
    centerline_pose_world: [x, y, yaw]
    indicator_color: RED
    indicator_confidence: 0.93
    status: unsafe

target_bay_id: 2
```

Planner and controller should consume this semantic map rather than raw image detections.

---

## 12. Planning and Control Baseline

### 12.1 Key Poses

After selecting the GREEN bay, generate:

```text
pre_dock_pose:
  8~10 m outside bay entrance, aligned with bay centerline

align_pose:
  3~5 m outside bay entrance, aligned with bay centerline

final_dock_pose:
  inside target bay
```

### 12.2 Far-range Approach

Use simple methods first:

```text
waypoint tracking
LOS guidance
pure pursuit
```

### 12.3 Near-range Dock-relative Control

Define dock-frame errors:

```text
e_y   = lateral error to bay centerline
e_psi = heading error relative to bay direction
e_x   = longitudinal progress into bay
```

First controller:

```text
u_cmd = small positive surge speed
r_cmd = -k_y * e_y - k_psi * e_psi
```

If WAM-V configuration supports lateral force, optionally use:

```text
v_cmd = -k_vy * e_y
```

Entry speed should be low:

```text
0.3~0.6 m/s
```

If lateral error becomes too large, abort entry and return to align pose.

---

## 13. Development Milestones

### Milestone 1: Run official VRX task

```text
Launch scan_dock_deliver_task.
Verify WAM-V, dock, sensors, teleop.
```

### Milestone 2: Create safe docking world

```text
Copy scan_dock_deliver_task.sdf.
Create safe_docking_task.sdf.
Copy dock_2022 to dock_2026_safe.
Add RED/GREEN bay indicators.
```

### Milestone 3: Evaluator

```text
Manual drive into GREEN bay.
Evaluator publishes CONFIRM_DOCKED.
No autonomy yet.
```

### Milestone 4: Perception baseline

```text
LiDAR detects dock/bay geometry.
Camera detects RED/GREEN indicators.
Semantic dock map publishes target bay.
```

### Milestone 5: Planning/control baseline

```text
Search dock.
Approach pre-dock pose.
Align with target bay.
Enter bay.
Hold.
```

### Milestone 6: Full autonomous loop

```text
Random initial USV pose.
Random GREEN bay.
No ground-truth access for autonomy.
System reaches CONFIRM_DOCKED.
```

### Milestone 7: Batch tests

```text
Run 50~100 randomized trials.
Measure success rate, collision count, time-to-dock, final lateral/yaw error.
```

---

## 14. Metrics for Evaluation

Recommended metrics:

```text
- Docking success rate
- Correct GREEN bay selection rate
- Time to detect dock
- Time to dock
- Collision count
- Final lateral error
- Final yaw error
- Minimum distance to dock wall
- Indicator classification accuracy
- Dock/bay geometry estimation error
- Robustness under randomized initial poses
- Robustness under wind/wave/lighting variations
```

---

## 15. First Prompt for Codex

Use this exact prompt for the first coding step:

```text
I forked the official osrf/vrx repository and want to build a RobotX-2026-style safe-bay autonomous docking task based on the existing VRX scan_dock_deliver_task.

Please inspect the repository structure and implement the first minimal simulation milestone:

1. Locate the existing scan_dock_deliver_task.sdf world and the dock_2022 model.
2. Create a new world called safe_docking_task.sdf derived from scan_dock_deliver_task.sdf.
3. Create a new dock model called dock_2026_safe derived from dock_2022.
4. Keep the three docking bays and the bay containment/detection geometry if present.
5. Replace or augment the old placard/scan-dock-deliver visual targets with simple RED/GREEN safety indicator panels, one per bay.
6. For now, set bay_2 as GREEN and bay_1/bay_3 as RED.
7. Do not implement perception, planning, or control yet.
8. Add or update launch/config files so I can run the new world.
9. Keep changes modular and avoid breaking the original VRX tasks.
10. After making changes, summarize modified files and how to launch/test the world.

The final goal of this project is: WAM-V starts from a randomized initial pose, detects the dock and bay indicators using camera/LiDAR without ground-truth dock information, selects the GREEN bay, docks into it, and publishes CONFIRM_DOCKED. But for this first milestone, only create and launch the modified simulation world.
```

---

## 16. Important Engineering Notes

- Prefer small, reviewable changes.
- Do not delete original VRX worlds/models.
- Keep new files clearly named with `safe_docking` or `dock_2026_safe`.
- Avoid implementing autonomy before the modified world is visually verified.
- Do not make the autonomy stack depend on evaluator ground truth.
- Use evaluator ground truth only for scoring and debugging.
- First make manual teleoperation into the GREEN bay work.
- Then implement evaluator.
- Then implement perception.
- Then implement planning/control.
