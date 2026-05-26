# Safe Docking Handoff Context

Date: 2026-05-27

Repository: `/home/eric2204/vrx_docking`

Branch: `safe-docking-dev`

Remote: `origin git@github.com:Ericzhaoforu/vrx_docking.git`

## What Was Added

This repository now has the first safe-docking simulation milestone layered on
top of the official VRX Humble branch.

New project context files:

- `AGENTS.md`
- `docs/codex_context_safe_bay_autonomous_docking.md`
- `docs/safe_docking_handoff_context.md`

New safe docking simulation assets:

- `vrx_gz/worlds/safe_docking_task.sdf`
- `vrx_gz/models/dock_2026_safe/model.sdf`
- `vrx_gz/models/dock_2026_safe/model.config`
- `vrx_gz/launch/safe_docking.launch.py`
- `vrx_gz/config/safe_docking_wamv.yaml`
- `vrx_gz/config/safe_docking_wamv_norender.yaml`

Supporting package changes:

- `vrx_gz/src/vrx_gz/model.py`
- `vrx_gz/package.xml`
- `.gitignore`

## Safe Docking World

`safe_docking_task.sdf` is derived from the existing VRX
`scan_dock_deliver_task.sdf`, but it keeps the change modular by using a new
world name and a new dock model.

The new dock model is `dock_2026_safe`, derived from the existing `dock_2022`
geometry. It keeps the three-bay dock layout and bay containment detector
plugins. The initial safe indicators are:

- `bay_1`: RED
- `bay_2`: GREEN
- `bay_3`: RED

The scan-the-code buoy, placard target semantics, projectile target, and
projectile delivery pieces were not carried into the new safe docking task.
No perception, planning, control, or custom evaluator behavior has been added.

## WAM-V Configuration

`safe_docking_wamv.yaml` is the intended sensor-bearing WAM-V configuration for
future autonomy work. It disables the old VRX full sensor bundle and explicitly
enables only the relevant safe-docking sensors:

- camera enabled
- GPS enabled
- IMU enabled
- LiDAR enabled
- pinger disabled
- ball shooter disabled

`safe_docking_wamv_norender.yaml` is a fallback smoke-test configuration for
machines that cannot provide the OpenGL version required by Gazebo/Ogre2. It
keeps GPS, IMU, thrusters, and the locked-start behavior, but disables camera
and GPU LiDAR.

## Launch Commands

Normal visual/sensor launch on a machine with OpenGL 3.3 or newer:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch vrx_gz safe_docking.launch.py
```

Headless no-render smoke test for the current VMware machine:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch vrx_gz safe_docking.launch.py \
  headless:=True \
  config_file:=$(pwd)/install/vrx_gz/share/vrx_gz/config/safe_docking_wamv_norender.yaml
```

## Build And Verification Performed

Build command:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select vrx_gz
source install/setup.bash
```

Checks performed:

- `colcon build --packages-select vrx_gz` passed.
- `python3 -m py_compile vrx_gz/src/vrx_gz/model.py vrx_gz/launch/safe_docking.launch.py` passed.
- `git diff --check` passed.
- Headless no-render launch reached:
  - `safe_docking_task.sdf` world load
  - `dock_2026_safe` bay detector startup
  - WAM-V spawn
  - GPS/IMU/thruster bridges
  - task transition to running

Important successful log lines:

```text
Loading SDF world file ... safe_docking_task.sdf
PerformerDetector publishing messages on /vrx/dock_2026_safe/bay_1/contain
PerformerDetector publishing messages on /vrx/dock_2026_safe/bay_2/contain
PerformerDetector publishing messages on /vrx/dock_2026_safe/bay_3/contain
OK creation of entity.
World [safe_docking_task] initialized
Created entity [211] named [wamv]
ScoringPlugin::OnRunning
```

## VMware OpenGL Issue

The current VMware machine cannot run the Gazebo GUI or render-backed sensors.
The VM setting "Accelerate 3D graphics" was enabled, and Ubuntu was switched
from Wayland to X11, but `glxinfo -B` still reported:

```text
Accelerated: no
Video memory: 1MB
Max core profile version: 0.0
Max compat profile version: 2.1
OpenGL version string: 2.1 Mesa 23.2.1-1ubuntu3.1~22.04.3
```

Gazebo Sim Fortress / Garden with Ogre2 needs OpenGL 3.3 or newer for the GUI,
camera, and GPU LiDAR paths. On this VM, the full visual launch failed with
errors like:

```text
GLXBadFBConfig
OpenGL 3.3 is not supported
Unable to create the rendering window
```

This is a host/VM graphics acceleration limitation, not a safe-docking SDF
problem.

## Next Machine Checklist

On a GL-capable machine, pull this branch and rebuild:

```bash
git clone git@github.com:Ericzhaoforu/vrx_docking.git
cd vrx_docking
git checkout safe-docking-dev
source /opt/ros/humble/setup.bash
colcon build --packages-up-to vrx_gz
source install/setup.bash
```

Before running the visual launch, check:

```bash
glxinfo -B
```

The output should show OpenGL 3.3 or newer. Then run:

```bash
ros2 launch vrx_gz safe_docking.launch.py
```

If the new machine still has graphics issues, use the no-render fallback to
verify world/task plumbing while leaving perception work for a proper
OpenGL-capable environment.
