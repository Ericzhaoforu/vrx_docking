#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DURATION="${DURATION:-90}"
FPS="${FPS:-15}"
GIF_FPS="${GIF_FPS:-8}"
GIF_SCALE="${GIF_SCALE:-960}"
SCREEN="${SCREEN:-1280x720x24}"
VIDEO_SIZE="${VIDEO_SIZE:-1280x720}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-synthetic}"
REFERENCE_NAME="${REFERENCE_NAME:-figure8}"
OUTPUT_LABEL="${OUTPUT_LABEL:-nmpc_${REFERENCE_NAME}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OUTPUT_LABEL}_gazebo}"
STATIC_REFERENCE="${STATIC_REFERENCE:-auto}"
GLOBAL_GOAL="${GLOBAL_GOAL:-3.0,6.0,2.4}"
VIS_MIN_STEP_M="${VIS_MIN_STEP_M:-0.25}"
VIS_ACTIVE_REFERENCE_RADIUS="${VIS_ACTIVE_REFERENCE_RADIUS:-0.16}"
VIS_ACTIVE_REFERENCE_Z_OFFSET="${VIS_ACTIVE_REFERENCE_Z_OFFSET:-1.85}"
VIS_ACTUAL_RADIUS="${VIS_ACTUAL_RADIUS:-0.18}"
VIS_ACTUAL_Z_OFFSET="${VIS_ACTUAL_Z_OFFSET:-2.15}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output/${OUTPUT_LABEL}_gazebo_video_${STAMP}}"
mkdir -p "${OUT_DIR}"

cleanup_processes() {
  pkill -f "ros2 launch vrx_gz safe_docking.launch.py" >/dev/null 2>&1 || true
  pkill -f "ros2 launch robotx_safe_docking_estimation estimator.launch.py" >/dev/null 2>&1 || true
  pkill -f "ros2 launch robotx_safe_docking_control flatness_mpc_controller.launch.py" >/dev/null 2>&1 || true
  pkill -f "ros2 launch robotx_safe_docking_planning minco_replanner.launch.py" >/dev/null 2>&1 || true
  pkill -f "minco_replanner_node" >/dev/null 2>&1 || true
  pkill -f "gazebo_nmpc_trail_visualizer.py" >/dev/null 2>&1 || true
  pkill -f "gazebo_nmpc_entity_trail_visualizer.py" >/dev/null 2>&1 || true
  pkill -f "ruby .*gz.* sim" >/dev/null 2>&1 || true
  pkill -f "gz sim" >/dev/null 2>&1 || true
  pkill -f "parameter_bridge" >/dev/null 2>&1 || true
  pkill -f "robot_state_publisher" >/dev/null 2>&1 || true
  pkill -f "gps_imu_ekf_node" >/dev/null 2>&1 || true
  ros2 daemon stop >/dev/null 2>&1 || true
  sleep 2
}

if [[ "${INSIDE_XVFB:-0}" != "1" ]]; then
  export ROOT_DIR OUT_DIR DURATION FPS GIF_FPS GIF_SCALE VIDEO_SIZE SCREEN
  export REFERENCE_SOURCE REFERENCE_NAME OUTPUT_LABEL OUTPUT_PREFIX STATIC_REFERENCE GLOBAL_GOAL
  export VIS_MIN_STEP_M VIS_ACTIVE_REFERENCE_RADIUS VIS_ACTIVE_REFERENCE_Z_OFFSET
  export VIS_ACTUAL_RADIUS VIS_ACTUAL_Z_OFFSET
  INSIDE_XVFB=1 xvfb-run -a -s "-screen 0 ${SCREEN}" "$0"
  echo "Saved video and GIF to ${OUT_DIR}"
  exit 0
fi

run_inside_xvfb() {
  set +u
  source /opt/ros/humble/setup.bash
  source "${ROOT_DIR}/install/setup.bash"
  set -u
  export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}:${ROOT_DIR}/install/wamv_description/share:${ROOT_DIR}/install/wamv_gazebo/share"
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

  cleanup_processes
  RECORDING_WORLD="${OUT_DIR}/safe_docking_task_recording.sdf"
  if [[ "${STATIC_REFERENCE}" == "auto" ]]; then
    if [[ "${REFERENCE_SOURCE}" == "synthetic" ]]; then
      STATIC_REFERENCE="1"
    else
      STATIC_REFERENCE="0"
    fi
  fi
  /usr/bin/python3 - <<PY
from pathlib import Path
import numpy as np

from robotx_safe_docking_control.feasible_references import make_synthetic_reference
from robotx_safe_docking_control.usv_flatness import UsvModelParams

src = Path("${ROOT_DIR}/vrx_gz/worlds/safe_docking_task.sdf")
dst = Path("${RECORDING_WORLD}")
text = src.read_text()
text = text.replace(
    '<world name="safe_docking_task">',
    '<world name="safe_docking_task_recording">',
    1)
text = text.replace(
    '<camera_pose>-478.1 148.2 13.2 0 0.25 2.94</camera_pose>',
    '<camera_pose>-536.2 167.8 58.0 0 1.55 0.0</camera_pose>',
    1)

def reference_sphere_model(name, x, y, z):
    rgba = "1.000 0.350 0.000 1.000"
    return f'''
      <model name="{name}">
        <static>true</static>
        <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
        <link name="link">
          <visual name="visual">
            <cast_shadows>false</cast_shadows>
            <geometry><sphere><radius>0.22</radius></sphere></geometry>
            <material>
              <ambient>{rgba}</ambient>
              <diffuse>{rgba}</diffuse>
              <emissive>{rgba}</emissive>
            </material>
          </visual>
        </link>
      </model>'''

def goal_marker_model(name, x, y, z):
    rgba = "0.000 0.900 0.150 1.000"
    return f'''
      <model name="{name}">
        <static>true</static>
        <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
        <link name="link">
          <visual name="visual">
            <cast_shadows>false</cast_shadows>
            <geometry><sphere><radius>0.35</radius></sphere></geometry>
            <material>
              <ambient>{rgba}</ambient>
              <diffuse>{rgba}</diffuse>
              <emissive>{rgba}</emissive>
            </material>
          </visual>
        </link>
      </model>'''

models = []
if "${STATIC_REFERENCE}" == "1":
    params = UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    ref = make_synthetic_reference(
        "${REFERENCE_NAME}",
        np.array([0.0, 0.0, 1.0], dtype=float),
        params,
        dt=0.2,
        initial_nu=np.zeros(3, dtype=float))
    for index, row in enumerate(ref.z[::8]):
        models.append(reference_sphere_model(
            f"nmpc_full_reference_{index:04d}",
            -532.0 + float(row[0]),
            162.0 + float(row[1]),
            0.35 + 1.55))
else:
    try:
        gx, gy, _ = [float(item) for item in "${GLOBAL_GOAL}".split(",")]
        models.append(goal_marker_model(
            "nmpc_global_goal_marker",
            -532.0 + gx,
            162.0 + gy,
            0.35 + 2.00))
    except Exception:
        pass
text = text.replace('</world>', '\\n'.join(models) + '\\n</world>', 1)
dst.write_text(text)
PY

  PIDS=()
  cleanup_children() {
    set +e
    for pid in "${PIDS[@]:-}"; do
      kill "${pid}" >/dev/null 2>&1 || true
    done
    cleanup_processes
  }
  trap cleanup_children EXIT

  ros2 launch vrx_gz safe_docking.launch.py \
    world:="${RECORDING_WORLD%.sdf}" \
    headless:=False \
    launch_rviz:=False \
    config_file:="${ROOT_DIR}/vrx_gz/config/safe_docking_wamv_norender_ground_truth.yaml" \
    > "${OUT_DIR}/gazebo.log" 2>&1 &
  PIDS+=("$!")

  sleep 18
  gz service -s /gui/move_to/model \
    --reqtype gz.msgs.StringMsg \
    --reptype gz.msgs.Boolean \
    --timeout 1000 \
    --req 'data: "wamv"' \
    >> "${OUT_DIR}/gazebo_camera.log" 2>&1 || true

  ros2 launch robotx_safe_docking_estimation estimator.launch.py \
    use_sim_time:=True \
    verify:=False \
    > "${OUT_DIR}/ekf.log" 2>&1 &
  PIDS+=("$!")

  sleep 5
  /usr/bin/python3 scripts/gazebo_nmpc_entity_trail_visualizer.py \
    --world-name safe_docking_task_recording \
    --reference-name "${REFERENCE_NAME}" \
    --skip-static-reference \
    --reference-stride 8 \
    --min-step-m "${VIS_MIN_STEP_M}" \
    --active-reference-radius "${VIS_ACTIVE_REFERENCE_RADIUS}" \
    --active-reference-z-offset "${VIS_ACTIVE_REFERENCE_Z_OFFSET}" \
    --actual-radius "${VIS_ACTUAL_RADIUS}" \
    --actual-z-offset "${VIS_ACTUAL_Z_OFFSET}" \
    > "${OUT_DIR}/visualizer.log" 2>&1 &
  PIDS+=("$!")

  sleep 2
  ffmpeg -y \
    -f x11grab \
    -video_size "${VIDEO_SIZE}" \
    -framerate "${FPS}" \
    -i "${DISPLAY}" \
    -t "${DURATION}" \
    -c:v libx264 \
    -preset veryfast \
    -pix_fmt yuv420p \
    "${OUT_DIR}/${OUTPUT_PREFIX}.mp4" \
    > "${OUT_DIR}/ffmpeg_record.log" 2>&1 &
  FFMPEG_PID="$!"

  sleep 1
  if [[ "${REFERENCE_SOURCE}" == "minco_topic" ]]; then
    ros2 launch robotx_safe_docking_control flatness_mpc_controller.launch.py \
      reference_source:=minco_topic \
      use_sim_time:=True \
      > "${OUT_DIR}/controller.log" 2>&1 &
    PIDS+=("$!")

    sleep 2
    ros2 launch robotx_safe_docking_planning minco_replanner.launch.py \
      use_sim_time:=True \
      > "${OUT_DIR}/planner.log" 2>&1 &
    PIDS+=("$!")
  else
    ros2 launch robotx_safe_docking_control flatness_mpc_controller.launch.py \
      reference_source:="${REFERENCE_SOURCE}" \
      reference_name:="${REFERENCE_NAME}" \
      use_sim_time:=True \
      > "${OUT_DIR}/controller.log" 2>&1 &
    PIDS+=("$!")
  fi

  wait "${FFMPEG_PID}"

  ffmpeg -y \
    -i "${OUT_DIR}/${OUTPUT_PREFIX}.mp4" \
    -vf "fps=${GIF_FPS},scale=${GIF_SCALE}:-1:flags=lanczos,palettegen" \
    "${OUT_DIR}/palette.png" \
    > "${OUT_DIR}/ffmpeg_palette.log" 2>&1
  ffmpeg -y \
    -i "${OUT_DIR}/${OUTPUT_PREFIX}.mp4" \
    -i "${OUT_DIR}/palette.png" \
    -filter_complex "fps=${GIF_FPS},scale=${GIF_SCALE}:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    "${OUT_DIR}/${OUTPUT_PREFIX}.gif" \
    > "${OUT_DIR}/ffmpeg_gif.log" 2>&1

  ffmpeg -y -ss 30 -i "${OUT_DIR}/${OUTPUT_PREFIX}.mp4" \
    -frames:v 1 "${OUT_DIR}/frame_30s.png" \
    > "${OUT_DIR}/ffmpeg_frame_30s.log" 2>&1 || true
  ffmpeg -y -ss 75 -i "${OUT_DIR}/${OUTPUT_PREFIX}.mp4" \
    -frames:v 1 "${OUT_DIR}/frame_75s.png" \
    > "${OUT_DIR}/ffmpeg_frame_75s.log" 2>&1 || true

  printf '%s\n' "${OUT_DIR}" > "${ROOT_DIR}/output/latest_${OUTPUT_LABEL}_gazebo_video_dir.txt"
}

run_inside_xvfb
