# Gazebo Performance Videos

Curated Gazebo GUI recordings for quick review on GitHub. These are intentionally
small MP4/GIF exports, not the full raw run directories.

## Color Legend

- Orange spheres: full synthetic reference path embedded in the recording world.
- Yellow spheres: active reference sampled by the running NMPC controller.
- Cyan spheres: WAM-V EKF / executed trajectory trail.
- Green sphere: global goal marker for the whole-pipeline MINCO-topic run.

## Recordings

| Scenario | Source run | Video | GIF |
| --- | --- | --- | --- |
| NMPC `figure8` synthetic reference | `output/nmpc_figure8_gazebo_video_20260611_025906/` | `nmpc_individual/figure8/nmpc_figure8_gazebo.mp4` | `nmpc_individual/figure8/nmpc_figure8_gazebo.gif` |
| NMPC `spiral` synthetic reference | `output/nmpc_spiral_gazebo_video_20260611_030736/` | `nmpc_individual/spiral/nmpc_spiral_gazebo.mp4` | `nmpc_individual/spiral/nmpc_spiral_gazebo.gif` |
| Full pipeline MINCO topic handoff | `output/whole_pipeline_minco_topic_gazebo_video_20260611_031405/` | `whole_pipeline_minco_topic/whole_pipeline_minco_topic_gazebo.mp4` | `whole_pipeline_minco_topic/whole_pipeline_minco_topic_gazebo.gif` |

The full-pipeline recording runs:

```text
EKF -> C++ goal_lattice MINCO replanner -> minco_topic ACK handoff -> RK4 NMPC -> differential thrust
```

The whole-pipeline planner log accepted 10 MINCO references with controller ACK
during the 70-second recording. Some intermediate MINCO candidates reached the
optimizer iteration limit and were rejected, but the accepted references were
executed through the topic handoff.

## Reproduce

Synthetic NMPC examples:

```bash
REFERENCE_NAME=figure8 OUTPUT_LABEL=nmpc_figure8 OUTPUT_PREFIX=nmpc_figure8_gazebo \
  GIF_FPS=6 GIF_SCALE=720 DURATION=90 \
  scripts/record_nmpc_figure8_gazebo_gif.sh

REFERENCE_NAME=spiral OUTPUT_LABEL=nmpc_spiral OUTPUT_PREFIX=nmpc_spiral_gazebo \
  GIF_FPS=6 GIF_SCALE=720 DURATION=90 \
  scripts/record_nmpc_figure8_gazebo_gif.sh
```

Whole-pipeline recording:

```bash
REFERENCE_SOURCE=minco_topic REFERENCE_NAME=goal_lattice \
  OUTPUT_LABEL=whole_pipeline_minco_topic \
  OUTPUT_PREFIX=whole_pipeline_minco_topic_gazebo \
  GIF_FPS=6 GIF_SCALE=720 DURATION=70 \
  VIS_MIN_STEP_M=0.12 VIS_ACTIVE_REFERENCE_RADIUS=0.24 \
  VIS_ACTIVE_REFERENCE_Z_OFFSET=2.60 \
  scripts/record_nmpc_figure8_gazebo_gif.sh
```
