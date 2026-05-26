# Repository Working Notes

This repository is the `safe-docking-dev` development branch of the official VRX
Humble codebase. Work should stay on the current branch and inside this
repository; do not switch branches, rebase, or reset unless explicitly asked.

Persistent project context lives in
`docs/codex_context_safe_bay_autonomous_docking.md`. Read that file before
larger safe-docking changes.

Safe docking work should stay modular:

- Keep original VRX worlds, models, launch files, and task behavior usable.
- Add new safe-docking assets with clear `safe_docking` or `dock_2026_safe`
  names instead of modifying the original scan-dock-deliver task in place.
- Reuse VRX WAM-V, marine environment, launch, sensor, dock geometry, and bay
  containment infrastructure.
- Do not implement perception, planning, control, or evaluator behavior until
  the simulation milestone is visually launchable and verified.
- Future autonomy must not consume ground-truth dock pose, target bay index,
  bay detector topics, evaluator state, or Gazebo entity poses. Ground truth is
  acceptable only for evaluator/scoring code.
- Keep changes small and reviewable, with launch/test instructions documented
  in summaries or task docs.
