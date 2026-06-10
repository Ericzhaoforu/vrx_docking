# Good Performance Figures

This folder contains a curated subset of closed-loop figures for quick review on
GitHub. It intentionally avoids the full raw `output/` tree.

## NMPC Individual Tests

Source run:

```text
output/nmpc_rk4_tight005_all_refs_clean_rerun_20260604/
```

These are controller-only tests using feasible synthetic references and the RK4
dynamics NMPC.

| Reference | Final distance (m) | Final yaw error (rad) | RMS reference distance (m) | Solver success | Saturation fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hold` | 0.043 | 0.0034 | 0.032 | 1.000 | 0.0000 |
| `straight` | 0.012 | 0.0002 | 0.182 | 1.000 | 0.0000 |
| `arc` | 0.044 | 0.0019 | 0.123 | 1.000 | 0.0000 |
| `stop` | 0.020 | 0.0023 | 0.122 | 1.000 | 0.0007 |
| `spiral` | 0.045 | 0.0008 | 0.216 | 1.000 | 0.0008 |
| `yaw` | 0.073 | 0.0102 | 0.101 | 1.000 | 0.0055 |
| `figure8` | 0.024 | 0.0004 | 0.208 | 1.000 | 0.0034 |

Each subfolder contains:

```text
01_trajectory_xy.png
02_states.png
03_velocities.png
04_forces.png
05_errors_solver.png
metrics.json
```

## Whole Framework Test

Source run:

```text
output/nmpc_recommended_terminal_sweep_20260611_013620/baseline_3_6_2p4/
```

This is the current full stack:

```text
EKF -> C++ goal_lattice MINCO replanner -> minco_topic ACK handoff -> RK4 NMPC -> differential thrust
```

Goal:

```text
(x, y, psi) = (3.0, 6.0, 2.4)
```

| Metric | Value |
| --- | ---: |
| final global distance | 0.074 m |
| final yaw error | 0.0014 rad |
| RMS active-reference distance | 0.259 m |
| RMS active-reference yaw error | 0.123 rad |
| solver success | 1.000 |
| mean solve time | 16.6 ms |
| thrust saturation fraction | 0.0000 |
| accepted reference swaps | 9 |
| accepted-reference max `abs(tau_v)` | 9.31 N |

The `whole_framework_minco_topic/` folder contains the full plot set, including
reference timing and body/world velocity plots.
