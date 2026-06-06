import numpy as np

from robotx_safe_docking_control.minco_s3nu import MINCO_S3NU


def representative_minco():
    head_pva = np.array([
        [0.0, 0.0, 0.0],
        [0.2, -0.1, 0.05],
        [0.01, 0.02, -0.01],
    ])
    tail_pva = np.array([
        [3.0, 1.0, 0.35],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    inPs = np.array([
        [0.8, 0.15, 0.08],
        [1.8, 0.55, 0.20],
        [2.5, 0.85, 0.30],
    ])
    ts = np.array([0.9, 1.15, 0.85, 1.25])
    traj = MINCO_S3NU(dim=3)
    traj.set_conditions(head_pva, tail_pva)
    traj.set_parameters(inPs, ts)
    return traj, head_pva, tail_pva, inPs, ts


def test_start_pva_constraints():
    traj, head_pva, _tail_pva, _inPs, _ts = representative_minco()
    sample = traj.sample(0.0)
    assert np.allclose(sample.z, head_pva[0], atol=1e-10)
    assert np.allclose(sample.z_dot, head_pva[1], atol=1e-10)
    assert np.allclose(sample.z_ddot, head_pva[2], atol=1e-10)


def test_terminal_pva_constraints():
    traj, _head_pva, tail_pva, _inPs, _ts = representative_minco()
    sample = traj.sample(traj.duration())
    assert np.allclose(sample.z, tail_pva[0], atol=1e-10)
    assert np.allclose(sample.z_dot, tail_pva[1], atol=1e-10)
    assert np.allclose(sample.z_ddot, tail_pva[2], atol=1e-10)


def test_intermediate_point_pass_through():
    traj, _head_pva, _tail_pva, inPs, ts = representative_minco()
    for joint_index, point in enumerate(inPs):
        left = traj.sample_piece(joint_index, ts[joint_index])
        right = traj.sample_piece(joint_index + 1, 0.0)
        assert np.allclose(left.z, point, atol=1e-10)
        assert np.allclose(right.z, point, atol=1e-10)


def test_velocity_acceleration_jerk_snap_continuity():
    traj, _head_pva, _tail_pva, inPs, ts = representative_minco()
    fields = ["z_dot", "z_ddot", "z_jerk", "z_snap"]
    for joint_index in range(len(inPs)):
        left = traj.sample_piece(joint_index, ts[joint_index])
        right = traj.sample_piece(joint_index + 1, 0.0)
        for field in fields:
            assert np.allclose(
                getattr(left, field),
                getattr(right, field),
                atol=1e-9,
            )


def test_analytical_jerk_energy_matches_numerical_quadrature():
    traj, _head_pva, _tail_pva, _inPs, ts = representative_minco()
    quadrature_energy = 0.0
    nodes, weights = np.polynomial.legendre.leggauss(32)
    for piece_index, T in enumerate(ts):
        for node, weight in zip(nodes, weights):
            local_time = 0.5 * T * (node + 1.0)
            jerk = traj.sample_piece(piece_index, local_time).z_jerk
            quadrature_energy += 0.5 * T * weight * float(jerk @ jerk)
    assert abs(traj.get_energy() - quadrature_energy) < 1e-9


def test_adjoint_energy_gradient_matches_finite_difference():
    traj, head_pva, tail_pva, inPs, ts = representative_minco()
    gradients = traj.get_energy_gradients_by_parameters()
    analytical = np.concatenate([
        gradients["inPs"].reshape(-1),
        gradients["ts"],
    ])
    base_energy = traj.get_energy()
    eps = 1e-6
    finite_difference = []

    flat_inPs = inPs.reshape(-1)
    for index in range(flat_inPs.size):
        perturbed = flat_inPs.copy()
        perturbed[index] += eps
        candidate = MINCO_S3NU(dim=3)
        candidate.set_conditions(head_pva, tail_pva)
        candidate.set_parameters(perturbed.reshape(inPs.shape), ts)
        finite_difference.append((candidate.get_energy() - base_energy) / eps)

    for index in range(ts.size):
        perturbed_ts = ts.copy()
        perturbed_ts[index] += eps
        candidate = MINCO_S3NU(dim=3)
        candidate.set_conditions(head_pva, tail_pva)
        candidate.set_parameters(inPs, perturbed_ts)
        finite_difference.append((candidate.get_energy() - base_energy) / eps)

    finite_difference = np.asarray(finite_difference, dtype=float)
    relative_error = np.linalg.norm(analytical - finite_difference) / max(
        1.0, np.linalg.norm(finite_difference)
    )
    assert relative_error < 1e-5
