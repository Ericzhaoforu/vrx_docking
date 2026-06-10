import math

import numpy as np

from robotx_safe_docking_control.feasible_references import FeasibleReference
from robotx_safe_docking_control.feasible_references import ReferenceCommittedSwitch
from robotx_safe_docking_control.feasible_references import ReferenceHandover
from robotx_safe_docking_control.usv_flatness import wrap_angle


def make_reference(name, y_offset, yaw):
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    z = np.column_stack((
        t,
        np.full_like(t, y_offset),
        np.full_like(t, yaw),
    ))
    z_dot = np.column_stack((
        np.ones_like(t),
        np.zeros_like(t),
        np.zeros_like(t),
    ))
    zeros3 = np.zeros((len(t), 3), dtype=float)
    zeros2 = np.zeros((len(t), 2), dtype=float)
    return FeasibleReference(
        name=name,
        t=t,
        z=z,
        z_dot=z_dot,
        z_ddot=zeros3.copy(),
        nu=z_dot.copy(),
        nu_dot=zeros3.copy(),
        tau=zeros3.copy(),
        thrust=zeros2.copy(),
        diagnostics={"max_abs_tau_v": 0.0},
    )


def test_handover_starts_on_old_reference_and_ends_on_new_reference():
    old_ref = make_reference("old", 0.0, 0.2)
    new_ref = make_reference("new", 10.0, 1.0)
    handover = ReferenceHandover(
        name="handover",
        previous_reference=old_ref,
        new_reference=new_ref,
        previous_elapsed_at_swap=1.0,
        handover_duration=2.0,
    )

    assert np.allclose(handover.sample(0.0)["z"], old_ref.sample(1.0)["z"])
    assert np.allclose(handover.sample(2.0)["z"], new_ref.sample(2.0)["z"])
    assert np.allclose(handover.sample(2.5)["z"], new_ref.sample(2.5)["z"])


def test_handover_yaw_uses_shortest_wrapped_difference():
    old_ref = make_reference("old", 0.0, 3.0)
    new_ref = make_reference("new", 0.0, -3.0)
    handover = ReferenceHandover(
        name="handover",
        previous_reference=old_ref,
        new_reference=new_ref,
        previous_elapsed_at_swap=0.0,
        handover_duration=2.0,
    )

    midpoint_yaw = handover.sample(1.0)["z"][2]
    expected = wrap_angle(3.0 + 0.5 * wrap_angle(-6.0))
    assert math.isclose(midpoint_yaw, expected, abs_tol=1e-12)


def test_committed_switch_tracks_old_then_new_reference():
    old_ref = make_reference("old", 0.0, 0.2)
    new_ref = make_reference("new", 10.0, 1.0)
    switch = ReferenceCommittedSwitch(
        name="switch",
        previous_reference=old_ref,
        new_reference=new_ref,
        previous_elapsed_at_swap=1.0,
        commit_duration=0.5,
    )

    assert np.allclose(switch.sample(0.0)["z"], old_ref.sample(1.0)["z"])
    assert np.allclose(switch.sample(0.49)["z"], old_ref.sample(1.49)["z"])
    assert np.allclose(switch.sample(0.5)["z"], new_ref.sample(0.0)["z"])
    assert np.allclose(switch.sample(1.0)["z"], new_ref.sample(0.5)["z"])
    assert math.isclose(switch.duration, 3.5, abs_tol=1e-12)
