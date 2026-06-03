from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List

import numpy as np

from .usv_flatness import UsvModelParams
from .usv_flatness import body_to_flat_acceleration
from .usv_flatness import body_to_flat_velocity
from .usv_flatness import clip_allocated_thrust
from .usv_flatness import nominal_coriolis_times_velocity
from .usv_flatness import nominal_damping
from .usv_flatness import theta_tau_nominal
from .usv_flatness import wrap_angle


@dataclass
class FeasibleReference:
    name: str
    t: np.ndarray
    z: np.ndarray
    z_dot: np.ndarray
    z_ddot: np.ndarray
    nu: np.ndarray
    nu_dot: np.ndarray
    tau: np.ndarray
    thrust: np.ndarray
    diagnostics: Dict[str, float]

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def sample(self, query_t: float):
        if query_t <= self.t[0]:
            return self._row(0)
        if query_t >= self.t[-1]:
            return self._row(len(self.t) - 1)
        index = int(np.searchsorted(self.t, query_t))
        t0 = self.t[index - 1]
        t1 = self.t[index]
        ratio = (query_t - t0) / max(t1 - t0, 1e-9)
        return {
            'z': self._interp(self.z, index, ratio, wrap_yaw=True),
            'z_dot': self._interp(self.z_dot, index, ratio),
            'z_ddot': self._interp(self.z_ddot, index, ratio),
            'nu': self._interp(self.nu, index, ratio),
            'nu_dot': self._interp(self.nu_dot, index, ratio),
            'tau': self._interp(self.tau, index, ratio),
            'thrust': self._interp(self.thrust, index, ratio),
        }

    def horizon(self, start_t: float, dt: float, count: int):
        samples = [self.sample(start_t + dt * i) for i in range(count)]
        return {
            key: np.asarray([sample[key] for sample in samples], dtype=float)
            for key in samples[0].keys()
        }

    def horizon_at_offsets(self, start_t: float, offsets):
        samples = [self.sample(start_t + float(offset)) for offset in offsets]
        return {
            key: np.asarray([sample[key] for sample in samples], dtype=float)
            for key in samples[0].keys()
        }

    def _row(self, index: int):
        return {
            'z': self.z[index],
            'z_dot': self.z_dot[index],
            'z_ddot': self.z_ddot[index],
            'nu': self.nu[index],
            'nu_dot': self.nu_dot[index],
            'tau': self.tau[index],
            'thrust': self.thrust[index],
        }

    @staticmethod
    def _interp(array, index: int, ratio: float, wrap_yaw: bool = False):
        value = array[index - 1] + ratio * (array[index] - array[index - 1])
        if wrap_yaw:
            value = np.asarray(value, dtype=float).copy()
            value[2] = wrap_angle(value[2])
        return value


def rollout_reference(name: str, initial_z, initial_nu,
                      command_fn: Callable[[float], np.ndarray],
                      duration: float, dt: float,
                      params: UsvModelParams) -> FeasibleReference:
    steps = int(np.floor(duration / dt)) + 1
    t = np.linspace(0.0, dt * (steps - 1), steps)
    z = np.zeros((steps, 3), dtype=float)
    nu = np.zeros((steps, 3), dtype=float)
    nu_dot = np.zeros((steps, 3), dtype=float)
    tau = np.zeros((steps, 3), dtype=float)
    thrust = np.zeros((steps, 2), dtype=float)
    z_dot = np.zeros((steps, 3), dtype=float)
    z_ddot = np.zeros((steps, 3), dtype=float)

    z[0] = np.asarray(initial_z, dtype=float)
    nu[0] = np.asarray(initial_nu, dtype=float)

    mass_inv = np.diag([
        1.0 / params.mass,
        1.0 / params.mass,
        1.0 / params.iz,
    ])

    for index in range(steps):
        tau_cmd = np.asarray(command_fn(float(t[index])), dtype=float)
        tau_cmd = np.array([tau_cmd[0], 0.0, tau_cmd[2]], dtype=float)
        left, right, tau_u_c, tau_r_c = clip_allocated_thrust(
            tau_cmd[0], tau_cmd[2], params)
        tau[index] = np.array([tau_u_c, 0.0, tau_r_c], dtype=float)
        thrust[index] = np.array([left, right], dtype=float)

        c_nu = nominal_coriolis_times_velocity(nu[index], params)
        d_nu = nominal_damping(nu[index], params)
        nu_dot[index] = mass_inv @ (tau[index] - c_nu - d_nu)
        z_dot[index] = body_to_flat_velocity(z[index, 2], nu[index])
        z_ddot[index] = body_to_flat_acceleration(
            z[index, 2], nu[index], nu_dot[index])

        if index == steps - 1:
            break
        nu[index + 1] = nu[index] + dt * nu_dot[index]
        z[index + 1] = z[index] + dt * z_dot[index] + 0.5 * dt * dt * z_ddot[index]
        z[index + 1, 2] = wrap_angle(z[index + 1, 2])

    enforce_terminal_rest(z, z_dot, z_ddot, nu, nu_dot, tau, thrust)
    tau_check = np.asarray([
        theta_tau_nominal(z_i, zd_i, zdd_i, params)
        for z_i, zd_i, zdd_i in zip(z, z_dot, z_ddot)
    ])
    diagnostics = reference_diagnostics(tau_check, thrust, z_dot, nu)
    return FeasibleReference(
        name=name,
        t=t,
        z=z,
        z_dot=z_dot,
        z_ddot=z_ddot,
        nu=nu,
        nu_dot=nu_dot,
        tau=tau_check,
        thrust=thrust,
        diagnostics=diagnostics)


def enforce_terminal_rest(z, z_dot, z_ddot, nu, nu_dot, tau, thrust):
    """Make the clamped terminal reference an exact pose-hold target."""
    z_dot[-1] = 0.0
    z_ddot[-1] = 0.0
    nu[-1] = 0.0
    nu_dot[-1] = 0.0
    tau[-1] = 0.0
    thrust[-1] = 0.0


def reference_diagnostics(tau, thrust, z_dot, nu):
    speed = np.linalg.norm(z_dot[:, :2], axis=1)
    return {
        'max_abs_tau_v': float(np.max(np.abs(tau[:, 1]))),
        'rms_tau_v': float(np.sqrt(np.mean(tau[:, 1]**2))),
        'max_abs_tau_u': float(np.max(np.abs(tau[:, 0]))),
        'max_abs_tau_r': float(np.max(np.abs(tau[:, 2]))),
        'max_abs_left_thrust': float(np.max(np.abs(thrust[:, 0]))),
        'max_abs_right_thrust': float(np.max(np.abs(thrust[:, 1]))),
        'max_speed': float(np.max(speed)),
        'max_yaw_rate': float(np.max(np.abs(nu[:, 2]))),
        'final_speed': float(speed[-1]),
        'final_yaw_rate': float(abs(nu[-1, 2])),
    }


def smooth_step(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    s = (t - start) / max(end - start, 1e-9)
    return s * s * (3.0 - 2.0 * s)


def make_synthetic_reference(name: str, initial_z, params: UsvModelParams,
                             dt: float = 0.2,
                             initial_nu=None) -> FeasibleReference:
    if initial_nu is None:
        initial_nu = np.zeros(3, dtype=float)
    else:
        initial_nu = np.asarray(initial_nu, dtype=float).reshape(3)
    if name == 'hold':
        duration = 8.0

        def command(_t):
            return np.zeros(3, dtype=float)

    elif name == 'straight':
        duration = 36.0

        def command(t):
            ramp_up = smooth_step(t, 1.0, 5.0)
            ramp_down = 1.0 - smooth_step(t, 20.0, 27.0)
            return np.array([70.0 * ramp_up * ramp_down, 0.0, 0.0])

    elif name == 'arc':
        duration = 50.0

        def command(t):
            ramp_up = smooth_step(t, 1.0, 8.0)
            ramp_down = 1.0 - smooth_step(t, 33.0, 41.0)
            scale = ramp_up * ramp_down
            return np.array([30.0 * scale, 0.0, 45.0 * scale])

    elif name == 'stop':
        duration = 44.0

        def command(t):
            if t < 8.0:
                return np.array([65.0 * smooth_step(t, 1.0, 5.0), 0.0, 0.0])
            if t < 14.0:
                return np.array([65.0, 0.0, 0.0])
            if t < 30.0:
                brake = (
                    smooth_step(t, 14.0, 18.0) *
                    (1.0 - smooth_step(t, 28.0, 30.0)))
                return np.array([-22.0 * brake, 0.0, 0.0])
            return np.zeros(3, dtype=float)

    elif name == 'yaw':
        duration = 26.0
        if params.min_thrust >= 0.0:
            raise ValueError('Differential yaw reference requires negative thrust.')

        def command(t):
            ramp_up = smooth_step(t, 1.0, 5.0)
            ramp_down = 1.0 - smooth_step(t, 14.0, 19.0)
            return np.array([0.0, 0.0, 180.0 * ramp_up * ramp_down])

    elif name == 'figure8':
        command_duration = 72.0
        duration = 80.0
        switch_time = 32.0

        def command(t):
            ramp = (
                smooth_step(t, 1.0, 7.0) *
                (1.0 - smooth_step(
                    t, command_duration - 7.0, command_duration)))
            turn_sign = 1.0 - 2.0 * smooth_step(
                t, switch_time - 3.0, switch_time + 3.0)
            return np.array([
                45.0 * ramp,
                0.0,
                80.0 * turn_sign * ramp,
            ])

    elif name == 'spiral':
        command_duration = 70.0
        duration = 78.0

        def command(t):
            ramp = (
                smooth_step(t, 1.0, 7.0) *
                (1.0 - smooth_step(
                    t, command_duration - 8.0, command_duration)))
            progress = min(max((t - 7.0) / 55.0, 0.0), 1.0)
            surge = 35.0 + (70.0 - 35.0) * progress
            return np.array([
                surge * ramp,
                0.0,
                75.0 * ramp,
            ])

    else:
        raise ValueError(f'Unknown synthetic reference scenario: {name}')

    return rollout_reference(name, initial_z, initial_nu, command,
                             duration, dt, params)


def available_reference_names(params: UsvModelParams) -> List[str]:
    names = ['hold', 'straight', 'arc', 'stop', 'figure8', 'spiral']
    if params.min_thrust < 0.0:
        names.append('yaw')
    return names
