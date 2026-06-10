# Exact MINCO polynomial reference for the safe-docking flat output.
#
# The trajectory is segment-major. For segment i, dimension d, and power p:
#
#   index = ((i * dimension + d) * (polynomial_order + 1)) + p
#   z_d(tau) = sum_p coefficients[index] * tau^p
#
# tau is local segment time in [0, segment_times[i]]. The flat output order is
# [x, y, psi], with psi stored unwrapped. Consumers should wrap only tracking
# errors.

std_msgs/Header header
builtin_interfaces/Time start_time
uint32 trajectory_id
string frame_id

uint8 dimension
uint8 polynomial_order
uint32 segment_count

float64[] segment_times
float64[] coefficients

float64[3] initial_z
float64[3] initial_z_dot
float64[3] initial_z_ddot
float64[3] terminal_z
float64[3] terminal_z_dot
float64[3] terminal_z_ddot

float64 max_abs_tau_v
float64 rms_tau_v
float64 velocity_violation
float64 acceleration_violation
float64 actuator_bound_violation
float64 solve_time_ms
