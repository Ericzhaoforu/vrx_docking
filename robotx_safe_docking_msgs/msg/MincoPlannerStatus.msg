# Online MINCO replanner status and diagnostics.

std_msgs/Header header
uint32 trajectory_id

bool should_plan
uint8 should_plan_reason
uint8 candidate_status
string failure_reason

uint32 candidate_count
uint32 accepted_count
uint32 rejected_count

float64 time_since_last_plan
float64 progress
float64 yaw_progress
float64 goal_distance
float64 goal_yaw_error

float64 max_abs_tau_v
float64 rms_tau_v
float64 velocity_violation
float64 acceleration_violation
float64 actuator_bound_violation
float64 solve_time_ms
float64 continuity_residual
