# Controller execution acknowledgement / feedback for online MINCO references.

std_msgs/Header header
uint32 trajectory_id

uint8 STATE_NONE=0
uint8 STATE_RECEIVED=1
uint8 STATE_EXECUTING=2
uint8 STATE_REJECTED=3
uint8 STATE_STOPPED=4

uint8 state
bool accepted
bool executing
string reason

builtin_interfaces/Time execution_start_time
float64 elapsed

float64[3] current_z
float64[3] current_z_dot
float64[3] reference_z
float64[3] reference_z_dot

float64 start_z_error
float64 start_z_dot_error
float64 tracking_position_error
float64 tracking_yaw_error
