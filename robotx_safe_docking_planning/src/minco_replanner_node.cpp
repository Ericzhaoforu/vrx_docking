#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nav_msgs/msg/odometry.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <rclcpp/rclcpp.hpp>
#include <robotx_safe_docking_msgs/msg/minco_execution_status.hpp>
#include <robotx_safe_docking_msgs/msg/minco_planner_status.hpp>
#include <robotx_safe_docking_msgs/msg/minco_trajectory.hpp>

namespace py = pybind11;

namespace {

constexpr uint8_t kReasonNone = 0;
constexpr uint8_t kReasonNoReference = 3;
constexpr uint8_t kReasonCheckInterval = 4;
constexpr uint8_t kReasonNearGoal = 5;
constexpr uint8_t kReasonProgressGate = 6;
constexpr uint8_t kReasonProgressNotMet = 7;
constexpr uint8_t kReasonWaitingForControllerAck = 8;
constexpr uint8_t kReasonTrackingErrorTooLarge = 9;

constexpr uint8_t kCandidateNone = 0;
constexpr uint8_t kCandidateAccepted = 1;
constexpr uint8_t kCandidateRejected = 2;

struct Pva
{
  Eigen::Vector3d z = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_ddot = Eigen::Vector3d::Zero();
};

struct Diagnostics
{
  double max_abs_tau_v = 0.0;
  double rms_tau_v = 0.0;
  double velocity_violation = 0.0;
  double acceleration_violation = 0.0;
  double actuator_bound_violation = 0.0;
  double penalty_total = 0.0;
  double pva_residual = 0.0;
  double continuity_residual = 0.0;
  double solve_time_ms = 0.0;
  double wall_time_ms = 0.0;
  double optimizer_iterations = 0.0;
  double frontend_selected_depth = 0.0;
  double frontend_expansion_count = 0.0;
};

struct Candidate
{
  std::vector<double> coefficients;
  std::vector<double> segment_times;
  Pva initial;
  Pva terminal;
  Diagnostics diagnostics;
};

double wrap_angle_local(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

builtin_interfaces::msg::Time sec_to_stamp(double time_sec)
{
  builtin_interfaces::msg::Time stamp;
  const double clamped = std::max(time_sec, 0.0);
  stamp.sec = static_cast<int32_t>(std::floor(clamped));
  stamp.nanosec = static_cast<uint32_t>(
    std::round((clamped - static_cast<double>(stamp.sec)) * 1e9));
  if (stamp.nanosec >= 1000000000U) {
    stamp.sec += 1;
    stamp.nanosec -= 1000000000U;
  }
  return stamp;
}

double stamp_to_sec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) +
    static_cast<double>(stamp.nanosec) * 1e-9;
}

py::array_t<double> vec3_to_numpy(const Eigen::Vector3d & value)
{
  py::array_t<double> array({3});
  auto info = array.mutable_unchecked<1>();
  for (int i = 0; i < 3; ++i) {
    info(i) = value(i);
  }
  return array;
}

py::array_t<double> pva_to_numpy(const Pva & pva)
{
  py::array_t<double> array({3, 3});
  auto info = array.mutable_unchecked<2>();
  for (int dim = 0; dim < 3; ++dim) {
    info(0, dim) = pva.z(dim);
    info(1, dim) = pva.z_dot(dim);
    info(2, dim) = pva.z_ddot(dim);
  }
  return array;
}

py::array_t<double> vector_to_numpy(const std::vector<double> & values)
{
  py::array_t<double> array({static_cast<py::ssize_t>(values.size())});
  auto info = array.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
    info(i) = values[static_cast<std::size_t>(i)];
  }
  return array;
}

Eigen::VectorXd py_to_vector(const py::handle & object, const std::string & name)
{
  auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
  if (!array) {
    throw std::runtime_error(name + " is not a float array.");
  }
  const auto info = array.request();
  if (info.ndim != 1) {
    throw std::runtime_error(name + " must be a one-dimensional array.");
  }
  Eigen::VectorXd out(static_cast<int>(info.shape[0]));
  const auto * data = static_cast<const double *>(info.ptr);
  for (int i = 0; i < out.size(); ++i) {
    out(i) = data[i];
  }
  return out;
}

Eigen::MatrixXd py_to_matrix(const py::handle & object, const std::string & name)
{
  auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
  if (!array) {
    throw std::runtime_error(name + " is not a float array.");
  }
  const auto info = array.request();
  if (info.ndim != 2) {
    throw std::runtime_error(name + " must be a two-dimensional array.");
  }
  Eigen::MatrixXd out(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]));
  const auto * data = static_cast<const double *>(info.ptr);
  const auto row_stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(double));
  const auto col_stride = info.strides[1] / static_cast<py::ssize_t>(sizeof(double));
  for (int row = 0; row < out.rows(); ++row) {
    for (int col = 0; col < out.cols(); ++col) {
      out(row, col) = data[row * row_stride + col * col_stride];
    }
  }
  return out;
}

Eigen::Vector3d py_to_vec3(const py::handle & object, const std::string & name)
{
  const Eigen::VectorXd value = py_to_vector(object, name);
  if (value.size() != 3) {
    throw std::runtime_error(name + " must have length 3.");
  }
  return value;
}

double dict_double(const py::dict & dict, const char * key, double fallback = 0.0)
{
  if (!dict.contains(key)) {
    return fallback;
  }
  return py::cast<double>(dict[key]);
}

bool dict_bool(const py::dict & dict, const char * key, bool fallback = false)
{
  if (!dict.contains(key)) {
    return fallback;
  }
  return py::cast<bool>(dict[key]);
}

std::vector<double> eigen_to_std(const Eigen::Vector3d & value)
{
  return {value(0), value(1), value(2)};
}

std::vector<double> matrix_coefficients_to_message(const Eigen::MatrixXd & coeffs)
{
  if (coeffs.cols() != 3 || coeffs.rows() % 6 != 0) {
    throw std::runtime_error("MINCO coefficient matrix has unexpected shape.");
  }
  const int pieces = coeffs.rows() / 6;
  std::vector<double> flattened;
  flattened.reserve(static_cast<std::size_t>(pieces * 3 * 6));
  for (int piece = 0; piece < pieces; ++piece) {
    for (int dim = 0; dim < 3; ++dim) {
      for (int power = 0; power < 6; ++power) {
        flattened.push_back(coeffs(piece * 6 + power, dim));
      }
    }
  }
  return flattened;
}

Eigen::Vector3d evaluate_piece(
  const std::vector<double> & coefficients,
  int piece,
  double local_time,
  int derivative)
{
  Eigen::Vector3d value = Eigen::Vector3d::Zero();
  const int order = 5;
  for (int dim = 0; dim < 3; ++dim) {
    for (int power = derivative; power <= order; ++power) {
      double scale = 1.0;
      for (int factor = 0; factor < derivative; ++factor) {
        scale *= static_cast<double>(power - factor);
      }
      const std::size_t index = static_cast<std::size_t>(
        ((piece * 3 + dim) * (order + 1)) + power);
      value(dim) += scale * coefficients[index] *
        std::pow(local_time, static_cast<double>(power - derivative));
    }
  }
  return value;
}

}  // namespace

class MincoReplannerNode : public rclcpp::Node
{
public:
  MincoReplannerNode()
  : Node("minco_replanner_node")
  {
    declare_parameter("odom_topic", "/safe_docking/odometry");
    declare_parameter("trajectory_topic", "/safe_docking/minco_trajectory");
    declare_parameter("status_topic", "/safe_docking/minco_planner_status");
    declare_parameter(
      "execution_status_topic", "/safe_docking/minco_execution_status");
    declare_parameter("frame_id", "map");
    declare_parameter("planner_rate_hz", 1.0);
    declare_parameter("check_rate_hz", 1.0);
    declare_parameter("progress_distance", 0.5);
    declare_parameter("progress_yaw", 0.12);
    declare_parameter("no_replan_goal_distance", 0.5);
    declare_parameter("no_replan_goal_yaw", 0.08);
    declare_parameter("max_replan_tracking_position_error", 0.75);
    declare_parameter("max_replan_tracking_yaw_error", 0.70);
    declare_parameter("global_goal", std::vector<double>{3.0, 6.0, 2.4});

    declare_parameter("mass", 180.0);
    declare_parameter("iz", 446.0);
    declare_parameter("du", 100.0);
    declare_parameter("duu", 150.0);
    declare_parameter("dv", 100.0);
    declare_parameter("dvv", 100.0);
    declare_parameter("dr", 800.0);
    declare_parameter("drr", 800.0);
    declare_parameter("thruster_half_spacing", 1.027135);
    declare_parameter("min_thrust", -100.0);
    declare_parameter("max_thrust", 100.0);

    declare_parameter("frontend_primitive_duration", 2.0);
    declare_parameter("frontend_primitive_dt", 0.25);
    declare_parameter("frontend_check_num", 5);
    declare_parameter("frontend_search_depth", 24);
    declare_parameter("frontend_max_expansions", 2000);
    declare_parameter("frontend_control_discretization", 1);
    declare_parameter("frontend_sample_thrust_limit", 60.0);
    declare_parameter("frontend_minco_piece_count_min", 3);
    declare_parameter("frontend_minco_piece_count_max", 8);
    declare_parameter("frontend_flat_accel_bounds", std::vector<double>{0.5, 0.5, 0.12});
    declare_parameter("frontend_flat_velocity_bounds", std::vector<double>{2.0, 2.0, 0.4});
    declare_parameter("frontend_time_weight", 1.0);
    declare_parameter("frontend_heuristic_weight", 3.0);
    declare_parameter("frontend_tau_v_bar", 12.0);
    declare_parameter("frontend_max_local_goal_distance", 5.0);
    declare_parameter("frontend_local_goal_speed", 0.45);
    declare_parameter("frontend_final_goal_radius", 0.8);
    declare_parameter("frontend_goal_tolerance", 0.15);
    declare_parameter("frontend_yaw_tolerance", 0.12);
    declare_parameter("frontend_obstacle_margin", 0.5);
    declare_parameter("frontend_grid_resolution_xy", 0.25);
    declare_parameter("frontend_grid_resolution_yaw", 0.25);
    declare_parameter("frontend_grid_resolution_vxy", 0.25);
    declare_parameter("frontend_grid_resolution_yaw_rate", 0.05);
    declare_parameter("frontend_analytic_expansion", true);

    declare_parameter("minco_dt", 0.1);
    declare_parameter("minco_fixed_total_time", true);
    declare_parameter("minco_reference_speed", 0.45);
    declare_parameter("minco_time_dilation", 1.3);
    declare_parameter("minco_min_segment_time", 0.2);
    declare_parameter("minco_smooth_weight", 1.0);
    declare_parameter("minco_time_weight", 0.0);
    declare_parameter("minco_max_iterations", 80);
    declare_parameter("minco_gradient_tolerance", 1e-6);
    declare_parameter("minco_function_tolerance", 1e-9);
    declare_parameter("minco_spatial_margin", 2.0);
    declare_parameter("minco_yaw_margin", 1.0);
    declare_parameter("minco_theta_bound", 6.0);
    declare_parameter("minco_tau_v_bar", 12.0);
    declare_parameter("minco_velocity_bounds", std::vector<double>{2.0, 0.5, 0.6});
    declare_parameter("minco_acceleration_bounds", std::vector<double>{0.9, 0.5, 0.8});
    declare_parameter("minco_lambda_tau_v", 1.0);
    declare_parameter("minco_lambda_velocity", 1.0);
    declare_parameter("minco_lambda_acceleration", 1.0);
    declare_parameter("minco_lambda_actuator", 10.0);
    declare_parameter("minco_quadrature_order", 8);
    declare_parameter("minco_penalty_mu", 20.0);

    declare_parameter("accept_tau_v_margin", 0.5);
    declare_parameter("accept_velocity_violation", 1e-3);
    declare_parameter("accept_acceleration_violation", 1e-3);
    declare_parameter("accept_actuator_violation", 1e-3);
    declare_parameter("accept_pva_residual", 1e-5);
    declare_parameter("accept_continuity_residual", 1e-5);
    declare_parameter("accept_penalty_total", 1e7);

    loadParameters();

    {
      py::gil_scoped_acquire gil;
      minco_module_ = py::module_::import("robotx_minco_cpp._robotx_safe_docking_minco_cpp");
    }

    auto trajectory_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    trajectory_pub_ = create_publisher<robotx_safe_docking_msgs::msg::MincoTrajectory>(
      get_parameter("trajectory_topic").as_string(), trajectory_qos);
    status_pub_ = create_publisher<robotx_safe_docking_msgs::msg::MincoPlannerStatus>(
      get_parameter("status_topic").as_string(), 10);
    execution_status_sub_ =
      create_subscription<robotx_safe_docking_msgs::msg::MincoExecutionStatus>(
      get_parameter("execution_status_topic").as_string(), 10,
      [this](const robotx_safe_docking_msgs::msg::MincoExecutionStatus::SharedPtr msg) {
        onExecutionStatus(msg);
      });
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), 10,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        latest_odom_ = msg;
      });
    timer_ = create_wall_timer(
      std::chrono::duration<double>(planner_period_),
      [this]() { onTimer(); });

    RCLCPP_INFO(
      get_logger(),
      "Strict GCOPTER/MINCO replanner loaded. goal=(%.2f, %.2f, %.2f), "
      "planner_rate=%.2f Hz, lattice thrust sample limit=%.1f N, tau_v_bar=%.2f N.",
      goal_(0), goal_(1), goal_(2), 1.0 / planner_period_,
      frontend_sample_thrust_limit_, minco_tau_v_bar_);
  }

private:
  void loadParameters()
  {
    frame_id_ = get_parameter("frame_id").as_string();
    planner_period_ = 1.0 / std::max(get_parameter("planner_rate_hz").as_double(), 1e-3);
    check_interval_ = 1.0 / std::max(get_parameter("check_rate_hz").as_double(), 1e-3);
    progress_distance_ = get_parameter("progress_distance").as_double();
    progress_yaw_ = get_parameter("progress_yaw").as_double();
    no_replan_goal_distance_ = get_parameter("no_replan_goal_distance").as_double();
    no_replan_goal_yaw_ = get_parameter("no_replan_goal_yaw").as_double();
    max_replan_tracking_position_error_ =
      get_parameter("max_replan_tracking_position_error").as_double();
    max_replan_tracking_yaw_error_ =
      get_parameter("max_replan_tracking_yaw_error").as_double();

    const auto goal_values = get_parameter("global_goal").as_double_array();
    if (goal_values.size() != 3) {
      throw std::runtime_error("global_goal must contain [x, y, psi].");
    }
    goal_ = Eigen::Vector3d(goal_values[0], goal_values[1], goal_values[2]);

    mass_ = get_parameter("mass").as_double();
    iz_ = get_parameter("iz").as_double();
    du_ = get_parameter("du").as_double();
    duu_ = get_parameter("duu").as_double();
    dv_ = get_parameter("dv").as_double();
    dvv_ = get_parameter("dvv").as_double();
    dr_ = get_parameter("dr").as_double();
    drr_ = get_parameter("drr").as_double();
    thruster_half_spacing_ = get_parameter("thruster_half_spacing").as_double();
    min_thrust_ = get_parameter("min_thrust").as_double();
    max_thrust_ = get_parameter("max_thrust").as_double();

    frontend_sample_thrust_limit_ =
      get_parameter("frontend_sample_thrust_limit").as_double();
    minco_dt_ = get_parameter("minco_dt").as_double();
    minco_tau_v_bar_ = get_parameter("minco_tau_v_bar").as_double();
    accept_tau_v_margin_ = get_parameter("accept_tau_v_margin").as_double();
    accept_velocity_violation_ = get_parameter("accept_velocity_violation").as_double();
    accept_acceleration_violation_ = get_parameter("accept_acceleration_violation").as_double();
    accept_actuator_violation_ = get_parameter("accept_actuator_violation").as_double();
    accept_pva_residual_ = get_parameter("accept_pva_residual").as_double();
    accept_continuity_residual_ = get_parameter("accept_continuity_residual").as_double();
    accept_penalty_total_ = get_parameter("accept_penalty_total").as_double();
  }

  void onTimer()
  {
    const double now = get_clock()->now().seconds();
    uint8_t reason = kReasonNone;
    uint8_t candidate_status = kCandidateNone;
    std::string failure_reason;
    const bool should_plan = shouldPlan(now, reason);

    if (should_plan) {
      Candidate candidate;
      const bool accepted = buildStrictCandidate(now, candidate, failure_reason);
      candidate_status = accepted ? kCandidateAccepted : kCandidateRejected;
      ++candidate_count_;
      if (accepted) {
        publishCandidate(now, candidate);
        last_plan_time_ = now;
      } else {
        ++rejected_count_;
      }
    }

    publishStatus(now, should_plan, reason, candidate_status, failure_reason);
  }

  bool shouldPlan(double now, uint8_t & reason)
  {
    if (!latest_odom_) {
      reason = kReasonNone;
      return false;
    }
    if (has_pending_) {
      reason = kReasonWaitingForControllerAck;
      return false;
    }
    if (!has_active_) {
      reason = kReasonNoReference;
      return true;
    }
    if (!controller_executing_active_) {
      reason = kReasonWaitingForControllerAck;
      return false;
    }
    if (controller_tracking_position_error_ > max_replan_tracking_position_error_ ||
        controller_tracking_yaw_error_ > max_replan_tracking_yaw_error_)
    {
      reason = kReasonTrackingErrorTooLarge;
      return false;
    }
    if (last_plan_time_ >= 0.0 && now - last_plan_time_ < check_interval_) {
      reason = kReasonCheckInterval;
      return false;
    }
    const double elapsed = std::max(now - active_start_time_, 0.0);
    const Pva current = sampleActive(elapsed);
    last_goal_distance_ = (goal_.head<2>() - current.z.head<2>()).norm();
    last_goal_yaw_error_ = std::abs(wrap_angle_local(goal_(2) - current.z(2)));
    if (last_goal_distance_ <= no_replan_goal_distance_ &&
        last_goal_yaw_error_ <= no_replan_goal_yaw_)
    {
      reason = kReasonNearGoal;
      return false;
    }
    const Pva start = sampleActive(0.0);
    last_progress_ = (current.z.head<2>() - start.z.head<2>()).norm();
    last_yaw_progress_ = std::abs(wrap_angle_local(current.z(2) - start.z(2)));
    if (last_progress_ >= progress_distance_ || last_yaw_progress_ >= progress_yaw_) {
      reason = kReasonProgressGate;
      return true;
    }
    reason = kReasonProgressNotMet;
    return false;
  }

  bool buildStrictCandidate(double now, Candidate & candidate, std::string & failure_reason)
  {
    Pva start;
    if (has_active_) {
      start = sampleActive(std::max(now - active_start_time_, 0.0));
    } else {
      start = stateFromOdom();
    }
    candidate.initial = start;

    try {
      py::gil_scoped_acquire gil;
      py::dict lattice = minco_module_.attr("plan_lattice_terminal")(
        vec3_to_numpy(start.z), vec3_to_numpy(start.z_dot), paramsDict(), frontendDict());
      const py::dict lattice_diag = py::cast<py::dict>(lattice["diagnostics"]);
      if (!py::cast<bool>(lattice["accepted"])) {
        last_candidate_diagnostics_ = diagnosticsFromDict(lattice_diag);
        failure_reason = "lattice rejected: expansions=" +
          std::to_string(static_cast<int>(last_candidate_diagnostics_.frontend_expansion_count));
        RCLCPP_WARN(get_logger(), "%s", failure_reason.c_str());
        return false;
      }

      if (lattice["initial_inPs"].is_none() || lattice["initial_ts"].is_none()) {
        failure_reason = "lattice did not provide primitive-chain initial q/T";
        RCLCPP_WARN(get_logger(), "%s", failure_reason.c_str());
        return false;
      }
      const Eigen::VectorXd initial_ts = py_to_vector(lattice["initial_ts"], "initial_ts");
      if (initial_ts.size() < 1) {
        failure_reason = "lattice initial_ts is empty";
        RCLCPP_WARN(get_logger(), "%s", failure_reason.c_str());
        return false;
      }
      const int piece_count = initial_ts.size();
      Eigen::MatrixXd initial_inPs = py_to_matrix(lattice["initial_inPs"], "initial_inPs");
      if (initial_inPs.rows() != piece_count - 1 || initial_inPs.cols() != 3) {
        failure_reason = "lattice initial_inPs shape does not match selected piece count";
        RCLCPP_WARN(get_logger(), "%s", failure_reason.c_str());
        return false;
      }

      Pva terminal;
      terminal.z = py_to_vec3(lattice["terminal_z"], "terminal_z");
      terminal.z_dot = py_to_vec3(lattice["terminal_z_dot"], "terminal_z_dot");
      terminal.z_ddot = py_to_vec3(lattice["terminal_z_ddot"], "terminal_z_ddot");
      terminal.z(2) = start.z(2) + wrap_angle_local(terminal.z(2) - start.z(2));
      candidate.terminal = terminal;

      const double fixed_total_time = initial_ts.sum();
      py::dict optimizer = optimizerDict(piece_count, fixed_total_time);
      py::dict result = minco_module_.attr("plan_minco_reference")(
        pva_to_numpy(start), pva_to_numpy(terminal), paramsDict(), optimizer,
        penaltyDict(), minco_dt_, lattice["initial_inPs"], lattice["initial_ts"]);
      const py::dict minco_diag = py::cast<py::dict>(result["diagnostics"]);
      Diagnostics diagnostics = diagnosticsFromDict(minco_diag);
      diagnostics.frontend_selected_depth =
        dict_double(lattice_diag, "frontend_selected_depth", 0.0);
      diagnostics.frontend_expansion_count =
        dict_double(lattice_diag, "frontend_expansion_count", 0.0);
      last_candidate_diagnostics_ = diagnostics;

      if (!py::cast<bool>(result["success"])) {
        failure_reason = "MINCO optimizer rejected: " + py::cast<std::string>(result["message"]);
        RCLCPP_WARN(get_logger(), "%s", failure_reason.c_str());
        return false;
      }
      if (!passesAcceptanceGates(diagnostics, failure_reason)) {
        RCLCPP_WARN(
          get_logger(),
          "Rejected strict MINCO candidate: %s (max|tau_v|=%.3f, vel_violation=%.3g, "
          "acc_violation=%.3g, actuator_violation=%.3g, pva=%.3g, continuity=%.3g)",
          failure_reason.c_str(), diagnostics.max_abs_tau_v, diagnostics.velocity_violation,
          diagnostics.acceleration_violation, diagnostics.actuator_bound_violation,
          diagnostics.pva_residual, diagnostics.continuity_residual);
        return false;
      }

      const Eigen::VectorXd ts = py_to_vector(result["ts"], "ts");
      const Eigen::MatrixXd coeffs = py_to_matrix(result["coefficients"], "coefficients");
      candidate.segment_times.assign(ts.data(), ts.data() + ts.size());
      candidate.coefficients = matrix_coefficients_to_message(coeffs);
      candidate.diagnostics = diagnostics;
      RCLCPP_INFO(
        get_logger(),
        "Accepted strict MINCO candidate: pieces=%d duration=%.2f s max|tau_v|=%.3f "
        "rms_tau_v=%.3f solve=%.2f ms depth=%.0f expansions=%.0f",
        piece_count, fixed_total_time, diagnostics.max_abs_tau_v, diagnostics.rms_tau_v,
        diagnostics.solve_time_ms, diagnostics.frontend_selected_depth,
        diagnostics.frontend_expansion_count);
      return true;
    } catch (const std::exception & error) {
      failure_reason = std::string("strict MINCO exception: ") + error.what();
      RCLCPP_WARN(get_logger(), "%s", failure_reason.c_str());
      return false;
    }
  }

  bool passesAcceptanceGates(
    const Diagnostics & diagnostics,
    std::string & failure_reason) const
  {
    if (diagnostics.pva_residual > accept_pva_residual_) {
      failure_reason = "PVA residual too large";
      return false;
    }
    if (diagnostics.continuity_residual > accept_continuity_residual_) {
      failure_reason = "continuity residual too large";
      return false;
    }
    if (diagnostics.max_abs_tau_v > minco_tau_v_bar_ + accept_tau_v_margin_) {
      failure_reason = "tau_v diagnostic bound exceeded";
      return false;
    }
    if (diagnostics.velocity_violation > accept_velocity_violation_) {
      failure_reason = "velocity bound violation";
      return false;
    }
    if (diagnostics.acceleration_violation > accept_acceleration_violation_) {
      failure_reason = "acceleration bound violation";
      return false;
    }
    if (diagnostics.actuator_bound_violation > accept_actuator_violation_) {
      failure_reason = "actuator bound violation";
      return false;
    }
    if (diagnostics.penalty_total > accept_penalty_total_) {
      failure_reason = "dense penalty total too large";
      return false;
    }
    return true;
  }

  void publishCandidate(double now, const Candidate & candidate)
  {
    ++trajectory_id_;
    pending_trajectory_id_ = trajectory_id_;
    pending_candidate_ = candidate;
    pending_publish_time_ = now;
    has_pending_ = true;

    auto msg = robotx_safe_docking_msgs::msg::MincoTrajectory();
    msg.header.stamp = sec_to_stamp(get_clock()->now().seconds());
    msg.header.frame_id = frame_id_;
    msg.start_time = sec_to_stamp(now);
    msg.trajectory_id = trajectory_id_;
    msg.frame_id = frame_id_;
    msg.dimension = 3;
    msg.polynomial_order = 5;
    msg.segment_count = static_cast<uint32_t>(candidate.segment_times.size());
    msg.segment_times = candidate.segment_times;
    msg.coefficients = candidate.coefficients;
    for (int dim = 0; dim < 3; ++dim) {
      msg.initial_z[dim] = candidate.initial.z(dim);
      msg.initial_z_dot[dim] = candidate.initial.z_dot(dim);
      msg.initial_z_ddot[dim] = candidate.initial.z_ddot(dim);
      msg.terminal_z[dim] = candidate.terminal.z(dim);
      msg.terminal_z_dot[dim] = candidate.terminal.z_dot(dim);
      msg.terminal_z_ddot[dim] = candidate.terminal.z_ddot(dim);
    }
    msg.max_abs_tau_v = candidate.diagnostics.max_abs_tau_v;
    msg.rms_tau_v = candidate.diagnostics.rms_tau_v;
    msg.velocity_violation = candidate.diagnostics.velocity_violation;
    msg.acceleration_violation = candidate.diagnostics.acceleration_violation;
    msg.actuator_bound_violation = candidate.diagnostics.actuator_bound_violation;
    msg.solve_time_ms = candidate.diagnostics.solve_time_ms;
    trajectory_pub_->publish(msg);
    RCLCPP_INFO(
      get_logger(),
      "Published pending MINCO candidate id=%u and waiting for controller ACK.",
      pending_trajectory_id_);
  }

  void onExecutionStatus(
    const robotx_safe_docking_msgs::msg::MincoExecutionStatus::SharedPtr msg)
  {
    const double now = get_clock()->now().seconds();
    latest_execution_status_time_ = now;

    if (has_active_ && msg->trajectory_id == active_trajectory_id_) {
      controller_executing_active_ = msg->executing;
      controller_tracking_position_error_ = msg->tracking_position_error;
      controller_tracking_yaw_error_ = msg->tracking_yaw_error;
      return;
    }

    if (!has_pending_ || msg->trajectory_id != pending_trajectory_id_) {
      return;
    }

    if (msg->accepted && msg->executing) {
      const double execution_start_time = stamp_to_sec(msg->execution_start_time);
      activatePending(execution_start_time);
      RCLCPP_INFO(
        get_logger(),
        "Controller ACK activated MINCO trajectory id=%u at sim time %.3f "
        "(start error %.3f m, tracking error %.3f m).",
        active_trajectory_id_, active_start_time_, msg->start_z_error,
        msg->tracking_position_error);
      return;
    }

    if (!msg->accepted ||
        msg->state == robotx_safe_docking_msgs::msg::MincoExecutionStatus::STATE_REJECTED)
    {
      RCLCPP_WARN(
        get_logger(),
        "Controller rejected pending MINCO trajectory id=%u: %s",
        pending_trajectory_id_, msg->reason.c_str());
      has_pending_ = false;
      pending_candidate_ = Candidate();
      ++rejected_count_;
    }
  }

  void activatePending(double execution_start_time)
  {
    active_trajectory_id_ = pending_trajectory_id_;
    active_coefficients_ = pending_candidate_.coefficients;
    active_segment_times_ = pending_candidate_.segment_times;
    active_start_time_ = execution_start_time;
    active_duration_ = 0.0;
    for (const double segment_time : active_segment_times_) {
      active_duration_ += segment_time;
    }
    active_diagnostics_ = pending_candidate_.diagnostics;
    has_active_ = true;
    has_pending_ = false;
    pending_candidate_ = Candidate();
    controller_executing_active_ = true;
    controller_tracking_position_error_ = 0.0;
    controller_tracking_yaw_error_ = 0.0;
    ++accepted_count_;
  }

  Pva stateFromOdom() const
  {
    Pva state;
    state.z = Eigen::Vector3d(
      latest_odom_->pose.pose.position.x,
      latest_odom_->pose.pose.position.y,
      yaw_from_quaternion(latest_odom_->pose.pose.orientation));
    state.z_dot = Eigen::Vector3d(
      latest_odom_->twist.twist.linear.x,
      latest_odom_->twist.twist.linear.y,
      latest_odom_->twist.twist.angular.z);
    state.z_ddot = Eigen::Vector3d::Zero();
    return state;
  }

  Pva sampleActive(double elapsed) const
  {
    const double clamped_elapsed = std::min(std::max(elapsed, 0.0), active_duration_);
    double remaining = clamped_elapsed;
    int piece = 0;
    for (; piece < static_cast<int>(active_segment_times_.size()); ++piece) {
      if (remaining <= active_segment_times_[piece] ||
          piece == static_cast<int>(active_segment_times_.size()) - 1)
      {
        break;
      }
      remaining -= active_segment_times_[piece];
    }
    const double local_time = std::clamp(remaining, 0.0, active_segment_times_[piece]);
    Pva sample;
    sample.z = evaluate_piece(active_coefficients_, piece, local_time, 0);
    sample.z_dot = evaluate_piece(active_coefficients_, piece, local_time, 1);
    sample.z_ddot = evaluate_piece(active_coefficients_, piece, local_time, 2);
    return sample;
  }

  py::dict paramsDict() const
  {
    py::dict dict;
    dict["mass"] = mass_;
    dict["iz"] = iz_;
    dict["du"] = du_;
    dict["duu"] = duu_;
    dict["dv"] = dv_;
    dict["dvv"] = dvv_;
    dict["dr"] = dr_;
    dict["drr"] = drr_;
    dict["thruster_half_spacing"] = thruster_half_spacing_;
    dict["min_thrust"] = min_thrust_;
    dict["max_thrust"] = max_thrust_;
    return dict;
  }

  py::dict frontendDict() const
  {
    py::dict dict;
    dict["goal_z"] = eigen_to_std(goal_);
    dict["primitive_duration"] = get_parameter("frontend_primitive_duration").as_double();
    dict["primitive_dt"] = get_parameter("frontend_primitive_dt").as_double();
    dict["check_num"] = static_cast<int>(get_parameter("frontend_check_num").as_int());
    dict["search_depth"] = static_cast<int>(get_parameter("frontend_search_depth").as_int());
    dict["max_expansions"] = static_cast<int>(get_parameter("frontend_max_expansions").as_int());
    dict["control_discretization"] =
      static_cast<int>(get_parameter("frontend_control_discretization").as_int());
    dict["sample_thrust_limit"] = frontend_sample_thrust_limit_;
    dict["minco_piece_count_min"] =
      static_cast<int>(get_parameter("frontend_minco_piece_count_min").as_int());
    dict["minco_piece_count_max"] =
      static_cast<int>(get_parameter("frontend_minco_piece_count_max").as_int());
    dict["flat_accel_bounds"] = doubleArrayParameter("frontend_flat_accel_bounds");
    dict["flat_velocity_bounds"] = doubleArrayParameter("frontend_flat_velocity_bounds");
    dict["time_weight"] = get_parameter("frontend_time_weight").as_double();
    dict["heuristic_weight"] = get_parameter("frontend_heuristic_weight").as_double();
    dict["tau_v_bar"] = get_parameter("frontend_tau_v_bar").as_double();
    dict["max_local_goal_distance"] =
      get_parameter("frontend_max_local_goal_distance").as_double();
    dict["local_goal_speed"] = get_parameter("frontend_local_goal_speed").as_double();
    dict["final_goal_radius"] = get_parameter("frontend_final_goal_radius").as_double();
    dict["goal_tolerance"] = get_parameter("frontend_goal_tolerance").as_double();
    dict["yaw_tolerance"] = get_parameter("frontend_yaw_tolerance").as_double();
    dict["obstacle_margin"] = get_parameter("frontend_obstacle_margin").as_double();
    dict["grid_resolution_xy"] = get_parameter("frontend_grid_resolution_xy").as_double();
    dict["grid_resolution_yaw"] = get_parameter("frontend_grid_resolution_yaw").as_double();
    dict["grid_resolution_vxy"] = get_parameter("frontend_grid_resolution_vxy").as_double();
    dict["grid_resolution_yaw_rate"] =
      get_parameter("frontend_grid_resolution_yaw_rate").as_double();
    dict["analytic_expansion"] = get_parameter("frontend_analytic_expansion").as_bool();
    return dict;
  }

  py::dict optimizerDict(int piece_count, double fixed_total_time) const
  {
    py::dict dict;
    dict["piece_count"] = piece_count;
    dict["fixed_total_time_enabled"] = get_parameter("minco_fixed_total_time").as_bool();
    dict["fixed_total_time"] = fixed_total_time;
    dict["reference_speed"] = get_parameter("minco_reference_speed").as_double();
    dict["time_dilation"] = get_parameter("minco_time_dilation").as_double();
    dict["min_segment_time"] = get_parameter("minco_min_segment_time").as_double();
    dict["smooth_weight"] = get_parameter("minco_smooth_weight").as_double();
    dict["time_weight"] = get_parameter("minco_time_weight").as_double();
    dict["max_iterations"] = static_cast<int>(get_parameter("minco_max_iterations").as_int());
    dict["gradient_tolerance"] = get_parameter("minco_gradient_tolerance").as_double();
    dict["function_tolerance"] = get_parameter("minco_function_tolerance").as_double();
    dict["spatial_margin"] = get_parameter("minco_spatial_margin").as_double();
    dict["yaw_margin"] = get_parameter("minco_yaw_margin").as_double();
    dict["theta_bound"] = get_parameter("minco_theta_bound").as_double();
    return dict;
  }

  py::dict penaltyDict() const
  {
    py::dict dict;
    dict["tau_v_bar"] = minco_tau_v_bar_;
    dict["velocity_bounds"] = doubleArrayParameter("minco_velocity_bounds");
    dict["acceleration_bounds"] = doubleArrayParameter("minco_acceleration_bounds");
    dict["lambda_tau_v"] = get_parameter("minco_lambda_tau_v").as_double();
    dict["lambda_velocity"] = get_parameter("minco_lambda_velocity").as_double();
    dict["lambda_acceleration"] = get_parameter("minco_lambda_acceleration").as_double();
    dict["lambda_actuator"] = get_parameter("minco_lambda_actuator").as_double();
    dict["quadrature_order"] =
      static_cast<int>(get_parameter("minco_quadrature_order").as_int());
    dict["penalty_mu"] = get_parameter("minco_penalty_mu").as_double();
    return dict;
  }

  std::vector<double> doubleArrayParameter(const std::string & name) const
  {
    const auto values = get_parameter(name).as_double_array();
    return std::vector<double>(values.begin(), values.end());
  }

  Diagnostics diagnosticsFromDict(const py::dict & dict) const
  {
    Diagnostics diagnostics;
    diagnostics.max_abs_tau_v = dict_double(dict, "max_abs_tau_v", 0.0);
    diagnostics.rms_tau_v = dict_double(dict, "rms_tau_v", 0.0);
    diagnostics.velocity_violation = dict_double(dict, "velocity_violation", 0.0);
    diagnostics.acceleration_violation = dict_double(dict, "acceleration_violation", 0.0);
    diagnostics.actuator_bound_violation =
      dict_double(dict, "actuator_bound_violation", 0.0);
    diagnostics.penalty_total = dict_double(dict, "penalty_total", 0.0);
    diagnostics.pva_residual = dict_double(dict, "pva_residual", 0.0);
    diagnostics.continuity_residual = dict_double(dict, "continuity_residual", 0.0);
    diagnostics.solve_time_ms = dict_double(dict, "minco_cpp_planner_solve_time_ms", 0.0);
    diagnostics.wall_time_ms = dict_double(dict, "minco_cpp_planner_wall_time_ms", 0.0);
    diagnostics.optimizer_iterations =
      dict_double(dict, "minco_optimizer_iterations", 0.0);
    diagnostics.frontend_selected_depth =
      dict_double(dict, "frontend_selected_depth", 0.0);
    diagnostics.frontend_expansion_count =
      dict_double(dict, "frontend_expansion_count", 0.0);
    return diagnostics;
  }

  void publishStatus(
    double now,
    bool should_plan,
    uint8_t reason,
    uint8_t candidate_status,
    const std::string & failure_reason)
  {
    auto msg = robotx_safe_docking_msgs::msg::MincoPlannerStatus();
    msg.header.stamp = sec_to_stamp(get_clock()->now().seconds());
    msg.header.frame_id = frame_id_;
    msg.trajectory_id = (
      has_pending_ ? pending_trajectory_id_ :
      (has_active_ ? active_trajectory_id_ : trajectory_id_));
    msg.should_plan = should_plan;
    msg.should_plan_reason = reason;
    msg.candidate_status = candidate_status;
    msg.failure_reason = failure_reason;
    msg.candidate_count = candidate_count_;
    msg.accepted_count = accepted_count_;
    msg.rejected_count = rejected_count_;
    msg.time_since_last_plan = (
      last_plan_time_ >= 0.0 ? now - last_plan_time_ : 0.0);
    msg.progress = last_progress_;
    msg.yaw_progress = last_yaw_progress_;
    msg.goal_distance = last_goal_distance_;
    msg.goal_yaw_error = last_goal_yaw_error_;
    const Diagnostics & diagnostics =
      (candidate_status == kCandidateRejected) ?
      last_candidate_diagnostics_ : active_diagnostics_;
    msg.max_abs_tau_v = diagnostics.max_abs_tau_v;
    msg.rms_tau_v = diagnostics.rms_tau_v;
    msg.velocity_violation = diagnostics.velocity_violation;
    msg.acceleration_violation = diagnostics.acceleration_violation;
    msg.actuator_bound_violation = diagnostics.actuator_bound_violation;
    msg.solve_time_ms = diagnostics.solve_time_ms;
    msg.continuity_residual = diagnostics.continuity_residual;
    status_pub_->publish(msg);
  }

  std::string frame_id_;
  double planner_period_ = 1.0;
  double check_interval_ = 1.0;
  double progress_distance_ = 0.5;
  double progress_yaw_ = 0.12;
  double no_replan_goal_distance_ = 0.5;
  double no_replan_goal_yaw_ = 0.08;
  double max_replan_tracking_position_error_ = 0.75;
  double max_replan_tracking_yaw_error_ = 0.70;
  Eigen::Vector3d goal_ = Eigen::Vector3d(3.0, 6.0, 2.4);

  double mass_ = 180.0;
  double iz_ = 446.0;
  double du_ = 100.0;
  double duu_ = 150.0;
  double dv_ = 100.0;
  double dvv_ = 100.0;
  double dr_ = 800.0;
  double drr_ = 800.0;
  double thruster_half_spacing_ = 1.027135;
  double min_thrust_ = -100.0;
  double max_thrust_ = 100.0;

  double frontend_sample_thrust_limit_ = 60.0;
  double minco_dt_ = 0.1;
  double minco_tau_v_bar_ = 12.0;
  double accept_tau_v_margin_ = 0.5;
  double accept_velocity_violation_ = 1e-3;
  double accept_acceleration_violation_ = 1e-3;
  double accept_actuator_violation_ = 1e-3;
  double accept_pva_residual_ = 1e-5;
  double accept_continuity_residual_ = 1e-5;
  double accept_penalty_total_ = 1e7;

  py::object minco_module_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<robotx_safe_docking_msgs::msg::MincoTrajectory>::SharedPtr
    trajectory_pub_;
  rclcpp::Publisher<robotx_safe_docking_msgs::msg::MincoPlannerStatus>::SharedPtr
    status_pub_;
  rclcpp::Subscription<robotx_safe_docking_msgs::msg::MincoExecutionStatus>::SharedPtr
    execution_status_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool has_pending_ = false;
  uint32_t pending_trajectory_id_ = 0;
  double pending_publish_time_ = 0.0;
  Candidate pending_candidate_;

  bool has_active_ = false;
  uint32_t active_trajectory_id_ = 0;
  double active_start_time_ = 0.0;
  double active_duration_ = 0.0;
  std::vector<double> active_segment_times_;
  std::vector<double> active_coefficients_;
  Diagnostics active_diagnostics_;
  Diagnostics last_candidate_diagnostics_;

  uint32_t trajectory_id_ = 0;
  uint32_t candidate_count_ = 0;
  uint32_t accepted_count_ = 0;
  uint32_t rejected_count_ = 0;
  double last_plan_time_ = -1.0;
  double last_progress_ = 0.0;
  double last_yaw_progress_ = 0.0;
  double last_goal_distance_ = -1.0;
  double last_goal_yaw_error_ = -1.0;
  bool controller_executing_active_ = false;
  double latest_execution_status_time_ = -1.0;
  double controller_tracking_position_error_ = 0.0;
  double controller_tracking_yaw_error_ = 0.0;
};

int main(int argc, char ** argv)
{
  py::initialize_interpreter();
  try {
    rclcpp::init(argc, argv);
    {
      auto node = std::make_shared<MincoReplannerNode>();
      rclcpp::spin(node);
    }
    rclcpp::shutdown();
  } catch (...) {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    py::finalize_interpreter();
    throw;
  }
  py::finalize_interpreter();
  return 0;
}
