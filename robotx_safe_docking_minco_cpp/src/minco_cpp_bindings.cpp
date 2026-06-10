#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>
#include "robotx_safe_docking_minco_cpp/lbfgs.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr int kDim = 3;
constexpr int kCoeff = 6;

struct UsvParams {
  double mass = 180.0;
  double iz = 446.0;
  double du = 100.0;
  double duu = 150.0;
  double dv = 100.0;
  double dvv = 100.0;
  double dr = 800.0;
  double drr = 800.0;
  double thruster_half_spacing = 1.027135;
  double min_thrust = -500.0;
  double max_thrust = 500.0;
};

struct OptimizerConfig {
  int piece_count = 3;
  bool fixed_total_time_enabled = false;
  double fixed_total_time = 0.0;
  double reference_speed = 0.8;
  double time_dilation = 1.3;
  double min_segment_time = 0.2;
  double smooth_weight = 1.0;
  double time_weight = 0.0;
  int max_iterations = 80;
  double gradient_tolerance = 1e-6;
  double function_tolerance = 1e-9;
  double spatial_margin = 2.0;
  double yaw_margin = 1.0;
  double theta_bound = 6.0;
};

struct PenaltyConfig {
  UsvParams params;
  double tau_v_bar = 250.0;
  Eigen::Vector3d velocity_bounds = Eigen::Vector3d(2.0, 0.5, 0.6);
  Eigen::Vector3d acceleration_bounds = Eigen::Vector3d(0.9, 0.5, 0.8);
  double lambda_tau_v = 1.0;
  double lambda_velocity = 1.0;
  double lambda_acceleration = 1.0;
  double lambda_actuator = 10.0;
  int quadrature_order = 8;
  double penalty_mu = 20.0;
};

struct NodeEval {
  double tau_v_penalty = 0.0;
  double velocity_penalty = 0.0;
  double acceleration_penalty = 0.0;
  double actuator_penalty = 0.0;
  double weighted_total = 0.0;
  Eigen::Vector3d grad_z = Eigen::Vector3d::Zero();
  Eigen::Vector3d grad_z_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d grad_z_ddot = Eigen::Vector3d::Zero();
  double tau_v = 0.0;
  double tau_v_sq = 0.0;
  double velocity_violation = 0.0;
  double acceleration_violation = 0.0;
  double actuator_violation = 0.0;
  double max_abs_tau_u = 0.0;
  double max_abs_tau_r = 0.0;
  double max_left = -1e300;
  double min_left = 1e300;
  double max_right = -1e300;
  double min_right = 1e300;
};

struct PenaltyResult {
  double total = 0.0;
  double tau_v_component = 0.0;
  double velocity_component = 0.0;
  double acceleration_component = 0.0;
  double actuator_component = 0.0;
  Eigen::MatrixXd coeff_grad;
  Eigen::VectorXd time_grad;
  double max_abs_tau_v = 0.0;
  double tau_v_sq_sum = 0.0;
  int metric_count = 0;
  double velocity_violation = 0.0;
  double acceleration_violation = 0.0;
  double actuator_violation = 0.0;
  double max_abs_tau_u = 0.0;
  double max_abs_tau_r = 0.0;
  double max_left = -1e300;
  double min_left = 1e300;
  double max_right = -1e300;
  double min_right = 1e300;
};

double dict_double(const py::dict &dict, const char *key, double fallback) {
  if (!dict.contains(key)) {
    return fallback;
  }
  return py::cast<double>(dict[key]);
}

int dict_int(const py::dict &dict, const char *key, int fallback) {
  if (!dict.contains(key)) {
    return fallback;
  }
  return py::cast<int>(dict[key]);
}

bool dict_bool(const py::dict &dict, const char *key, bool fallback) {
  if (!dict.contains(key)) {
    return fallback;
  }
  return py::cast<bool>(dict[key]);
}

Eigen::Vector3d dict_vec3(
    const py::dict &dict, const char *key, const Eigen::Vector3d &fallback) {
  if (!dict.contains(key)) {
    return fallback;
  }
  std::vector<double> values = py::cast<std::vector<double>>(dict[key]);
  if (values.size() != 3) {
    throw std::runtime_error(std::string(key) + " must contain three values.");
  }
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

UsvParams params_from_dict(const py::dict &dict) {
  UsvParams params;
  params.mass = dict_double(dict, "mass", params.mass);
  params.iz = dict_double(dict, "iz", params.iz);
  params.du = dict_double(dict, "du", params.du);
  params.duu = dict_double(dict, "duu", params.duu);
  params.dv = dict_double(dict, "dv", params.dv);
  params.dvv = dict_double(dict, "dvv", params.dvv);
  params.dr = dict_double(dict, "dr", params.dr);
  params.drr = dict_double(dict, "drr", params.drr);
  params.thruster_half_spacing = dict_double(
      dict, "thruster_half_spacing", params.thruster_half_spacing);
  params.min_thrust = dict_double(dict, "min_thrust", params.min_thrust);
  params.max_thrust = dict_double(dict, "max_thrust", params.max_thrust);
  return params;
}

OptimizerConfig optimizer_config_from_dict(const py::dict &dict) {
  OptimizerConfig config;
  config.piece_count = dict_int(dict, "piece_count", config.piece_count);
  config.fixed_total_time_enabled = dict_bool(
      dict, "fixed_total_time_enabled", config.fixed_total_time_enabled);
  config.fixed_total_time = dict_double(
      dict, "fixed_total_time", config.fixed_total_time);
  config.reference_speed = dict_double(
      dict, "reference_speed", config.reference_speed);
  config.time_dilation = dict_double(
      dict, "time_dilation", config.time_dilation);
  config.min_segment_time = dict_double(
      dict, "min_segment_time", config.min_segment_time);
  config.smooth_weight = dict_double(dict, "smooth_weight", config.smooth_weight);
  config.time_weight = dict_double(dict, "time_weight", config.time_weight);
  config.max_iterations = dict_int(
      dict, "max_iterations", config.max_iterations);
  config.gradient_tolerance = dict_double(
      dict, "gradient_tolerance", config.gradient_tolerance);
  config.function_tolerance = dict_double(
      dict, "function_tolerance", config.function_tolerance);
  config.spatial_margin = dict_double(
      dict, "spatial_margin", config.spatial_margin);
  config.yaw_margin = dict_double(dict, "yaw_margin", config.yaw_margin);
  config.theta_bound = dict_double(
      dict, "theta_bound", config.theta_bound);
  return config;
}

PenaltyConfig penalty_config_from_dict(
    const py::dict &penalty_dict, const py::dict &params_dict) {
  PenaltyConfig config;
  config.params = params_from_dict(params_dict);
  config.tau_v_bar = dict_double(penalty_dict, "tau_v_bar", config.tau_v_bar);
  config.velocity_bounds = dict_vec3(
      penalty_dict, "velocity_bounds", config.velocity_bounds);
  config.acceleration_bounds = dict_vec3(
      penalty_dict, "acceleration_bounds", config.acceleration_bounds);
  config.lambda_tau_v = dict_double(
      penalty_dict, "lambda_tau_v", config.lambda_tau_v);
  config.lambda_velocity = dict_double(
      penalty_dict, "lambda_velocity", config.lambda_velocity);
  config.lambda_acceleration = dict_double(
      penalty_dict, "lambda_acceleration", config.lambda_acceleration);
  config.lambda_actuator = dict_double(
      penalty_dict, "lambda_actuator", config.lambda_actuator);
  config.quadrature_order = dict_int(
      penalty_dict, "quadrature_order", config.quadrature_order);
  config.penalty_mu = dict_double(penalty_dict, "penalty_mu", config.penalty_mu);
  if (config.quadrature_order <= 0) {
    throw std::runtime_error("quadrature_order must be positive.");
  }
  return config;
}

Eigen::VectorXd array_to_vector(const py::array_t<double> &array) {
  py::buffer_info info = array.request();
  if (info.ndim != 1) {
    throw std::runtime_error("Expected a 1-D array.");
  }
  Eigen::VectorXd out(info.shape[0]);
  const auto *data = static_cast<const double *>(info.ptr);
  for (ssize_t i = 0; i < info.shape[0]; ++i) {
    out(static_cast<int>(i)) = data[i];
  }
  return out;
}

Eigen::MatrixXd array_to_matrix(
    const py::array_t<double> &array, int rows, int cols, const char *name) {
  py::buffer_info info = array.request();
  if (info.ndim != 2 || info.shape[0] != rows || info.shape[1] != cols) {
    throw std::runtime_error(std::string(name) + " has wrong shape.");
  }
  Eigen::MatrixXd out(rows, cols);
  const auto *data = static_cast<const double *>(info.ptr);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out(r, c) = data[r * info.strides[0] / sizeof(double) +
                       c * info.strides[1] / sizeof(double)];
    }
  }
  return out;
}

py::array_t<double> vector_to_numpy(const Eigen::VectorXd &vector) {
  py::array_t<double> result(vector.size());
  py::buffer_info info = result.request();
  auto *data = static_cast<double *>(info.ptr);
  for (int i = 0; i < vector.size(); ++i) {
    data[i] = vector(i);
  }
  return result;
}

py::array_t<double> matrix_to_numpy(const Eigen::MatrixXd &matrix) {
  py::array_t<double> result({matrix.rows(), matrix.cols()});
  py::buffer_info info = result.request();
  auto *data = static_cast<double *>(info.ptr);
  for (int r = 0; r < matrix.rows(); ++r) {
    for (int c = 0; c < matrix.cols(); ++c) {
      data[r * matrix.cols() + c] = matrix(r, c);
    }
  }
  return result;
}

Eigen::Vector3d array_to_vec3(const py::array_t<double> &array, const char *name) {
  const Eigen::VectorXd vector = array_to_vector(array);
  if (vector.size() != 3) {
    throw std::runtime_error(std::string(name) + " must contain three values.");
  }
  return vector;
}

Eigen::Vector3d row_vec3(const Eigen::MatrixXd &matrix, int row) {
  return matrix.row(row).transpose();
}

Eigen::MatrixXd single_row_matrix(const Eigen::Vector3d &vector) {
  Eigen::MatrixXd matrix(1, 3);
  matrix.row(0) = vector.transpose();
  return matrix;
}

double factorial_ratio(int power, int derivative) {
  if (power < derivative) {
    return 0.0;
  }
  double value = 1.0;
  for (int item = power - derivative + 1; item <= power; ++item) {
    value *= static_cast<double>(item);
  }
  return value;
}

Eigen::Matrix<double, kCoeff, 1> basis(double t, int derivative) {
  Eigen::Matrix<double, kCoeff, 1> values;
  values.setZero();
  for (int power = 0; power < kCoeff; ++power) {
    if (power < derivative) {
      continue;
    }
    values(power) = factorial_ratio(power, derivative) *
                    std::pow(t, power - derivative);
  }
  return values;
}

void add_basis_row(
    Eigen::MatrixXd &A, int row, int piece_index, double t, int derivative,
    double scale = 1.0) {
  const auto b = basis(t, derivative);
  const int col = piece_index * kCoeff;
  for (int i = 0; i < kCoeff; ++i) {
    A(row, col + i) += scale * b(i);
  }
}

struct MincoTrajectory {
  int piece_count = 0;
  Eigen::Matrix<double, 3, 3> head_pva;
  Eigen::Matrix<double, 3, 3> tail_pva;
  Eigen::MatrixXd inPs;
  Eigen::VectorXd ts;
  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
  Eigen::MatrixXd coeffs;
};

void build_minco_system(MincoTrajectory &traj) {
  const int M = traj.piece_count;
  const int row_count = M * kCoeff;
  traj.A.resize(row_count, row_count);
  traj.b.resize(row_count, kDim);
  traj.A.setZero();
  traj.b.setZero();

  int row = 0;
  for (int derivative = 0; derivative < 3; ++derivative) {
    add_basis_row(traj.A, row, 0, 0.0, derivative);
    traj.b.row(row) = traj.head_pva.row(derivative);
    row += 1;
  }

  for (int joint_index = 0; joint_index < M - 1; ++joint_index) {
    const double T = traj.ts(joint_index);
    const int next_piece = joint_index + 1;
    const Eigen::RowVector3d q = traj.inPs.row(joint_index);

    add_basis_row(traj.A, row, joint_index, T, 0);
    traj.b.row(row) = q;
    row += 1;

    add_basis_row(traj.A, row, next_piece, 0.0, 0);
    traj.b.row(row) = q;
    row += 1;

    for (int derivative = 1; derivative < 5; ++derivative) {
      add_basis_row(traj.A, row, joint_index, T, derivative);
      add_basis_row(traj.A, row, next_piece, 0.0, derivative, -1.0);
      row += 1;
    }
  }

  const int tail_piece = M - 1;
  const double tail_T = traj.ts(tail_piece);
  for (int derivative = 0; derivative < 3; ++derivative) {
    add_basis_row(traj.A, row, tail_piece, tail_T, derivative);
    traj.b.row(row) = traj.tail_pva.row(derivative);
    row += 1;
  }

  if (row != row_count) {
    throw std::runtime_error("Internal MINCO linear-system row mismatch.");
  }
  traj.coeffs = traj.A.partialPivLu().solve(traj.b);
}

Eigen::Matrix<double, kCoeff, kCoeff> jerk_gram(double T) {
  Eigen::Matrix<double, kCoeff, kCoeff> gram;
  gram.setZero();
  for (int i = 0; i < kCoeff; ++i) {
    const double ai = factorial_ratio(i, 3);
    if (ai == 0.0) {
      continue;
    }
    for (int j = 0; j < kCoeff; ++j) {
      const double aj = factorial_ratio(j, 3);
      if (aj == 0.0) {
        continue;
      }
      const int power = i + j - 5;
      gram(i, j) = ai * aj * std::pow(T, power) / static_cast<double>(power);
    }
  }
  return gram;
}

double trajectory_energy(const MincoTrajectory &traj) {
  double energy = 0.0;
  for (int piece = 0; piece < traj.piece_count; ++piece) {
    const auto gram = jerk_gram(traj.ts(piece));
    const Eigen::Matrix<double, kCoeff, kDim> coeff =
        traj.coeffs.block(piece * kCoeff, 0, kCoeff, kDim);
    energy += (coeff.transpose() * gram * coeff).trace();
  }
  return energy;
}

void energy_gradients(
    const MincoTrajectory &traj, Eigen::MatrixXd &coeff_grad,
    Eigen::VectorXd &time_grad) {
  coeff_grad = Eigen::MatrixXd::Zero(traj.piece_count * kCoeff, kDim);
  time_grad = Eigen::VectorXd::Zero(traj.piece_count);
  for (int piece = 0; piece < traj.piece_count; ++piece) {
    const double T = traj.ts(piece);
    const auto gram = jerk_gram(T);
    const Eigen::Matrix<double, kCoeff, kDim> coeff =
        traj.coeffs.block(piece * kCoeff, 0, kCoeff, kDim);
    coeff_grad.block(piece * kCoeff, 0, kCoeff, kDim) = 2.0 * gram * coeff;
    const auto jerk_T = basis(T, 3).transpose() * coeff;
    time_grad(piece) = jerk_T.squaredNorm();
  }
}

double softplus_scalar(double x) {
  if (x > 40.0) {
    return x;
  }
  return std::log1p(std::exp(-std::abs(x))) + std::max(x, 0.0);
}

double sigmoid_scalar(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double exp_x = std::exp(x);
  return exp_x / (1.0 + exp_x);
}

double smooth_positive_penalty(double g, double mu) {
  const double clipped = std::clamp(g, -1e6, 1e6);
  const double safe_mu = std::max(mu, 1e-6);
  const double sp = softplus_scalar(safe_mu * clipped);
  return (sp / safe_mu) * (sp / safe_mu);
}

double smooth_positive_penalty_derivative(double g, double mu) {
  if (g < -1e6 || g > 1e6) {
    return 0.0;
  }
  const double safe_mu = std::max(mu, 1e-6);
  const double x = safe_mu * std::clamp(g, -1e6, 1e6);
  return 2.0 * (softplus_scalar(x) / safe_mu) * sigmoid_scalar(x);
}

Eigen::Vector3d flat_to_body_velocity(double psi, const Eigen::Vector3d &z_dot) {
  const double c = std::cos(psi);
  const double s = std::sin(psi);
  return Eigen::Vector3d(
      c * z_dot(0) + s * z_dot(1),
      -s * z_dot(0) + c * z_dot(1),
      z_dot(2));
}

Eigen::Vector3d flat_to_body_acceleration(
    double psi, const Eigen::Vector3d &z_dot, const Eigen::Vector3d &z_ddot) {
  const double c = std::cos(psi);
  const double s = std::sin(psi);
  const Eigen::Vector3d nu = flat_to_body_velocity(psi, z_dot);
  const double u = nu(0);
  const double v = nu(1);
  const double r = nu(2);
  return Eigen::Vector3d(
      c * z_ddot(0) + s * z_ddot(1) + r * v,
      -s * z_ddot(0) + c * z_ddot(1) - r * u,
      z_ddot(2));
}

Eigen::Vector3d theta_tau_nominal(
    const Eigen::Vector3d &z, const Eigen::Vector3d &z_dot,
    const Eigen::Vector3d &z_ddot, const UsvParams &params) {
  const double psi = z(2);
  const Eigen::Vector3d nu = flat_to_body_velocity(psi, z_dot);
  const Eigen::Vector3d nu_dot = flat_to_body_acceleration(psi, z_dot, z_ddot);
  const double u = nu(0);
  const double v = nu(1);
  const double r = nu(2);
  return Eigen::Vector3d(
      params.mass * nu_dot(0) - params.mass * v * r +
          params.du * u + params.duu * std::abs(u) * u,
      params.mass * nu_dot(1) + params.mass * u * r +
          params.dv * v + params.dvv * std::abs(v) * v,
      params.iz * nu_dot(2) +
          params.dr * r + params.drr * std::abs(r) * r);
}

std::pair<double, double> generalized_force_to_thrust(
    double tau_u, double tau_r, const UsvParams &params) {
  const double l = params.thruster_half_spacing;
  return {
      0.5 * (tau_u - tau_r / l),
      0.5 * (tau_u + tau_r / l),
  };
}

Eigen::Vector3d abs_gradient(const Eigen::Vector3d &value) {
  Eigen::Vector3d grad;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(value(i)) < 1e-12) {
      grad(i) = 0.0;
    } else {
      grad(i) = value(i) > 0.0 ? 1.0 : -1.0;
    }
  }
  return grad;
}

NodeEval evaluate_node(
    const Eigen::Vector3d &z, const Eigen::Vector3d &z_dot,
    const Eigen::Vector3d &z_ddot, const PenaltyConfig &config) {
  const auto &params = config.params;
  const double psi = z(2);
  const double c = std::cos(psi);
  const double s = std::sin(psi);
  const Eigen::Vector3d nu = flat_to_body_velocity(psi, z_dot);
  const Eigen::Vector3d nu_dot = flat_to_body_acceleration(psi, z_dot, z_ddot);
  const Eigen::Vector3d tau = theta_tau_nominal(z, z_dot, z_ddot, params);
  const auto thrust = generalized_force_to_thrust(tau(0), tau(2), params);
  const double left = thrust.first;
  const double right = thrust.second;
  const double mu = config.penalty_mu;

  const Eigen::Vector2d tau_v_g(
      tau(1) - config.tau_v_bar,
      -tau(1) - config.tau_v_bar);
  const Eigen::Vector3d velocity_g =
      nu.cwiseAbs() - config.velocity_bounds;
  const Eigen::Vector3d acceleration_g =
      nu_dot.cwiseAbs() - config.acceleration_bounds;
  Eigen::Matrix<double, 4, 1> actuator_g;
  actuator_g << params.min_thrust - left, left - params.max_thrust,
      params.min_thrust - right, right - params.max_thrust;

  NodeEval out;
  for (int i = 0; i < 2; ++i) {
    out.tau_v_penalty += smooth_positive_penalty(tau_v_g(i), mu);
  }
  for (int i = 0; i < 3; ++i) {
    out.velocity_penalty += smooth_positive_penalty(velocity_g(i), mu);
    out.acceleration_penalty += smooth_positive_penalty(acceleration_g(i), mu);
  }
  for (int i = 0; i < 4; ++i) {
    out.actuator_penalty += smooth_positive_penalty(actuator_g(i), mu);
  }

  const double u = nu(0);
  const double v = nu(1);
  const double r = nu(2);
  const double x_ddot = z_ddot(0);
  const double y_ddot = z_ddot(1);
  const double a_u = c * x_ddot + s * y_ddot;
  const double a_v = -s * x_ddot + c * y_ddot;

  Eigen::Matrix3d jac_nu_z = Eigen::Matrix3d::Zero();
  jac_nu_z(0, 2) = v;
  jac_nu_z(1, 2) = -u;
  Eigen::Matrix3d jac_nu_z_dot;
  jac_nu_z_dot << c, s, 0.0,
      -s, c, 0.0,
      0.0, 0.0, 1.0;

  Eigen::Matrix3d jac_nu_dot_z = Eigen::Matrix3d::Zero();
  jac_nu_dot_z(0, 2) = a_v - r * u;
  jac_nu_dot_z(1, 2) = -a_u - r * v;
  Eigen::Matrix3d jac_nu_dot_z_dot = Eigen::Matrix3d::Zero();
  jac_nu_dot_z_dot.row(0) =
      r * Eigen::RowVector3d(-s, c, 0.0) + Eigen::RowVector3d(0.0, 0.0, v);
  jac_nu_dot_z_dot.row(1) =
      -r * Eigen::RowVector3d(c, s, 0.0) - Eigen::RowVector3d(0.0, 0.0, u);
  Eigen::Matrix3d jac_nu_dot_z_ddot;
  jac_nu_dot_z_ddot << c, s, 0.0,
      -s, c, 0.0,
      0.0, 0.0, 1.0;

  const double du_gain = params.du + 2.0 * params.duu * std::abs(u);
  const double dv_gain = params.dv + 2.0 * params.dvv * std::abs(v);
  const double dr_gain = params.dr + 2.0 * params.drr * std::abs(r);
  Eigen::Matrix3d jac_tau_z = Eigen::Matrix3d::Zero();
  jac_tau_z(0, 2) = params.mass * a_v + du_gain * v;
  jac_tau_z(1, 2) = -params.mass * a_u - dv_gain * u;
  Eigen::Matrix3d jac_tau_z_dot = Eigen::Matrix3d::Zero();
  jac_tau_z_dot.row(0) = du_gain * Eigen::RowVector3d(c, s, 0.0);
  jac_tau_z_dot.row(1) = dv_gain * Eigen::RowVector3d(-s, c, 0.0);
  jac_tau_z_dot.row(2) = dr_gain * Eigen::RowVector3d(0.0, 0.0, 1.0);
  Eigen::Matrix3d jac_tau_z_ddot;
  jac_tau_z_ddot << params.mass * c, params.mass * s, 0.0,
      -params.mass * s, params.mass * c, 0.0,
      0.0, 0.0, params.iz;

  Eigen::Vector3d grad_z = Eigen::Vector3d::Zero();
  Eigen::Vector3d grad_z_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d grad_z_ddot = Eigen::Vector3d::Zero();

  const Eigen::Vector2d tau_v_dphi(
      smooth_positive_penalty_derivative(tau_v_g(0), mu),
      smooth_positive_penalty_derivative(tau_v_g(1), mu));
  Eigen::Vector3d grad_tau = Eigen::Vector3d::Zero();
  grad_tau(1) = config.lambda_tau_v * (tau_v_dphi(0) - tau_v_dphi(1));
  grad_z += jac_tau_z.transpose() * grad_tau;
  grad_z_dot += jac_tau_z_dot.transpose() * grad_tau;
  grad_z_ddot += jac_tau_z_ddot.transpose() * grad_tau;

  Eigen::Vector3d velocity_dphi;
  Eigen::Vector3d acceleration_dphi;
  for (int i = 0; i < 3; ++i) {
    velocity_dphi(i) = smooth_positive_penalty_derivative(velocity_g(i), mu);
    acceleration_dphi(i) =
        smooth_positive_penalty_derivative(acceleration_g(i), mu);
  }
  const Eigen::Vector3d grad_nu =
      config.lambda_velocity * velocity_dphi.cwiseProduct(abs_gradient(nu));
  grad_z += jac_nu_z.transpose() * grad_nu;
  grad_z_dot += jac_nu_z_dot.transpose() * grad_nu;

  const Eigen::Vector3d grad_nu_dot =
      config.lambda_acceleration *
      acceleration_dphi.cwiseProduct(abs_gradient(nu_dot));
  grad_z += jac_nu_dot_z.transpose() * grad_nu_dot;
  grad_z_dot += jac_nu_dot_z_dot.transpose() * grad_nu_dot;
  grad_z_ddot += jac_nu_dot_z_ddot.transpose() * grad_nu_dot;

  Eigen::Matrix<double, 4, 1> actuator_dphi;
  for (int i = 0; i < 4; ++i) {
    actuator_dphi(i) = smooth_positive_penalty_derivative(actuator_g(i), mu);
  }
  const double grad_left =
      config.lambda_actuator * (-actuator_dphi(0) + actuator_dphi(1));
  const double grad_right =
      config.lambda_actuator * (-actuator_dphi(2) + actuator_dphi(3));
  const double l = params.thruster_half_spacing;
  grad_tau << 0.5 * (grad_left + grad_right),
      0.0,
      0.5 * (grad_right - grad_left) / l;
  grad_z += jac_tau_z.transpose() * grad_tau;
  grad_z_dot += jac_tau_z_dot.transpose() * grad_tau;
  grad_z_ddot += jac_tau_z_ddot.transpose() * grad_tau;

  out.weighted_total =
      config.lambda_tau_v * out.tau_v_penalty +
      config.lambda_velocity * out.velocity_penalty +
      config.lambda_acceleration * out.acceleration_penalty +
      config.lambda_actuator * out.actuator_penalty;
  out.grad_z = grad_z;
  out.grad_z_dot = grad_z_dot;
  out.grad_z_ddot = grad_z_ddot;
  out.tau_v = tau(1);
  out.tau_v_sq = tau(1) * tau(1);
  out.velocity_violation = std::max(0.0, velocity_g.maxCoeff());
  out.acceleration_violation = std::max(0.0, acceleration_g.maxCoeff());
  out.actuator_violation = std::max(0.0, actuator_g.maxCoeff());
  out.max_abs_tau_u = std::abs(tau(0));
  out.max_abs_tau_r = std::abs(tau(2));
  out.max_left = left;
  out.min_left = left;
  out.max_right = right;
  out.min_right = right;
  return out;
}

void gauss_legendre(int order, std::vector<double> &alphas,
                    std::vector<double> &weights) {
  const int n = std::max(order, 1);
  std::vector<double> nodes(n);
  std::vector<double> raw_weights(n);
  const int m = (n + 1) / 2;
  const double eps = 1e-14;
  for (int i = 0; i < m; ++i) {
    double z = std::cos(M_PI * (static_cast<double>(i) + 0.75) /
                        (static_cast<double>(n) + 0.5));
    double z_prev = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double pp = 0.0;
    while (std::abs(z - z_prev) > eps) {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z_prev = z;
      z = z_prev - p1 / pp;
    }
    nodes[i] = -z;
    nodes[n - 1 - i] = z;
    const double w = 2.0 / ((1.0 - z * z) * pp * pp);
    raw_weights[i] = w;
    raw_weights[n - 1 - i] = w;
  }
  alphas.resize(n);
  weights.resize(n);
  for (int i = 0; i < n; ++i) {
    alphas[i] = 0.5 * (nodes[i] + 1.0);
    weights[i] = 0.5 * raw_weights[i];
  }
}

void reset_penalty_result(PenaltyResult &result, int piece_count) {
  result.total = 0.0;
  result.tau_v_component = 0.0;
  result.velocity_component = 0.0;
  result.acceleration_component = 0.0;
  result.actuator_component = 0.0;
  result.coeff_grad.resize(piece_count * kCoeff, kDim);
  result.coeff_grad.setZero();
  result.time_grad.resize(piece_count);
  result.time_grad.setZero();
  result.max_abs_tau_v = 0.0;
  result.tau_v_sq_sum = 0.0;
  result.metric_count = 0;
  result.velocity_violation = 0.0;
  result.acceleration_violation = 0.0;
  result.actuator_violation = 0.0;
  result.max_abs_tau_u = 0.0;
  result.max_abs_tau_r = 0.0;
  result.max_left = -1e300;
  result.min_left = 1e300;
  result.max_right = -1e300;
  result.min_right = 1e300;
}

void penalty_and_gradients_with_quadrature(
    const MincoTrajectory &traj, const PenaltyConfig &config,
    const std::vector<double> &alphas, const std::vector<double> &weights,
    PenaltyResult &result) {
  reset_penalty_result(result, traj.piece_count);
  for (int piece = 0; piece < traj.piece_count; ++piece) {
    const double T = traj.ts(piece);
    const Eigen::Matrix<double, kCoeff, kDim> coeff =
        traj.coeffs.block(piece * kCoeff, 0, kCoeff, kDim);
    for (size_t qi = 0; qi < alphas.size(); ++qi) {
      const double alpha = alphas[qi];
      const double weight = weights[qi];
      const double local_time = alpha * T;
      const auto b0 = basis(local_time, 0);
      const auto b1 = basis(local_time, 1);
      const auto b2 = basis(local_time, 2);
      const auto b3 = basis(local_time, 3);
      const Eigen::Vector3d z = coeff.transpose() * b0;
      const Eigen::Vector3d z_dot = coeff.transpose() * b1;
      const Eigen::Vector3d z_ddot = coeff.transpose() * b2;
      const Eigen::Vector3d z_jerk = coeff.transpose() * b3;
      const NodeEval node = evaluate_node(z, z_dot, z_ddot, config);
      const double scale = T * weight;
      result.tau_v_component += scale * node.tau_v_penalty;
      result.velocity_component += scale * node.velocity_penalty;
      result.acceleration_component += scale * node.acceleration_penalty;
      result.actuator_component += scale * node.actuator_penalty;
      result.coeff_grad.block(piece * kCoeff, 0, kCoeff, kDim) +=
          scale * (
              b0 * node.grad_z.transpose() +
              b1 * node.grad_z_dot.transpose() +
              b2 * node.grad_z_ddot.transpose());
      const double df_ds =
          node.grad_z.dot(z_dot) +
          node.grad_z_dot.dot(z_ddot) +
          node.grad_z_ddot.dot(z_jerk);
      result.time_grad(piece) +=
          weight * (node.weighted_total + T * alpha * df_ds);

      result.max_abs_tau_v =
          std::max(result.max_abs_tau_v, std::abs(node.tau_v));
      result.tau_v_sq_sum += node.tau_v_sq;
      result.metric_count += 1;
      result.velocity_violation =
          std::max(result.velocity_violation, node.velocity_violation);
      result.acceleration_violation =
          std::max(result.acceleration_violation, node.acceleration_violation);
      result.actuator_violation =
          std::max(result.actuator_violation, node.actuator_violation);
      result.max_abs_tau_u =
          std::max(result.max_abs_tau_u, node.max_abs_tau_u);
      result.max_abs_tau_r =
          std::max(result.max_abs_tau_r, node.max_abs_tau_r);
      result.max_left = std::max(result.max_left, node.max_left);
      result.min_left = std::min(result.min_left, node.min_left);
      result.max_right = std::max(result.max_right, node.max_right);
      result.min_right = std::min(result.min_right, node.min_right);
    }
  }
  result.total =
      config.lambda_tau_v * result.tau_v_component +
      config.lambda_velocity * result.velocity_component +
      config.lambda_acceleration * result.acceleration_component +
      config.lambda_actuator * result.actuator_component;
}

PenaltyResult penalty_and_gradients(
    const MincoTrajectory &traj, const PenaltyConfig &config) {
  std::vector<double> alphas;
  std::vector<double> weights;
  gauss_legendre(config.quadrature_order, alphas, weights);
  PenaltyResult result;
  penalty_and_gradients_with_quadrature(traj, config, alphas, weights, result);
  return result;
}

Eigen::VectorXd theta_to_times(
    const Eigen::VectorXd &theta, const OptimizerConfig &config) {
  if (theta.size() != config.piece_count) {
    throw std::runtime_error("theta must have one value per piece.");
  }
  Eigen::VectorXd ts(theta.size());
  if (!config.fixed_total_time_enabled) {
    for (int i = 0; i < theta.size(); ++i) {
      const double clipped = std::clamp(theta(i), -50.0, 50.0);
      ts(i) = config.min_segment_time + std::exp(clipped);
    }
    return ts;
  }
  const double max_theta = theta.maxCoeff();
  Eigen::VectorXd exp_theta(theta.size());
  for (int i = 0; i < theta.size(); ++i) {
    exp_theta(i) = std::exp(theta(i) - max_theta);
  }
  const double total_time = std::max(
      config.fixed_total_time,
      config.piece_count * config.min_segment_time);
  const double remaining =
      total_time - config.piece_count * config.min_segment_time;
  if (remaining <= 1e-12) {
    ts.setConstant(config.min_segment_time);
  } else {
    ts = Eigen::VectorXd::Constant(theta.size(), config.min_segment_time) +
         remaining * exp_theta / exp_theta.sum();
  }
  return ts;
}

Eigen::VectorXd times_gradient_to_theta_gradient(
    const Eigen::VectorXd &grad_ts, const Eigen::VectorXd &ts,
    const OptimizerConfig &config) {
  Eigen::VectorXd grad_theta(grad_ts.size());
  if (!config.fixed_total_time_enabled) {
    grad_theta = grad_ts.cwiseProduct(
        ts.array().unaryExpr([&](double t) {
          return t - config.min_segment_time;
        }).matrix());
    return grad_theta;
  }
  Eigen::VectorXd shifted = ts.array() - config.min_segment_time;
  const double remaining = shifted.sum();
  if (remaining <= 1e-12) {
    grad_theta.setZero();
    return grad_theta;
  }
  const Eigen::VectorXd pi = shifted / remaining;
  const double weighted_mean = grad_ts.dot(pi);
  grad_theta = shifted.cwiseProduct(
      grad_ts.array().unaryExpr([&](double value) {
        return value - weighted_mean;
      }).matrix());
  return grad_theta;
}

double shortest_yaw_delta(double start, double goal);

Eigen::VectorXd times_to_theta(
    const Eigen::VectorXd &ts, const OptimizerConfig &config) {
  if (ts.size() != config.piece_count) {
    throw std::runtime_error("ts must have one value per piece.");
  }
  for (int i = 0; i < ts.size(); ++i) {
    if (ts(i) <= 0.0) {
      throw std::runtime_error("Segment times must be positive.");
    }
  }
  Eigen::VectorXd theta(ts.size());
  if (!config.fixed_total_time_enabled) {
    for (int i = 0; i < ts.size(); ++i) {
      theta(i) = std::log(std::max(ts(i) - config.min_segment_time, 1e-9));
    }
    return theta;
  }
  for (int i = 0; i < ts.size(); ++i) {
    theta(i) = std::log(std::max(ts(i), 1e-9));
  }
  return theta;
}

void unpack_variables(
    const Eigen::VectorXd &variables, const OptimizerConfig &config,
    Eigen::MatrixXd &inPs, Eigen::VectorXd &theta) {
  const int point_size = (config.piece_count - 1) * kDim;
  if (variables.size() != point_size + config.piece_count) {
    throw std::runtime_error("Optimization vector has wrong size.");
  }
  inPs = Eigen::MatrixXd::Zero(std::max(config.piece_count - 1, 0), kDim);
  for (int row = 0; row < config.piece_count - 1; ++row) {
    for (int col = 0; col < kDim; ++col) {
      inPs(row, col) = variables(row * kDim + col);
    }
  }
  theta = variables.segment(point_size, config.piece_count);
}

Eigen::VectorXd pack_variables(
    const Eigen::MatrixXd &inPs, const Eigen::VectorXd &theta) {
  Eigen::VectorXd out(inPs.rows() * kDim + theta.size());
  int index = 0;
  for (int row = 0; row < inPs.rows(); ++row) {
    for (int col = 0; col < kDim; ++col) {
      out(index++) = inPs(row, col);
    }
  }
  out.segment(index, theta.size()) = theta;
  return out;
}

void initial_guess(
    const Eigen::Matrix<double, 3, 3> &head_pva,
    const Eigen::Matrix<double, 3, 3> &tail_pva,
    const OptimizerConfig &config,
    Eigen::MatrixXd &inPs,
    Eigen::VectorXd &ts) {
  const int piece_count = config.piece_count;
  if (piece_count <= 0) {
    throw std::runtime_error("piece_count must be positive.");
  }
  inPs = Eigen::MatrixXd::Zero(std::max(piece_count - 1, 0), kDim);
  Eigen::Vector3d start = head_pva.row(0).transpose();
  Eigen::Vector3d goal = tail_pva.row(0).transpose();
  goal(2) = start(2) + shortest_yaw_delta(start(2), goal(2));

  for (int index = 1; index < piece_count; ++index) {
    const double alpha = static_cast<double>(index) /
                         static_cast<double>(piece_count);
    inPs.row(index - 1) =
        ((1.0 - alpha) * start + alpha * goal).transpose();
  }

  double total_time = 0.0;
  if (config.fixed_total_time_enabled) {
    total_time = std::max(
        config.fixed_total_time,
        static_cast<double>(piece_count) * config.min_segment_time);
  } else {
    const double distance = (goal.head<2>() - start.head<2>()).norm();
    total_time = std::max(
        config.time_dilation * distance / std::max(config.reference_speed, 1e-9),
        static_cast<double>(piece_count) * config.min_segment_time);
  }

  ts = Eigen::VectorXd::Constant(piece_count, total_time / piece_count);
  if (piece_count == 1) {
    return;
  }

  Eigen::VectorXd weights(piece_count);
  Eigen::Vector2d previous = start.head<2>();
  for (int i = 0; i < piece_count - 1; ++i) {
    const Eigen::Vector2d current = inPs.row(i).head<2>().transpose();
    weights(i) = (current - previous).norm() + 1e-6;
    previous = current;
  }
  weights(piece_count - 1) = (goal.head<2>() - previous).norm() + 1e-6;
  ts = total_time * weights / weights.sum();
  for (int i = 0; i < ts.size(); ++i) {
    ts(i) = std::max(ts(i), config.min_segment_time);
  }
  ts *= total_time / ts.sum();
}

double wrap_angle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

Eigen::Vector3d body_to_flat_velocity(double psi, const Eigen::Vector3d &nu) {
  const double c = std::cos(psi);
  const double s = std::sin(psi);
  return Eigen::Vector3d(
      c * nu(0) - s * nu(1),
      s * nu(0) + c * nu(1),
      nu(2));
}

Eigen::Vector3d body_to_flat_acceleration(
    double psi, const Eigen::Vector3d &nu, const Eigen::Vector3d &nu_dot) {
  const double c = std::cos(psi);
  const double s = std::sin(psi);
  const double u = nu(0);
  const double v = nu(1);
  const double r = nu(2);
  const double u_dot = nu_dot(0);
  const double v_dot = nu_dot(1);
  const double r_dot = nu_dot(2);
  return Eigen::Vector3d(
      c * u_dot - s * v_dot - r * (s * u + c * v),
      s * u_dot + c * v_dot + r * (c * u - s * v),
      r_dot);
}

Eigen::Vector3d usv_body_acceleration(
    const Eigen::Vector3d &nu, const Eigen::Vector2d &control,
    const UsvParams &params) {
  const double u = nu(0);
  const double v = nu(1);
  const double r = nu(2);
  const Eigen::Vector3d tau(control(0), 0.0, control(1));
  const Eigen::Vector3d coriolis(-params.mass * v * r, params.mass * u * r, 0.0);
  const Eigen::Vector3d damping(
      params.du * u + params.duu * std::abs(u) * u,
      params.dv * v + params.dvv * std::abs(v) * v,
      params.dr * r + params.drr * std::abs(r) * r);
  const Eigen::Vector3d rhs = tau - coriolis - damping;
  return Eigen::Vector3d(rhs(0) / params.mass, rhs(1) / params.mass,
                         rhs(2) / params.iz);
}

Eigen::Matrix<double, 6, 1> usv_state_derivative(
    const Eigen::Matrix<double, 6, 1> &state,
    const Eigen::Vector2d &control, const UsvParams &params) {
  const Eigen::Vector3d z = state.segment<3>(0);
  const Eigen::Vector3d nu = state.segment<3>(3);
  Eigen::Matrix<double, 6, 1> derivative;
  derivative.segment<3>(0) = body_to_flat_velocity(z(2), nu);
  derivative.segment<3>(3) = usv_body_acceleration(nu, control, params);
  return derivative;
}

Eigen::Matrix<double, 6, 1> rk4_usv_step(
    const Eigen::Matrix<double, 6, 1> &state,
    const Eigen::Vector2d &control, const UsvParams &params, double dt) {
  const auto k1 = usv_state_derivative(state, control, params);
  const auto k2 = usv_state_derivative(state + 0.5 * dt * k1, control, params);
  const auto k3 = usv_state_derivative(state + 0.5 * dt * k2, control, params);
  const auto k4 = usv_state_derivative(state + dt * k3, control, params);
  Eigen::Matrix<double, 6, 1> next =
      state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  next(2) = wrap_angle(next(2));
  return next;
}

struct Obstacle {
  double x = 0.0;
  double y = 0.0;
  double radius = 0.0;
};

struct FrontendConfig {
  Eigen::Vector3d goal_z = Eigen::Vector3d(6.0, 12.0, 2.4);
  double primitive_duration = 2.0;
  double primitive_dt = 0.25;
  int check_num = 5;
  int search_depth = 24;
  int max_expansions = 2000;
  int control_discretization = 1;
  double sample_thrust_limit = 0.0;
  int minco_piece_count_min = 3;
  int minco_piece_count_max = 8;
  Eigen::Vector3d flat_accel_bounds = Eigen::Vector3d(0.5, 0.5, 0.12);
  Eigen::Vector3d flat_velocity_bounds = Eigen::Vector3d(2.0, 2.0, 0.4);
  double time_weight = 1.0;
  double heuristic_weight = 3.0;
  double tau_v_bar = 12.0;
  double max_local_goal_distance = 5.0;
  double local_goal_speed = 0.45;
  double final_goal_radius = 0.8;
  double goal_tolerance = 0.15;
  double yaw_tolerance = 0.12;
  double obstacle_margin = 0.5;
  std::vector<Obstacle> obstacles;
  double grid_resolution_xy = 0.25;
  double grid_resolution_yaw = 0.25;
  double grid_resolution_vxy = 0.25;
  double grid_resolution_yaw_rate = 0.05;
  bool analytic_expansion = true;
};

FrontendConfig frontend_config_from_dict(const py::dict &dict) {
  FrontendConfig config;
  config.goal_z = dict_vec3(dict, "goal_z", config.goal_z);
  config.primitive_duration = dict_double(
      dict, "primitive_duration", config.primitive_duration);
  config.primitive_dt = dict_double(dict, "primitive_dt", config.primitive_dt);
  config.check_num = dict_int(dict, "check_num", config.check_num);
  config.search_depth = dict_int(dict, "search_depth", config.search_depth);
  config.max_expansions = dict_int(dict, "max_expansions", config.max_expansions);
  config.control_discretization = dict_int(
      dict, "control_discretization", config.control_discretization);
  config.sample_thrust_limit = dict_double(
      dict, "sample_thrust_limit", config.sample_thrust_limit);
  config.minco_piece_count_min = dict_int(
      dict, "minco_piece_count_min", config.minco_piece_count_min);
  config.minco_piece_count_max = dict_int(
      dict, "minco_piece_count_max", config.minco_piece_count_max);
  config.flat_accel_bounds = dict_vec3(
      dict, "flat_accel_bounds", config.flat_accel_bounds);
  config.flat_velocity_bounds = dict_vec3(
      dict, "flat_velocity_bounds", config.flat_velocity_bounds);
  config.time_weight = dict_double(dict, "time_weight", config.time_weight);
  config.heuristic_weight = dict_double(
      dict, "heuristic_weight", config.heuristic_weight);
  config.tau_v_bar = dict_double(dict, "tau_v_bar", config.tau_v_bar);
  config.max_local_goal_distance = dict_double(
      dict, "max_local_goal_distance", config.max_local_goal_distance);
  config.local_goal_speed = dict_double(
      dict, "local_goal_speed", config.local_goal_speed);
  config.final_goal_radius = dict_double(
      dict, "final_goal_radius", config.final_goal_radius);
  config.goal_tolerance = dict_double(
      dict, "goal_tolerance", config.goal_tolerance);
  config.yaw_tolerance = dict_double(dict, "yaw_tolerance", config.yaw_tolerance);
  config.obstacle_margin = dict_double(
      dict, "obstacle_margin", config.obstacle_margin);
  config.grid_resolution_xy = dict_double(
      dict, "grid_resolution_xy", config.grid_resolution_xy);
  config.grid_resolution_yaw = dict_double(
      dict, "grid_resolution_yaw", config.grid_resolution_yaw);
  config.grid_resolution_vxy = dict_double(
      dict, "grid_resolution_vxy", config.grid_resolution_vxy);
  config.grid_resolution_yaw_rate = dict_double(
      dict, "grid_resolution_yaw_rate", config.grid_resolution_yaw_rate);
  config.analytic_expansion = dict_bool(
      dict, "analytic_expansion", config.analytic_expansion);
  if (dict.contains("obstacles")) {
    for (const auto item : py::cast<py::list>(dict["obstacles"])) {
      const auto values = py::cast<std::vector<double>>(item);
      if (values.size() != 3) {
        throw std::runtime_error("Each obstacle must have x,y,r.");
      }
      config.obstacles.push_back({values[0], values[1], values[2]});
    }
  }
  return config;
}

std::vector<Eigen::Vector2d> discretized_generalized_inputs(
    const UsvParams &params, const FrontendConfig &config) {
  const int r = std::max(config.control_discretization, 1);
  const int count = 2 * r + 1;
  double lower = params.min_thrust;
  double upper = params.max_thrust;
  if (config.sample_thrust_limit > 0.0) {
    const double limit = std::abs(config.sample_thrust_limit);
    lower = std::max(params.min_thrust, -limit);
    upper = std::min(params.max_thrust, limit);
  }
  std::vector<Eigen::Vector2d> controls;
  std::set<std::pair<long long, long long>> seen;
  for (int i = 0; i < count; ++i) {
    const double left = lower + (upper - lower) * i /
                                    std::max(count - 1, 1);
    for (int j = 0; j < count; ++j) {
      const double right = lower + (upper - lower) * j /
                                      std::max(count - 1, 1);
      const double tau_u = left + right;
      const double tau_r = params.thruster_half_spacing * (right - left);
      const auto key = std::make_pair(
          static_cast<long long>(std::llround(tau_u * 1e6)),
          static_cast<long long>(std::llround(tau_r * 1e6)));
      if (seen.insert(key).second) {
        controls.emplace_back(tau_u, tau_r);
      }
    }
  }
  if (controls.empty()) {
    controls.emplace_back(0.0, 0.0);
  }
  return controls;
}

struct LatticeNodeCpp {
  enum Status { NOT_EXPAND = 0, OPENSET = 1, CLOSED = 2 };
  Eigen::Vector3d z = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_ddot = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu = Eigen::Vector3d::Zero();
  double g_cost = 0.0;
  double h_cost = 0.0;
  double f_cost = 0.0;
  int depth = 0;
  int parent = -1;
  int status = NOT_EXPAND;
  Eigen::Vector2d primitive_u = Eigen::Vector2d::Zero();
  Eigen::MatrixXd segment_path;
  Eigen::MatrixXd segment_z_dot;
  Eigen::MatrixXd segment_z_ddot;
  Eigen::MatrixXd segment_nu;
  Eigen::VectorXd segment_times;
  std::uint64_t key = 0;
};

std::uint64_t mix_grid_word(std::uint64_t seed, std::int64_t value) {
  const std::uint64_t z =
      (static_cast<std::uint64_t>(value) << 1) ^
      static_cast<std::uint64_t>(value >> 63);
  seed ^= z + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  return seed;
}

std::uint64_t grid_key(
    const Eigen::Vector3d &z, const Eigen::Vector3d &nu,
    const FrontendConfig &config) {
  const auto floordiv = [](double value, double resolution) {
    return static_cast<long long>(std::floor(value / std::max(resolution, 1e-6)));
  };
  std::uint64_t seed = 0x517cc1b727220a95ULL;
  seed = mix_grid_word(seed, floordiv(z(0), config.grid_resolution_xy));
  seed = mix_grid_word(seed, floordiv(z(1), config.grid_resolution_xy));
  seed = mix_grid_word(seed, floordiv(wrap_angle(z(2)), config.grid_resolution_yaw));
  seed = mix_grid_word(seed, floordiv(nu(0), config.grid_resolution_vxy));
  seed = mix_grid_word(seed, floordiv(nu(2), config.grid_resolution_yaw_rate));
  return seed;
}

bool collides(
    const Eigen::MatrixXd &path_xy, const std::vector<Obstacle> &obstacles,
    double margin) {
  if (obstacles.empty()) {
    return false;
  }
  for (const auto &obstacle : obstacles) {
    for (int i = 0; i < path_xy.rows(); ++i) {
      const double dx = path_xy(i, 0) - obstacle.x;
      const double dy = path_xy(i, 1) - obstacle.y;
      if (std::sqrt(dx * dx + dy * dy) <= obstacle.radius + margin) {
        return true;
      }
    }
  }
  return false;
}

bool collides_point(
    double x, double y, const std::vector<Obstacle> &obstacles, double margin) {
  for (const auto &obstacle : obstacles) {
    const double dx = x - obstacle.x;
    const double dy = y - obstacle.y;
    if (std::sqrt(dx * dx + dy * dy) <= obstacle.radius + margin) {
      return true;
    }
  }
  return false;
}

bool check_flat_bounds(
    const Eigen::Vector3d &z_dot, const Eigen::Vector3d &z_ddot,
    const FrontendConfig &config) {
  return ((z_dot.cwiseAbs().array() <=
           (config.flat_velocity_bounds.array() + 1e-9)).all() &&
          (z_ddot.cwiseAbs().array() <=
           (config.flat_accel_bounds.array() + 1e-9)).all());
}

bool check_body_velocity_bounds(
    const Eigen::Vector3d &nu, const FrontendConfig &config) {
  return (nu.cwiseAbs().array() <=
          (config.flat_velocity_bounds.array() + 1e-9)).all();
}

bool check_usv_flat_feasible(
    const Eigen::Vector3d &z, const Eigen::Vector3d &z_dot,
    const Eigen::Vector3d &z_ddot, const UsvParams &params,
    const FrontendConfig &config) {
  const Eigen::Vector3d tau = theta_tau_nominal(z, z_dot, z_ddot, params);
  if (std::abs(tau(1)) > config.tau_v_bar + 1e-9) {
    return false;
  }
  const auto thrust = generalized_force_to_thrust(tau(0), tau(2), params);
  return params.min_thrust <= thrust.first && thrust.first <= params.max_thrust &&
         params.min_thrust <= thrust.second && thrust.second <= params.max_thrust;
}

bool check_feasible(
    const LatticeNodeCpp &node, const UsvParams &params,
    const FrontendConfig &config) {
  (void)params;
  if (node.segment_path.rows() < 2) {
    return false;
  }
  for (int i = 0; i < node.segment_path.rows(); ++i) {
    if (!check_body_velocity_bounds(row_vec3(node.segment_nu, i), config)) {
      return false;
    }
  }
  return true;
}

bool rollout_usv_primitive(
    const LatticeNodeCpp &node, const Eigen::Vector2d &control,
    const UsvParams &params, const FrontendConfig &config,
    LatticeNodeCpp &child, bool &collision_reject) {
  collision_reject = false;
  const double duration = std::max(config.primitive_duration, 1e-6);
  const int steps = std::max(config.check_num, 1);
  child = LatticeNodeCpp();
  child.depth = node.depth + 1;
  child.parent = -1;
  child.primitive_u = control;
  Eigen::Matrix<double, 6, 1> state;
  state.segment<3>(0) = node.z;
  state.segment<3>(3) = node.nu;
  const double dt = duration / static_cast<double>(steps);
  for (int i = 1; i <= steps; ++i) {
    state = rk4_usv_step(state, control, params, dt);
    const Eigen::Vector3d z_t = state.segment<3>(0);
    const Eigen::Vector3d nu_t = state.segment<3>(3);
    if (!check_body_velocity_bounds(nu_t, config)) {
      return false;
    }
    if (collides_point(z_t(0), z_t(1), config.obstacles,
                       config.obstacle_margin)) {
      collision_reject = true;
      return false;
    }
  }
  child.z = state.segment<3>(0);
  child.z(2) = wrap_angle(child.z(2));
  child.nu = state.segment<3>(3);
  child.z_dot = body_to_flat_velocity(child.z(2), child.nu);
  const Eigen::Vector3d nu_dot =
      usv_body_acceleration(child.nu, control, params);
  child.z_ddot = body_to_flat_acceleration(child.z(2), child.nu, nu_dot);
  child.key = grid_key(child.z, child.nu, config);
  return true;
}

std::pair<double, double> generalized_input_scales(
    const UsvParams &params) {
  const double tau_u_scale =
      std::max(2.0 * std::max(std::abs(params.min_thrust),
                              std::abs(params.max_thrust)), 1e-6);
  const double tau_r_scale =
      std::max(std::abs(params.thruster_half_spacing *
                       (params.max_thrust - params.min_thrust)), 1e-6);
  return {tau_u_scale, tau_r_scale};
}

double edge_cost(
    const Eigen::Vector2d &control, const UsvParams &params,
    const FrontendConfig &config) {
  const auto thrust = generalized_force_to_thrust(control(0), control(1), params);
  const double thrust_scale =
      std::max({std::abs(params.min_thrust), std::abs(params.max_thrust), 1e-6});
  const double effort = std::pow(thrust.first / thrust_scale, 2.0) +
                        std::pow(thrust.second / thrust_scale, 2.0);
  return (effort + config.time_weight) * std::max(config.primitive_duration, 1e-6);
}

std::pair<double, double> heuristic_cost(
    const Eigen::Vector3d &z, const Eigen::Vector3d &z_dot,
    Eigen::Vector3d goal, const Eigen::Vector3d &goal_dot,
    const FrontendConfig &config) {
  goal(2) = z(2) + wrap_angle(goal(2) - z(2));
  const Eigen::Vector3d p_delta = goal - z;
  const Eigen::Vector3d v_delta = goal_dot - z_dot;
  const double rho = std::max(config.time_weight, 1e-9);
  const double a = 12.0 * p_delta.squaredNorm();
  const double b = (-24.0 * p_delta.dot(z_dot) -
                    12.0 * p_delta.dot(v_delta));
  const double c = (12.0 * z_dot.squaredNorm() +
                    12.0 * z_dot.dot(v_delta) +
                    4.0 * v_delta.squaredNorm());

  Eigen::Matrix<double, 5, 1> poly;
  poly << -3.0 * a, -2.0 * b, -c, 0.0, rho;
  Eigen::PolynomialSolver<double, 4> solver;
  solver.compute(poly);
  std::vector<double> candidates;
  for (const auto &root : solver.roots()) {
    if (std::abs(root.imag()) < 1e-7 && root.real() > 1e-6) {
      candidates.push_back(root.real());
    }
  }
  const Eigen::Vector3d velocity_bounds =
      config.flat_velocity_bounds.cwiseMax(Eigen::Vector3d::Constant(1e-6));
  double lower_bound = std::max(config.primitive_duration, 1e-3);
  for (int i = 0; i < 3; ++i) {
    lower_bound = std::max(
        lower_bound, std::abs(p_delta(i)) / (0.5 * velocity_bounds(i)));
  }
  if (candidates.empty()) {
    const double distance = p_delta.head<2>().norm();
    const double speed = std::max(config.local_goal_speed, 0.1);
    candidates.push_back(std::max(distance / speed, lower_bound));
  }
  candidates.push_back(lower_bound);

  auto cubic_cost = [&](double duration) {
    const double T = std::max(duration, 1e-9);
    const Eigen::Vector3d d = goal - z - z_dot * T;
    const Eigen::Vector3d v = goal_dot - z_dot;
    const Eigen::Vector3d alpha = -12.0 * d / std::pow(T, 3) +
                                  6.0 * v / std::pow(T, 2);
    const Eigen::Vector3d beta = 6.0 * d / std::pow(T, 2) - 2.0 * v / T;
    return (alpha.array().square() * std::pow(T, 3) / 3.0 +
            alpha.array() * beta.array() * std::pow(T, 2) +
            beta.array().square() * T).sum() + rho * T;
  };

  double best_cost = 1e300;
  double best_time = lower_bound;
  for (double duration : candidates) {
    duration = std::max(duration, lower_bound);
    const double cost = cubic_cost(duration);
    if (cost < best_cost) {
      best_cost = cost;
      best_time = duration;
    }
  }
  return {best_cost, best_time};
}

bool goal_inside_local_horizon(
    const Eigen::Vector3d &start_z, const Eigen::Vector3d &goal,
    const FrontendConfig &config) {
  return (goal.head<2>() - start_z.head<2>()).norm() <=
         config.max_local_goal_distance + 1e-9;
}

bool path_inside_local_horizon(
    const Eigen::Vector3d &start_z, const Eigen::MatrixXd &path,
    const FrontendConfig &config) {
  for (int i = 0; i < path.rows(); ++i) {
    if ((path.row(i).head<2>().transpose() - start_z.head<2>()).norm() >
        config.max_local_goal_distance + 1e-9) {
      return false;
    }
  }
  return path.rows() > 0;
}

bool reach_goal(
    const Eigen::Vector3d &z, const Eigen::Vector3d &goal,
    const FrontendConfig &config) {
  const double distance = (goal.head<2>() - z.head<2>()).norm();
  const double yaw_error = std::abs(wrap_angle(goal(2) - z(2)));
  return distance <= config.goal_tolerance && yaw_error <= config.yaw_tolerance;
}

bool reach_horizon(
    const Eigen::Vector3d &start_z, const Eigen::Vector3d &current_z,
    const FrontendConfig &config) {
  return (current_z.head<2>() - start_z.head<2>()).norm() >=
         config.max_local_goal_distance;
}

bool reach_horizon_toward_goal(
    const Eigen::Vector3d &start_z, const Eigen::Vector3d &current_z,
    const Eigen::Vector3d &goal, const FrontendConfig &config) {
  if (!reach_horizon(start_z, current_z, config)) {
    return false;
  }
  const Eigen::Vector2d goal_delta = goal.head<2>() - start_z.head<2>();
  const double goal_distance = goal_delta.norm();
  if (goal_distance < 1e-9) {
    return true;
  }
  const Eigen::Vector2d direction = goal_delta / goal_distance;
  const Eigen::Vector2d progress_vec = current_z.head<2>() - start_z.head<2>();
  const double progress = progress_vec.dot(direction);
  const double lateral_error = std::abs(
      progress_vec(0) * direction(1) - progress_vec(1) * direction(0));
  return progress >= 0.75 * config.max_local_goal_distance &&
         lateral_error <= 0.55 * config.max_local_goal_distance;
}

Eigen::Vector3d cubic_position(
    const Eigen::Vector3d &z, const Eigen::Vector3d &z_dot,
    const Eigen::Vector3d &alpha, const Eigen::Vector3d &beta, double t) {
  return z + z_dot * t + 0.5 * beta * t * t + alpha * std::pow(t, 3) / 6.0;
}

bool analytic_expand(
    const LatticeNodeCpp &node, const Eigen::Vector3d &goal,
    const Eigen::Vector3d &goal_dot, const Eigen::Vector3d &root_z,
    const UsvParams &params, const FrontendConfig &config,
    LatticeNodeCpp &out) {
  const auto ht = heuristic_cost(node.z, node.z_dot, goal, goal_dot, config);
  if (!std::isfinite(ht.first) || !std::isfinite(ht.second)) {
    return false;
  }
  const double duration = std::max(ht.second, std::max(config.primitive_duration, 1e-3));
  Eigen::Vector3d wrapped_goal = goal;
  wrapped_goal(2) = node.z(2) + wrap_angle(goal(2) - node.z(2));
  const Eigen::Vector3d d = wrapped_goal - node.z - node.z_dot * duration;
  const Eigen::Vector3d v = goal_dot - node.z_dot;
  const Eigen::Vector3d alpha =
      -12.0 * d / std::pow(duration, 3) + 6.0 * v / std::pow(duration, 2);
  const Eigen::Vector3d beta =
      6.0 * d / std::pow(duration, 2) - 2.0 * v / duration;
  const int steps = std::max({10, config.check_num, 2});
  out = LatticeNodeCpp();
  out.segment_path = Eigen::MatrixXd::Zero(steps + 1, 3);
  out.segment_z_dot = Eigen::MatrixXd::Zero(steps + 1, 3);
  out.segment_z_ddot = Eigen::MatrixXd::Zero(steps + 1, 3);
  out.segment_nu = Eigen::MatrixXd::Zero(steps + 1, 3);
  out.segment_times = Eigen::VectorXd::Zero(steps + 1);
  for (int i = 0; i <= steps; ++i) {
    const double t = duration * i / static_cast<double>(steps);
    Eigen::Vector3d z_t = cubic_position(node.z, node.z_dot, alpha, beta, t);
    z_t(2) = wrap_angle(z_t(2));
    const Eigen::Vector3d z_dot_t = node.z_dot + beta * t + 0.5 * alpha * t * t;
    const Eigen::Vector3d z_ddot_t = beta + alpha * t;
    out.segment_path.row(i) = z_t.transpose();
    out.segment_z_dot.row(i) = z_dot_t.transpose();
    out.segment_z_ddot.row(i) = z_ddot_t.transpose();
    out.segment_nu.row(i) = flat_to_body_velocity(z_t(2), z_dot_t).transpose();
    out.segment_times(i) = t;
  }
  if (!path_inside_local_horizon(root_z, out.segment_path, config)) {
    return false;
  }
  if (collides(out.segment_path.leftCols(2), config.obstacles,
               config.obstacle_margin)) {
    return false;
  }
  for (int i = 0; i <= steps; ++i) {
    const Eigen::Vector3d z_t = row_vec3(out.segment_path, i);
    const Eigen::Vector3d z_dot_t = row_vec3(out.segment_z_dot, i);
    const Eigen::Vector3d z_ddot_t = row_vec3(out.segment_z_ddot, i);
    if (!check_flat_bounds(z_dot_t, z_ddot_t, config)) {
      return false;
    }
    if (!check_usv_flat_feasible(z_t, z_dot_t, z_ddot_t, params, config)) {
      return false;
    }
  }
  out.z = row_vec3(out.segment_path, steps);
  out.z_dot = row_vec3(out.segment_z_dot, steps);
  out.z_ddot = row_vec3(out.segment_z_ddot, steps);
  out.nu = row_vec3(out.segment_nu, steps);
  out.g_cost = node.g_cost + ht.first;
  out.h_cost = 0.0;
  out.f_cost = out.g_cost;
  out.depth = node.depth + 1;
  out.primitive_u = Eigen::Vector2d::Zero();
  out.key = grid_key(out.z, out.nu, config);
  return true;
}

void build_primitive_segment(
    const LatticeNodeCpp &parent, const LatticeNodeCpp &node,
    const UsvParams &params, const FrontendConfig &config,
    Eigen::MatrixXd &path, Eigen::VectorXd &times) {
  const int steps = std::max(config.check_num, 1);
  const double duration = std::max(config.primitive_duration, 1e-6);
  path = Eigen::MatrixXd::Zero(steps + 1, 3);
  times = Eigen::VectorXd::Zero(steps + 1);
  Eigen::Matrix<double, 6, 1> state;
  state.segment<3>(0) = parent.z;
  state.segment<3>(3) = parent.nu;
  path.row(0) = parent.z.transpose();
  for (int i = 1; i <= steps; ++i) {
    state = rk4_usv_step(
        state, node.primitive_u, params,
        duration / static_cast<double>(steps));
    Eigen::Vector3d z_t = state.segment<3>(0);
    z_t(2) = wrap_angle(z_t(2));
    path.row(i) = z_t.transpose();
    times(i) = duration * static_cast<double>(i) /
               static_cast<double>(steps);
  }
}

void retrieve_path(
    const std::vector<LatticeNodeCpp> &nodes, int node_index,
    const UsvParams &params, const FrontendConfig &config,
    Eigen::MatrixXd &path, Eigen::VectorXd &path_times) {
  std::vector<int> chain;
  int current = node_index;
  while (current >= 0) {
    chain.push_back(current);
    current = nodes[current].parent;
  }
  std::reverse(chain.begin(), chain.end());
  int total_rows = 0;
  for (size_t i = 0; i < chain.size(); ++i) {
    const auto &node = nodes[chain[i]];
    int rows = 1;
    if (i > 0) {
      rows = node.segment_path.rows() > 0
                 ? node.segment_path.rows()
                 : std::max(config.check_num, 1) + 1;
    }
    if (i > 0) {
      rows -= 1;
    }
    total_rows += rows;
  }
  path = Eigen::MatrixXd::Zero(total_rows, 3);
  path_times = Eigen::VectorXd::Zero(total_rows);
  int cursor = 0;
  double offset = 0.0;
  for (size_t ci = 0; ci < chain.size(); ++ci) {
    const auto &node = nodes[chain[ci]];
    Eigen::MatrixXd segment_path;
    Eigen::VectorXd segment_times;
    if (ci == 0) {
      segment_path = Eigen::MatrixXd::Zero(1, 3);
      segment_times = Eigen::VectorXd::Zero(1);
      segment_path.row(0) = node.z.transpose();
    } else if (node.segment_path.rows() > 0) {
      segment_path = node.segment_path;
      segment_times = node.segment_times;
    } else {
      build_primitive_segment(
          nodes[node.parent], node, params, config, segment_path,
          segment_times);
    }
    const int start = ci > 0 ? 1 : 0;
    for (int r = start; r < segment_path.rows(); ++r) {
      path.row(cursor) = segment_path.row(r);
      path_times(cursor) = segment_times(r) + offset;
      cursor += 1;
    }
    offset = path_times(cursor - 1);
  }
}

Eigen::MatrixXd interpolate_path(
    const Eigen::MatrixXd &path, const Eigen::VectorXd &path_times,
    const Eigen::VectorXd &sample_times) {
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(sample_times.size(), 3);
  std::vector<double> yaw(path.rows());
  for (int i = 0; i < path.rows(); ++i) {
    yaw[i] = path(i, 2);
    if (i > 0) {
      while (yaw[i] - yaw[i - 1] > M_PI) yaw[i] -= 2.0 * M_PI;
      while (yaw[i] - yaw[i - 1] < -M_PI) yaw[i] += 2.0 * M_PI;
    }
  }
  for (int si = 0; si < sample_times.size(); ++si) {
    const double t = sample_times(si);
    int upper = 0;
    while (upper < path_times.size() - 1 && path_times(upper + 1) < t) {
      upper += 1;
    }
    const int next = std::min(upper + 1, static_cast<int>(path_times.size()) - 1);
    const double denom = std::max(path_times(next) - path_times(upper), 1e-12);
    const double a = std::clamp((t - path_times(upper)) / denom, 0.0, 1.0);
    out(si, 0) = (1.0 - a) * path(upper, 0) + a * path(next, 0);
    out(si, 1) = (1.0 - a) * path(upper, 1) + a * path(next, 1);
    out(si, 2) = wrap_angle((1.0 - a) * yaw[upper] + a * yaw[next]);
  }
  return out;
}

py::dict accepted_lattice_result(
    const std::vector<LatticeNodeCpp> &nodes, int node_index,
    const UsvParams &params,
    const FrontendConfig &config, const Eigen::Vector3d &global_goal,
    const Eigen::Vector3d &search_goal, bool terminal_stop,
    int candidate_count, int collision_rejects, int infeasible_rejects,
    int expansions, int analytic_attempts, bool analytic_success,
    bool reached_goal, bool reached_horizon_flag) {
  Eigen::MatrixXd path;
  Eigen::VectorXd path_times;
  retrieve_path(nodes, node_index, params, config, path, path_times);
  const auto &node = nodes[node_index];
  Eigen::Vector3d terminal_z = node.z;
  Eigen::Vector3d terminal_z_dot = node.z_dot;
  Eigen::Vector3d terminal_z_ddot = node.z_ddot;
  if (terminal_stop) {
    terminal_z = search_goal;
    terminal_z_dot.setZero();
    terminal_z_ddot.setZero();
  }
  const double total_time = std::max(path_times(path_times.size() - 1), 1e-6);
  const int selected_depth = std::max(
      static_cast<int>(std::round(total_time / std::max(config.primitive_duration, 1e-6))), 1);
  const int piece_min = std::max(config.minco_piece_count_min, 1);
  const int piece_max = std::max(config.minco_piece_count_max, piece_min);
  const int piece_count = std::min(std::max(selected_depth, piece_min), piece_max);
  Eigen::MatrixXd initial_inPs(0, 3);
  Eigen::VectorXd initial_ts(0);
  if (piece_count > 1) {
    Eigen::VectorXd sample_times(piece_count + 1);
    for (int i = 0; i <= piece_count; ++i) {
      sample_times(i) = path_times(0) + (path_times(path_times.size() - 1) -
                                         path_times(0)) * i / piece_count;
    }
    const Eigen::MatrixXd samples = interpolate_path(path, path_times, sample_times);
    initial_inPs = samples.block(1, 0, piece_count - 1, 3);
    initial_ts = Eigen::VectorXd::Constant(piece_count, total_time / piece_count);
  }
  Eigen::Vector3d first_tau = Eigen::Vector3d::Zero();
  std::vector<int> chain;
  for (int idx = node_index; idx >= 0; idx = nodes[idx].parent) {
    chain.push_back(idx);
  }
  std::reverse(chain.begin(), chain.end());
  for (const int idx : chain) {
    if (nodes[idx].parent >= 0) {
      if (nodes[idx].segment_path.rows() > 0 &&
          nodes[idx].segment_z_dot.rows() > 0 &&
          nodes[idx].segment_z_ddot.rows() > 0) {
        first_tau = theta_tau_nominal(
            row_vec3(nodes[idx].segment_path, 0),
            row_vec3(nodes[idx].segment_z_dot, 0),
            row_vec3(nodes[idx].segment_z_ddot, 0), params);
      } else {
        const auto &parent = nodes[nodes[idx].parent];
        const Eigen::Vector3d z_dot0 =
            body_to_flat_velocity(parent.z(2), parent.nu);
        const Eigen::Vector3d nu_dot0 =
            usv_body_acceleration(parent.nu, nodes[idx].primitive_u, params);
        const Eigen::Vector3d z_ddot0 =
            body_to_flat_acceleration(parent.z(2), parent.nu, nu_dot0);
        first_tau = theta_tau_nominal(parent.z, z_dot0, z_ddot0, params);
      }
      break;
    }
  }

  py::dict diagnostics;
  diagnostics["frontend_mode"] = 2.0;
  diagnostics["frontend_goal_distance"] = (global_goal.head<2>() - terminal_z.head<2>()).norm();
  const Eigen::Vector3d path_start = row_vec3(path, 0);
  diagnostics["frontend_goal_inside_horizon"] =
      goal_inside_local_horizon(path_start, global_goal, config) ? 1.0 : 0.0;
  diagnostics["frontend_goal_position_error"] =
      (global_goal.head<2>() - terminal_z.head<2>()).norm();
  diagnostics["frontend_goal_yaw_error"] =
      std::abs(wrap_angle(global_goal(2) - terminal_z(2)));
  diagnostics["frontend_local_goal_error"] =
      (search_goal.head<2>() - terminal_z.head<2>()).norm();
  diagnostics["frontend_selected_depth"] = static_cast<double>(node.depth);
  diagnostics["frontend_collision_free"] = 1.0;
  diagnostics["frontend_terminal_stop"] = terminal_stop ? 1.0 : 0.0;
  diagnostics["frontend_candidate_count"] = static_cast<double>(candidate_count);
  diagnostics["frontend_collision_reject_count"] = static_cast<double>(collision_rejects);
  diagnostics["frontend_infeasible_reject_count"] = static_cast<double>(infeasible_rejects);
  diagnostics["frontend_selected_cost"] = node.f_cost;
  diagnostics["frontend_expansion_count"] = static_cast<double>(expansions);
  diagnostics["frontend_analytic_attempted"] = analytic_attempts > 0 ? 1.0 : 0.0;
  diagnostics["frontend_analytic_attempt_count"] = static_cast<double>(analytic_attempts);
  diagnostics["frontend_analytic_success"] = analytic_success ? 1.0 : 0.0;
  diagnostics["frontend_reach_goal"] = reached_goal ? 1.0 : 0.0;
  diagnostics["frontend_reach_horizon"] = reached_horizon_flag ? 1.0 : 0.0;
  diagnostics["frontend_tau_v_bar"] = config.tau_v_bar;
  diagnostics["frontend_heuristic_weight"] = config.heuristic_weight;
  diagnostics["frontend_control_discretization"] = static_cast<double>(config.control_discretization);
  diagnostics["frontend_sample_thrust_limit"] = config.sample_thrust_limit;
  diagnostics["frontend_check_num"] = static_cast<double>(config.check_num);
  diagnostics["frontend_terminal_x"] = terminal_z(0);
  diagnostics["frontend_terminal_y"] = terminal_z(1);
  diagnostics["frontend_terminal_psi"] = terminal_z(2);
  diagnostics["frontend_global_goal_x"] = global_goal(0);
  diagnostics["frontend_global_goal_y"] = global_goal(1);
  diagnostics["frontend_global_goal_psi"] = global_goal(2);
  diagnostics["frontend_local_goal_x"] = search_goal(0);
  diagnostics["frontend_local_goal_y"] = search_goal(1);
  diagnostics["frontend_local_goal_psi"] = search_goal(2);

  py::dict result;
  result["accepted"] = true;
  result["terminal_z"] = vector_to_numpy(terminal_z);
  result["terminal_z_dot"] = vector_to_numpy(terminal_z_dot);
  result["terminal_z_ddot"] = vector_to_numpy(terminal_z_ddot);
  result["path"] = matrix_to_numpy(path);
  result["path_times"] = vector_to_numpy(path_times);
  result["first_tau"] = vector_to_numpy(first_tau);
  result["initial_inPs"] = matrix_to_numpy(initial_inPs);
  result["initial_ts"] = vector_to_numpy(initial_ts);
  result["diagnostics"] = diagnostics;
  return result;
}

py::dict rejected_lattice_result(
    const FrontendConfig &config, const Eigen::Vector3d &z,
    const Eigen::Vector3d &goal, const Eigen::Vector3d &search_goal,
    const LatticeNodeCpp &best_seen, int candidate_count, int collision_rejects,
    int infeasible_rejects, int expansions, int analytic_attempts) {
  py::dict diagnostics;
  diagnostics["frontend_mode"] = 2.0;
  diagnostics["frontend_goal_distance"] = (goal.head<2>() - z.head<2>()).norm();
  diagnostics["frontend_goal_inside_horizon"] =
      goal_inside_local_horizon(z, goal, config) ? 1.0 : 0.0;
  diagnostics["frontend_goal_position_error"] =
      (goal.head<2>() - best_seen.z.head<2>()).norm();
  diagnostics["frontend_goal_yaw_error"] =
      std::abs(wrap_angle(goal(2) - best_seen.z(2)));
  diagnostics["frontend_local_goal_error"] =
      (search_goal.head<2>() - best_seen.z.head<2>()).norm();
  diagnostics["frontend_selected_depth"] = static_cast<double>(best_seen.depth);
  diagnostics["frontend_collision_free"] = 0.0;
  diagnostics["frontend_terminal_stop"] = 0.0;
  diagnostics["frontend_candidate_count"] = static_cast<double>(candidate_count);
  diagnostics["frontend_collision_reject_count"] = static_cast<double>(collision_rejects);
  diagnostics["frontend_infeasible_reject_count"] = static_cast<double>(infeasible_rejects);
  diagnostics["frontend_selected_cost"] = best_seen.f_cost;
  diagnostics["frontend_expansion_count"] = static_cast<double>(expansions);
  diagnostics["frontend_analytic_attempted"] = analytic_attempts > 0 ? 1.0 : 0.0;
  diagnostics["frontend_analytic_attempt_count"] = static_cast<double>(analytic_attempts);
  diagnostics["frontend_analytic_success"] = 0.0;
  diagnostics["frontend_reach_goal"] = 0.0;
  diagnostics["frontend_reach_horizon"] = 0.0;
  diagnostics["frontend_tau_v_bar"] = config.tau_v_bar;
  diagnostics["frontend_heuristic_weight"] = config.heuristic_weight;
  diagnostics["frontend_control_discretization"] = static_cast<double>(config.control_discretization);
  diagnostics["frontend_sample_thrust_limit"] = config.sample_thrust_limit;
  diagnostics["frontend_check_num"] = static_cast<double>(config.check_num);
  diagnostics["frontend_terminal_x"] = z(0);
  diagnostics["frontend_terminal_y"] = z(1);
  diagnostics["frontend_terminal_psi"] = z(2);
  diagnostics["frontend_global_goal_x"] = goal(0);
  diagnostics["frontend_global_goal_y"] = goal(1);
  diagnostics["frontend_global_goal_psi"] = goal(2);
  diagnostics["frontend_local_goal_x"] = search_goal(0);
  diagnostics["frontend_local_goal_y"] = search_goal(1);
  diagnostics["frontend_local_goal_psi"] = search_goal(2);

  py::dict result;
  result["accepted"] = false;
  result["terminal_z"] = vector_to_numpy(z);
  result["terminal_z_dot"] = vector_to_numpy(Eigen::Vector3d::Zero());
  result["terminal_z_ddot"] = vector_to_numpy(Eigen::Vector3d::Zero());
  result["path"] = matrix_to_numpy(single_row_matrix(z));
  result["path_times"] = vector_to_numpy(Eigen::VectorXd::Zero(1));
  result["first_tau"] = vector_to_numpy(Eigen::Vector3d::Zero());
  result["initial_inPs"] = py::none();
  result["initial_ts"] = py::none();
  result["diagnostics"] = diagnostics;
  return result;
}

py::dict plan_lattice_terminal_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> z_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_dot_array,
    const py::dict &params_dict, const py::dict &config_dict) {
  const UsvParams params = params_from_dict(params_dict);
  const FrontendConfig config = frontend_config_from_dict(config_dict);
  const Eigen::Vector3d z = array_to_vec3(z_array, "z");
  const Eigen::Vector3d z_dot = array_to_vec3(z_dot_array, "z_dot");
  Eigen::Vector3d goal = config.goal_z;
  goal(2) = z(2) + wrap_angle(goal(2) - z(2));
  const Eigen::Vector3d search_goal = goal;
  const Eigen::Vector3d search_goal_dot = Eigen::Vector3d::Zero();
  const bool terminal_stop = true;
  const double global_goal_distance = (goal.head<2>() - z.head<2>()).norm();
  (void)global_goal_distance;

  const auto controls = discretized_generalized_inputs(params, config);
  const std::size_t max_node_count = std::max<std::size_t>(
      4096, std::min<std::size_t>(
                1000000,
                2 + static_cast<std::size_t>(std::max(config.max_expansions, 1)) *
                        std::max<std::size_t>(controls.size(), 1)));
  std::vector<LatticeNodeCpp> nodes(max_node_count);
  int node_count = 0;
  auto allocate_node = [&]() -> int {
    if (node_count >= static_cast<int>(nodes.size())) {
      return -1;
    }
    return node_count++;
  };

  const int root_index = allocate_node();
  LatticeNodeCpp &root = nodes[root_index];
  root.z = z;
  root.z_dot = z_dot;
  root.nu = flat_to_body_velocity(z(2), z_dot);
  const double heuristic_weight = std::max(config.heuristic_weight, 0.0);
  root.h_cost = heuristic_cost(z, z_dot, search_goal, search_goal_dot, config).first;
  root.f_cost = heuristic_weight * root.h_cost;
  root.segment_path = Eigen::MatrixXd::Zero(1, 3);
  root.segment_z_dot = Eigen::MatrixXd::Zero(1, 3);
  root.segment_z_ddot = Eigen::MatrixXd::Zero(1, 3);
  root.segment_nu = Eigen::MatrixXd::Zero(1, 3);
  root.segment_times = Eigen::VectorXd::Zero(1);
  root.segment_path.row(0) = z.transpose();
  root.segment_z_dot.row(0) = z_dot.transpose();
  root.segment_nu.row(0) = root.nu.transpose();
  root.key = grid_key(root.z, root.nu, config);
  root.status = LatticeNodeCpp::OPENSET;

  struct QueueItem {
    double f;
    int serial;
    int index;
    bool operator>(const QueueItem &other) const {
      if (f == other.f) {
        return serial > other.serial;
      }
      return f > other.f;
    }
  };
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> open;
  std::unordered_map<std::uint64_t, int> best_by_key;
  best_by_key.reserve(max_node_count);
  std::unordered_map<std::uint64_t, std::pair<double, double>> heuristic_cache;
  heuristic_cache.reserve(max_node_count);
  open.push({root.f_cost, 0, root_index});
  best_by_key[root.key] = root_index;
  heuristic_cache[root.key] = {root.h_cost, 0.0};
  int serial = 1;
  int candidate_count = 0;
  int collision_rejects = 0;
  int infeasible_rejects = 0;
  int expansions = 0;
  int analytic_attempts = 0;
  int best_seen = root_index;
  const bool analytic_allowed = goal_inside_local_horizon(z, search_goal, config);

  while (!open.empty() && expansions < config.max_expansions) {
    const auto item = open.top();
    open.pop();
    const int current_index = item.index;
    if (current_index < 0 || current_index >= node_count) {
      continue;
    }
    LatticeNodeCpp &current = nodes[current_index];
    if (current.status == LatticeNodeCpp::CLOSED ||
        item.f > current.f_cost + 1e-9) {
      continue;
    }
    current.status = LatticeNodeCpp::CLOSED;
    expansions += 1;
    if (current.f_cost < nodes[best_seen].f_cost) {
      best_seen = current_index;
    }

    if (config.analytic_expansion && analytic_allowed) {
      analytic_attempts += 1;
      LatticeNodeCpp analytic;
      if (analytic_expand(current, search_goal, search_goal_dot, z, params,
                          config, analytic)) {
        analytic.parent = current_index;
        analytic.status = LatticeNodeCpp::CLOSED;
        const int analytic_index = allocate_node();
        if (analytic_index < 0) {
          break;
        }
        nodes[analytic_index] = analytic;
        return accepted_lattice_result(
            nodes, analytic_index, params, config, goal, search_goal,
            terminal_stop, candidate_count, collision_rejects, infeasible_rejects,
            expansions, analytic_attempts, true, false, false);
      }
    }

    if (reach_goal(current.z, search_goal, config)) {
      return accepted_lattice_result(
          nodes, current_index, params, config, goal, search_goal, false, candidate_count,
          collision_rejects, infeasible_rejects, expansions, analytic_attempts,
          false, true, false);
    }
    if (!analytic_allowed &&
        reach_horizon_toward_goal(z, current.z, search_goal, config)) {
      return accepted_lattice_result(
          nodes, current_index, params, config, goal, search_goal, false, candidate_count,
          collision_rejects, infeasible_rejects, expansions, analytic_attempts,
          false, false, true);
    }
    if (current.depth >= config.search_depth) {
      continue;
    }

    for (const auto &control : controls) {
      candidate_count += 1;
      LatticeNodeCpp child;
      bool collision_reject = false;
      if (!rollout_usv_primitive(
              current, control, params, config, child, collision_reject)) {
        if (collision_reject) {
          collision_rejects += 1;
        } else {
          infeasible_rejects += 1;
        }
        continue;
      }
      child.parent = current_index;
      const double edge = edge_cost(control, params, config);
      child.g_cost = current.g_cost + edge;
      auto existing = best_by_key.find(child.key);
      if (existing != best_by_key.end()) {
        const int existing_index = existing->second;
        if (nodes[existing_index].status == LatticeNodeCpp::CLOSED ||
            child.g_cost >= nodes[existing_index].g_cost) {
          continue;
        }
      }
      auto h_it = heuristic_cache.find(child.key);
      if (h_it == heuristic_cache.end()) {
        h_it = heuristic_cache.emplace(
            child.key,
            heuristic_cost(child.z, child.z_dot, search_goal,
                           search_goal_dot, config)).first;
      }
      const auto h = h_it->second;
      child.h_cost = h.first;
      child.f_cost = child.g_cost + heuristic_weight * child.h_cost;
      if (existing != best_by_key.end()) {
        const int existing_index = existing->second;
        child.status = LatticeNodeCpp::OPENSET;
        nodes[existing_index] = child;
        open.push({nodes[existing_index].f_cost, serial++, existing_index});
        continue;
      }
      const int child_index = allocate_node();
      if (child_index < 0) {
        continue;
      }
      child.status = LatticeNodeCpp::OPENSET;
      nodes[child_index] = child;
      best_by_key[child.key] = child_index;
      open.push({nodes[child_index].f_cost, serial++, child_index});
    }
  }

  return rejected_lattice_result(
      config, z, goal, search_goal, nodes[best_seen], candidate_count,
      collision_rejects, infeasible_rejects, expansions, analytic_attempts);
}

Eigen::MatrixXd dA_dT_times_coefficients(
    const MincoTrajectory &traj, int piece_index) {
  const int M = traj.piece_count;
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(M * kCoeff, kDim);
  const Eigen::Matrix<double, kCoeff, kDim> coeff =
      traj.coeffs.block(piece_index * kCoeff, 0, kCoeff, kDim);
  if (piece_index < M - 1) {
    const int row = 3 + 6 * piece_index;
    const double T = traj.ts(piece_index);
    const int offsets[5] = {0, 2, 3, 4, 5};
    for (int derivative = 0; derivative < 5; ++derivative) {
      result.row(row + offsets[derivative]) =
          basis(T, derivative + 1).transpose() * coeff;
    }
  }
  if (piece_index == M - 1) {
    const int row = 3 + 6 * (M - 1);
    const double T = traj.ts(piece_index);
    for (int derivative = 0; derivative < 3; ++derivative) {
      result.row(row + derivative) =
          basis(T, derivative + 1).transpose() * coeff;
    }
  }
  return result;
}

double dA_dT_adjoint_dot(
    const MincoTrajectory &traj, int piece_index,
    const Eigen::MatrixXd &adjoint) {
  const int M = traj.piece_count;
  const Eigen::Matrix<double, kCoeff, kDim> coeff =
      traj.coeffs.block(piece_index * kCoeff, 0, kCoeff, kDim);
  double total = 0.0;
  if (piece_index < M - 1) {
    const int row = 3 + 6 * piece_index;
    const double T = traj.ts(piece_index);
    const int offsets[5] = {0, 2, 3, 4, 5};
    for (int derivative = 0; derivative < 5; ++derivative) {
      const Eigen::RowVector3d dA_row =
          basis(T, derivative + 1).transpose() * coeff;
      total += adjoint.row(row + offsets[derivative]).dot(dA_row);
    }
  }
  if (piece_index == M - 1) {
    const int row = 3 + 6 * (M - 1);
    const double T = traj.ts(piece_index);
    for (int derivative = 0; derivative < 3; ++derivative) {
      const Eigen::RowVector3d dA_row =
          basis(T, derivative + 1).transpose() * coeff;
      total += adjoint.row(row + derivative).dot(dA_row);
    }
  }
  return total;
}

struct ObjectiveEvalResult {
  double objective = 0.0;
  double energy = 0.0;
  Eigen::VectorXd gradient;
  Eigen::MatrixXd inPs;
  Eigen::VectorXd theta;
  Eigen::VectorXd ts;
  PenaltyResult penalty_result;
};

ObjectiveEvalResult evaluate_minco_objective(
    const Eigen::VectorXd &variables,
    const Eigen::Matrix<double, 3, 3> &head_pva,
    const Eigen::Matrix<double, 3, 3> &tail_pva,
    const OptimizerConfig &opt,
    const PenaltyConfig &penalty,
    const std::vector<double> *quadrature_alphas = nullptr,
    const std::vector<double> *quadrature_weights = nullptr) {
  ObjectiveEvalResult result;
  unpack_variables(variables, opt, result.inPs, result.theta);
  result.ts = theta_to_times(result.theta, opt);

  MincoTrajectory traj;
  traj.piece_count = opt.piece_count;
  traj.head_pva = head_pva;
  traj.tail_pva = tail_pva;
  traj.inPs = result.inPs;
  traj.ts = result.ts;
  build_minco_system(traj);

  result.energy = trajectory_energy(traj);
  Eigen::MatrixXd energy_coeff_grad;
  Eigen::VectorXd energy_time_grad;
  energy_gradients(traj, energy_coeff_grad, energy_time_grad);
  if (quadrature_alphas != nullptr && quadrature_weights != nullptr) {
    penalty_and_gradients_with_quadrature(
        traj, penalty, *quadrature_alphas, *quadrature_weights,
        result.penalty_result);
  } else {
    result.penalty_result = penalty_and_gradients(traj, penalty);
  }

  Eigen::MatrixXd coeff_grad =
      opt.smooth_weight * energy_coeff_grad + result.penalty_result.coeff_grad;
  Eigen::VectorXd time_grad =
      opt.smooth_weight * energy_time_grad + result.penalty_result.time_grad;
  if (!opt.fixed_total_time_enabled) {
    time_grad.array() += opt.time_weight;
  }

  const Eigen::MatrixXd adjoint =
      traj.A.transpose().partialPivLu().solve(coeff_grad);

  Eigen::MatrixXd grad_inPs =
      Eigen::MatrixXd::Zero(std::max(opt.piece_count - 1, 0), kDim);
  int row = 3;
  for (int joint = 0; joint < opt.piece_count - 1; ++joint) {
    grad_inPs.row(joint) = adjoint.row(row) + adjoint.row(row + 1);
    row += 6;
  }

  Eigen::VectorXd grad_ts = time_grad;
  for (int piece = 0; piece < opt.piece_count; ++piece) {
    grad_ts(piece) -= dA_dT_adjoint_dot(traj, piece, adjoint);
  }
  const Eigen::VectorXd grad_theta =
      times_gradient_to_theta_gradient(grad_ts, result.ts, opt);

  result.objective = opt.smooth_weight * result.energy + result.penalty_result.total;
  if (!opt.fixed_total_time_enabled) {
    result.objective += opt.time_weight * result.ts.sum();
  }
  result.gradient = pack_variables(grad_inPs, grad_theta);
  return result;
}

py::dict diagnostics_dict(const ObjectiveEvalResult &eval) {
  const PenaltyResult &penalty_result = eval.penalty_result;
  py::dict diagnostics;
  diagnostics["max_abs_tau_v"] = penalty_result.max_abs_tau_v;
  diagnostics["rms_tau_v"] =
      penalty_result.metric_count > 0
          ? std::sqrt(penalty_result.tau_v_sq_sum /
                      static_cast<double>(penalty_result.metric_count))
          : 0.0;
  diagnostics["velocity_violation"] = penalty_result.velocity_violation;
  diagnostics["acceleration_violation"] = penalty_result.acceleration_violation;
  diagnostics["actuator_bound_violation"] = penalty_result.actuator_violation;
  diagnostics["max_abs_tau_u"] = penalty_result.max_abs_tau_u;
  diagnostics["max_abs_tau_r"] = penalty_result.max_abs_tau_r;
  diagnostics["max_left_thrust"] = penalty_result.max_left;
  diagnostics["min_left_thrust"] = penalty_result.min_left;
  diagnostics["max_right_thrust"] = penalty_result.max_right;
  diagnostics["min_right_thrust"] = penalty_result.min_right;
  diagnostics["penalty_total"] = penalty_result.total;
  diagnostics["energy"] = eval.energy;
  return diagnostics;
}

struct VariableBounds {
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

double shortest_yaw_delta(double start, double goal) {
  return std::atan2(std::sin(goal - start), std::cos(goal - start));
}

VariableBounds variable_bounds(
    const Eigen::Matrix<double, 3, 3> &head_pva,
    const Eigen::Matrix<double, 3, 3> &tail_pva,
    const OptimizerConfig &opt) {
  const int point_size = std::max(opt.piece_count - 1, 0) * kDim;
  const int variable_size = point_size + opt.piece_count;
  VariableBounds bounds;
  bounds.lower = Eigen::VectorXd::Constant(
      variable_size, -std::numeric_limits<double>::infinity());
  bounds.upper = Eigen::VectorXd::Constant(
      variable_size, std::numeric_limits<double>::infinity());

  const Eigen::Vector3d start = head_pva.row(0).transpose();
  Eigen::Vector3d goal = tail_pva.row(0).transpose();
  goal(2) = start(2) + shortest_yaw_delta(start(2), goal(2));
  Eigen::Vector3d lower = start.cwiseMin(goal);
  Eigen::Vector3d upper = start.cwiseMax(goal);
  const double spatial_margin = std::max(opt.spatial_margin, 0.0);
  const double yaw_margin = std::max(opt.yaw_margin, 0.0);
  lower.head<2>().array() -= spatial_margin;
  upper.head<2>().array() += spatial_margin;
  lower(2) -= yaw_margin;
  upper(2) += yaw_margin;

  int index = 0;
  for (int row = 0; row < opt.piece_count - 1; ++row) {
    for (int axis = 0; axis < kDim; ++axis) {
      bounds.lower(index) = lower(axis);
      bounds.upper(index) = upper(axis);
      ++index;
    }
  }
  const double theta_bound = std::max(opt.theta_bound, 1.0);
  for (int i = 0; i < opt.piece_count; ++i) {
    bounds.lower(point_size + i) = -theta_bound;
    bounds.upper(point_size + i) = theta_bound;
  }
  return bounds;
}

void clip_to_bounds(Eigen::VectorXd &x, const VariableBounds &bounds) {
  constexpr double eps = 1e-10;
  for (int i = 0; i < x.size(); ++i) {
    if (std::isfinite(bounds.lower(i))) {
      x(i) = std::max(x(i), bounds.lower(i) + eps);
    }
    if (std::isfinite(bounds.upper(i))) {
      x(i) = std::min(x(i), bounds.upper(i) - eps);
    }
  }
}

struct LbfgsMincoContext {
  Eigen::Matrix<double, 3, 3> head_pva;
  Eigen::Matrix<double, 3, 3> tail_pva;
  OptimizerConfig opt;
  PenaltyConfig penalty;
  VariableBounds bounds;
  std::vector<double> quadrature_alphas;
  std::vector<double> quadrature_weights;
  int eval_count = 0;
  int iterations = 0;
  int line_search_evals = 0;
  double last_objective = 0.0;
  double last_grad_inf_norm = 0.0;
  std::string failure_message;
};

double lbfgs_minco_evaluate(
    void *instance, const Eigen::VectorXd &x, Eigen::VectorXd &g) {
  auto *context = static_cast<LbfgsMincoContext *>(instance);
  context->eval_count += 1;
  try {
    const ObjectiveEvalResult eval = evaluate_minco_objective(
        x, context->head_pva, context->tail_pva, context->opt, context->penalty,
        &context->quadrature_alphas, &context->quadrature_weights);
    g = eval.gradient;
    context->last_objective = eval.objective;
    context->last_grad_inf_norm =
        g.size() > 0 ? g.cwiseAbs().maxCoeff() : 0.0;
    if (!std::isfinite(eval.objective) || !g.allFinite()) {
      context->failure_message = "Objective or gradient became non-finite.";
      g.setZero(x.size());
      return 1e100;
    }
    return eval.objective;
  } catch (const std::exception &error) {
    context->failure_message = error.what();
    g.setZero(x.size());
    return 1e100;
  }
}

double lbfgs_minco_stepbound(
    void *instance, const Eigen::VectorXd &xp, const Eigen::VectorXd &d) {
  const auto *context = static_cast<LbfgsMincoContext *>(instance);
  double step_bound = std::numeric_limits<double>::infinity();
  for (int i = 0; i < xp.size(); ++i) {
    if (d(i) > 0.0 && std::isfinite(context->bounds.upper(i))) {
      step_bound = std::min(step_bound, (context->bounds.upper(i) - xp(i)) / d(i));
    } else if (d(i) < 0.0 && std::isfinite(context->bounds.lower(i))) {
      step_bound = std::min(step_bound, (context->bounds.lower(i) - xp(i)) / d(i));
    }
  }
  if (!std::isfinite(step_bound)) {
    return 1e20;
  }
  return std::max(1e-10, 0.999 * step_bound);
}

int lbfgs_minco_progress(
    void *instance,
    const Eigen::VectorXd &,
    const Eigen::VectorXd &g,
    const double fx,
    const double,
    const int k,
    const int ls) {
  auto *context = static_cast<LbfgsMincoContext *>(instance);
  context->iterations = k;
  context->line_search_evals += ls;
  context->last_objective = fx;
  context->last_grad_inf_norm = g.size() > 0 ? g.cwiseAbs().maxCoeff() : 0.0;
  return 0;
}

py::dict minco_objective_gradient(
    py::array_t<double, py::array::c_style | py::array::forcecast> variables_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> head_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> tail_array,
    const py::dict &params_dict,
    const py::dict &optimizer_dict,
    const py::dict &penalty_dict) {
  const OptimizerConfig opt = optimizer_config_from_dict(optimizer_dict);
  const PenaltyConfig penalty = penalty_config_from_dict(penalty_dict, params_dict);

  const Eigen::VectorXd variables = array_to_vector(variables_array);
  const Eigen::Matrix<double, 3, 3> head_pva =
      array_to_matrix(head_array, 3, 3, "head_pva");
  const Eigen::Matrix<double, 3, 3> tail_pva =
      array_to_matrix(tail_array, 3, 3, "tail_pva");
  const ObjectiveEvalResult eval =
      evaluate_minco_objective(variables, head_pva, tail_pva, opt, penalty);

  py::dict result;
  result["objective"] = eval.objective;
  result["gradient"] = vector_to_numpy(eval.gradient);
  result["diagnostics"] = diagnostics_dict(eval);
  return result;
}

py::dict minco_optimize_lbfgs(
    py::array_t<double, py::array::c_style | py::array::forcecast> variables_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> head_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> tail_array,
    const py::dict &params_dict,
    const py::dict &optimizer_dict,
    const py::dict &penalty_dict) {
  LbfgsMincoContext context;
  context.opt = optimizer_config_from_dict(optimizer_dict);
  context.penalty = penalty_config_from_dict(penalty_dict, params_dict);
  context.head_pva = array_to_matrix(head_array, 3, 3, "head_pva");
  context.tail_pva = array_to_matrix(tail_array, 3, 3, "tail_pva");
  context.bounds = variable_bounds(context.head_pva, context.tail_pva, context.opt);
  gauss_legendre(
      context.penalty.quadrature_order,
      context.quadrature_alphas,
      context.quadrature_weights);

  Eigen::VectorXd variables = array_to_vector(variables_array);
  clip_to_bounds(variables, context.bounds);
  const ObjectiveEvalResult before = evaluate_minco_objective(
      variables, context.head_pva, context.tail_pva, context.opt, context.penalty,
      &context.quadrature_alphas, &context.quadrature_weights);

  lbfgs::lbfgs_parameter_t params;
  params.mem_size = 256;
  params.past = 3;
  params.min_step = 1.0e-32;
  params.g_epsilon = std::max(0.0, context.opt.gradient_tolerance);
  params.delta = std::max(0.0, context.opt.function_tolerance);
  params.max_iterations = std::max(0, context.opt.max_iterations);

  double final_objective = before.objective;
  int status_code = 0;
  {
    py::gil_scoped_release release;
    status_code = lbfgs::lbfgs_optimize(
        variables,
        final_objective,
        lbfgs_minco_evaluate,
        lbfgs_minco_stepbound,
        lbfgs_minco_progress,
        &context,
        params);
  }

  clip_to_bounds(variables, context.bounds);
  const ObjectiveEvalResult after = evaluate_minco_objective(
      variables, context.head_pva, context.tail_pva, context.opt, context.penalty,
      &context.quadrature_alphas, &context.quadrature_weights);

  py::dict diagnostics = diagnostics_dict(after);
  diagnostics["optimizer_backend"] = "cpp_lbfgs_lite";
  diagnostics["optimizer_eval_count"] = context.eval_count;
  diagnostics["optimizer_line_search_evals"] = context.line_search_evals;
  diagnostics["optimizer_grad_inf_norm"] =
      after.gradient.size() > 0 ? after.gradient.cwiseAbs().maxCoeff() : 0.0;
  diagnostics["optimizer_status_code"] = status_code;

  py::dict result;
  result["success"] = status_code >= 0;
  result["status_code"] = status_code;
  result["message"] =
      context.failure_message.empty()
          ? std::string(lbfgs::lbfgs_strerror(status_code))
          : context.failure_message;
  result["objective_before"] = before.objective;
  result["objective_after"] = after.objective;
  result["optimizer_fun"] = final_objective;
  result["iterations"] = context.iterations;
  result["eval_count"] = context.eval_count;
  result["variables"] = vector_to_numpy(variables);
  result["gradient"] = vector_to_numpy(after.gradient);
  result["inPs"] = matrix_to_numpy(after.inPs);
  result["theta"] = vector_to_numpy(after.theta);
  result["ts"] = vector_to_numpy(after.ts);
  result["diagnostics"] = diagnostics;
  return result;
}

double trajectory_duration(const MincoTrajectory &traj) {
  return traj.ts.size() > 0 ? traj.ts.sum() : 0.0;
}

MincoTrajectory build_trajectory_from_eval(
    const ObjectiveEvalResult &eval,
    const Eigen::Matrix<double, 3, 3> &head_pva,
    const Eigen::Matrix<double, 3, 3> &tail_pva,
    const OptimizerConfig &opt) {
  MincoTrajectory traj;
  traj.piece_count = opt.piece_count;
  traj.head_pva = head_pva;
  traj.tail_pva = tail_pva;
  traj.inPs = eval.inPs;
  traj.ts = eval.ts;
  build_minco_system(traj);
  return traj;
}

Eigen::Vector3d sample_piece_derivative(
    const MincoTrajectory &traj, int piece, double local_time, int derivative) {
  const Eigen::Matrix<double, kCoeff, kDim> coeff =
      traj.coeffs.block(piece * kCoeff, 0, kCoeff, kDim);
  return coeff.transpose() * basis(local_time, derivative);
}

Eigen::Vector3d sample_trajectory_derivative(
    const MincoTrajectory &traj, double query_time, int derivative) {
  if (traj.piece_count <= 0) {
    throw std::runtime_error("Cannot sample an empty MINCO trajectory.");
  }
  double remaining = std::clamp(query_time, 0.0, trajectory_duration(traj));
  for (int piece = 0; piece < traj.piece_count; ++piece) {
    const double T = traj.ts(piece);
    if (remaining <= T || piece == traj.piece_count - 1) {
      return sample_piece_derivative(
          traj, piece, std::clamp(remaining, 0.0, T), derivative);
    }
    remaining -= T;
  }
  return sample_piece_derivative(
      traj, traj.piece_count - 1, traj.ts(traj.piece_count - 1), derivative);
}

Eigen::VectorXd sample_times(double duration, double dt) {
  if (dt <= 0.0) {
    throw std::runtime_error("dt must be positive.");
  }
  if (duration < 0.0) {
    throw std::runtime_error("duration must be nonnegative.");
  }
  if (duration <= 0.0) {
    return Eigen::VectorXd::Zero(1);
  }
  const int base_steps = static_cast<int>(std::floor(duration / dt)) + 1;
  const bool append_final =
      (static_cast<double>(base_steps - 1) * dt) < duration - 1e-12;
  const int count = base_steps + (append_final ? 1 : 0);
  Eigen::VectorXd times(count);
  for (int i = 0; i < base_steps; ++i) {
    times(i) = static_cast<double>(i) * dt;
  }
  times(count - 1) = duration;
  return times;
}

py::dict minco_residuals_dict(
    const MincoTrajectory &traj,
    const Eigen::Matrix<double, 3, 3> &head_pva,
    const Eigen::Matrix<double, 3, 3> &tail_pva) {
  const Eigen::Vector3d start_z = sample_trajectory_derivative(traj, 0.0, 0);
  const Eigen::Vector3d start_z_dot = sample_trajectory_derivative(traj, 0.0, 1);
  const Eigen::Vector3d start_z_ddot = sample_trajectory_derivative(traj, 0.0, 2);
  const double duration = trajectory_duration(traj);
  const Eigen::Vector3d end_z = sample_trajectory_derivative(traj, duration, 0);
  const Eigen::Vector3d end_z_dot = sample_trajectory_derivative(traj, duration, 1);
  const Eigen::Vector3d end_z_ddot = sample_trajectory_derivative(traj, duration, 2);
  double pva_residual = 0.0;
  pva_residual = std::max(pva_residual, (start_z - head_pva.row(0).transpose()).norm());
  pva_residual = std::max(pva_residual, (start_z_dot - head_pva.row(1).transpose()).norm());
  pva_residual = std::max(pva_residual, (start_z_ddot - head_pva.row(2).transpose()).norm());
  pva_residual = std::max(pva_residual, (end_z - tail_pva.row(0).transpose()).norm());
  pva_residual = std::max(pva_residual, (end_z_dot - tail_pva.row(1).transpose()).norm());
  pva_residual = std::max(pva_residual, (end_z_ddot - tail_pva.row(2).transpose()).norm());

  double intermediate_residual = 0.0;
  double z_dot_continuity_residual = 0.0;
  double z_ddot_continuity_residual = 0.0;
  double z_jerk_continuity_residual = 0.0;
  double z_snap_continuity_residual = 0.0;
  for (int joint = 0; joint < traj.piece_count - 1; ++joint) {
    const double T = traj.ts(joint);
    const Eigen::Vector3d point = traj.inPs.row(joint).transpose();
    const Eigen::Vector3d left_z =
        sample_piece_derivative(traj, joint, T, 0);
    const Eigen::Vector3d right_z =
        sample_piece_derivative(traj, joint + 1, 0.0, 0);
    intermediate_residual = std::max(intermediate_residual, (left_z - point).norm());
    intermediate_residual = std::max(intermediate_residual, (right_z - point).norm());

    const Eigen::Vector3d left_z_dot = sample_piece_derivative(traj, joint, T, 1);
    const Eigen::Vector3d right_z_dot = sample_piece_derivative(traj, joint + 1, 0.0, 1);
    const Eigen::Vector3d left_z_ddot = sample_piece_derivative(traj, joint, T, 2);
    const Eigen::Vector3d right_z_ddot = sample_piece_derivative(traj, joint + 1, 0.0, 2);
    const Eigen::Vector3d left_z_jerk = sample_piece_derivative(traj, joint, T, 3);
    const Eigen::Vector3d right_z_jerk = sample_piece_derivative(traj, joint + 1, 0.0, 3);
    const Eigen::Vector3d left_z_snap = sample_piece_derivative(traj, joint, T, 4);
    const Eigen::Vector3d right_z_snap = sample_piece_derivative(traj, joint + 1, 0.0, 4);
    z_dot_continuity_residual =
        std::max(z_dot_continuity_residual, (left_z_dot - right_z_dot).norm());
    z_ddot_continuity_residual =
        std::max(z_ddot_continuity_residual, (left_z_ddot - right_z_ddot).norm());
    z_jerk_continuity_residual =
        std::max(z_jerk_continuity_residual, (left_z_jerk - right_z_jerk).norm());
    z_snap_continuity_residual =
        std::max(z_snap_continuity_residual, (left_z_snap - right_z_snap).norm());
  }
  const double continuity_residual = std::max({
      z_dot_continuity_residual,
      z_ddot_continuity_residual,
      z_jerk_continuity_residual,
      z_snap_continuity_residual,
  });

  py::dict out;
  out["pva_residual"] = pva_residual;
  out["intermediate_residual"] = intermediate_residual;
  out["continuity_residual"] = continuity_residual;
  out["z_dot_continuity_residual"] = z_dot_continuity_residual;
  out["z_ddot_continuity_residual"] = z_ddot_continuity_residual;
  out["z_jerk_continuity_residual"] = z_jerk_continuity_residual;
  out["z_snap_continuity_residual"] = z_snap_continuity_residual;
  return out;
}

py::dict sample_minco_reference(
    const MincoTrajectory &traj, const UsvParams &params, double dt,
    Eigen::VectorXd &times, Eigen::MatrixXd &z, Eigen::MatrixXd &z_dot,
    Eigen::MatrixXd &z_ddot, Eigen::MatrixXd &nu, Eigen::MatrixXd &nu_dot,
    Eigen::MatrixXd &tau, Eigen::MatrixXd &thrust) {
  times = sample_times(trajectory_duration(traj), dt);
  const int count = static_cast<int>(times.size());
  z = Eigen::MatrixXd::Zero(count, 3);
  z_dot = Eigen::MatrixXd::Zero(count, 3);
  z_ddot = Eigen::MatrixXd::Zero(count, 3);
  nu = Eigen::MatrixXd::Zero(count, 3);
  nu_dot = Eigen::MatrixXd::Zero(count, 3);
  tau = Eigen::MatrixXd::Zero(count, 3);
  thrust = Eigen::MatrixXd::Zero(count, 2);

  double max_abs_tau_v = 0.0;
  double tau_v_sq_sum = 0.0;
  double max_abs_tau_u = 0.0;
  double max_abs_tau_r = 0.0;
  double max_abs_left = 0.0;
  double max_abs_right = 0.0;
  double max_speed = 0.0;
  double max_yaw_rate = 0.0;

  for (int i = 0; i < count; ++i) {
    const double query_time = times(i);
    const Eigen::Vector3d z_i = sample_trajectory_derivative(traj, query_time, 0);
    const Eigen::Vector3d z_dot_i = sample_trajectory_derivative(traj, query_time, 1);
    const Eigen::Vector3d z_ddot_i = sample_trajectory_derivative(traj, query_time, 2);
    const Eigen::Vector3d nu_i = flat_to_body_velocity(z_i(2), z_dot_i);
    const Eigen::Vector3d nu_dot_i = flat_to_body_acceleration(
        z_i(2), z_dot_i, z_ddot_i);
    const Eigen::Vector3d tau_i = theta_tau_nominal(z_i, z_dot_i, z_ddot_i, params);
    const auto thrust_i = generalized_force_to_thrust(tau_i(0), tau_i(2), params);

    z.row(i) = z_i.transpose();
    z_dot.row(i) = z_dot_i.transpose();
    z_ddot.row(i) = z_ddot_i.transpose();
    nu.row(i) = nu_i.transpose();
    nu_dot.row(i) = nu_dot_i.transpose();
    tau.row(i) = tau_i.transpose();
    thrust(i, 0) = thrust_i.first;
    thrust(i, 1) = thrust_i.second;

    max_abs_tau_v = std::max(max_abs_tau_v, std::abs(tau_i(1)));
    tau_v_sq_sum += tau_i(1) * tau_i(1);
    max_abs_tau_u = std::max(max_abs_tau_u, std::abs(tau_i(0)));
    max_abs_tau_r = std::max(max_abs_tau_r, std::abs(tau_i(2)));
    max_abs_left = std::max(max_abs_left, std::abs(thrust_i.first));
    max_abs_right = std::max(max_abs_right, std::abs(thrust_i.second));
    max_speed = std::max(max_speed, z_dot_i.head<2>().norm());
    max_yaw_rate = std::max(max_yaw_rate, std::abs(nu_i(2)));
  }

  py::dict diagnostics;
  diagnostics["max_abs_tau_v"] = max_abs_tau_v;
  diagnostics["rms_tau_v"] =
      count > 0 ? std::sqrt(tau_v_sq_sum / static_cast<double>(count)) : 0.0;
  diagnostics["max_abs_tau_u"] = max_abs_tau_u;
  diagnostics["max_abs_tau_r"] = max_abs_tau_r;
  diagnostics["max_abs_left_thrust"] = max_abs_left;
  diagnostics["max_abs_right_thrust"] = max_abs_right;
  diagnostics["max_speed"] = max_speed;
  diagnostics["max_yaw_rate"] = max_yaw_rate;
  diagnostics["final_speed"] =
      count > 0 ? z_dot.row(count - 1).head<2>().norm() : 0.0;
  diagnostics["final_yaw_rate"] =
      count > 0 ? std::abs(nu(count - 1, 2)) : 0.0;
  return diagnostics;
}

void dict_update(py::dict &target, const py::dict &source) {
  for (const auto item : source) {
    target[item.first] = item.second;
  }
}

py::dict plan_minco_reference_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> head_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> tail_array,
    const py::dict &params_dict,
    const py::dict &optimizer_dict,
    const py::dict &penalty_dict,
    double dt,
    py::object initial_inPs_obj = py::none(),
    py::object initial_ts_obj = py::none()) {
  const auto wall_start = std::chrono::steady_clock::now();
  LbfgsMincoContext context;
  context.opt = optimizer_config_from_dict(optimizer_dict);
  context.penalty = penalty_config_from_dict(penalty_dict, params_dict);
  context.head_pva = array_to_matrix(head_array, 3, 3, "head_pva");
  context.tail_pva = array_to_matrix(tail_array, 3, 3, "tail_pva");
  context.bounds = variable_bounds(context.head_pva, context.tail_pva, context.opt);
  gauss_legendre(
      context.penalty.quadrature_order,
      context.quadrature_alphas,
      context.quadrature_weights);

  Eigen::MatrixXd inPs0;
  Eigen::VectorXd ts0;
  const bool has_initial_inPs = !initial_inPs_obj.is_none();
  const bool has_initial_ts = !initial_ts_obj.is_none();
  if (has_initial_inPs != has_initial_ts) {
    throw std::runtime_error("initial_inPs and initial_ts must be provided together.");
  }
  if (has_initial_inPs) {
    auto inPs_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            initial_inPs_obj);
    auto ts_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            initial_ts_obj);
    if (!inPs_array || !ts_array) {
      throw std::runtime_error("Failed to convert initial MINCO guess arrays.");
    }
    inPs0 = array_to_matrix(
        inPs_array, std::max(context.opt.piece_count - 1, 0), kDim,
        "initial_inPs");
    ts0 = array_to_vector(ts_array);
    if (ts0.size() != context.opt.piece_count) {
      throw std::runtime_error("initial_ts has wrong shape.");
    }
  } else {
    initial_guess(context.head_pva, context.tail_pva, context.opt, inPs0, ts0);
  }

  Eigen::VectorXd variables = pack_variables(inPs0, times_to_theta(ts0, context.opt));
  clip_to_bounds(variables, context.bounds);
  const ObjectiveEvalResult before = evaluate_minco_objective(
      variables, context.head_pva, context.tail_pva, context.opt, context.penalty,
      &context.quadrature_alphas, &context.quadrature_weights);

  lbfgs::lbfgs_parameter_t params;
  params.mem_size = 256;
  params.past = 3;
  params.min_step = 1.0e-32;
  params.g_epsilon = std::max(0.0, context.opt.gradient_tolerance);
  params.delta = std::max(0.0, context.opt.function_tolerance);
  params.max_iterations = std::max(0, context.opt.max_iterations);

  const auto solve_start = std::chrono::steady_clock::now();
  double final_objective = before.objective;
  int status_code = 0;
  {
    py::gil_scoped_release release;
    status_code = lbfgs::lbfgs_optimize(
        variables,
        final_objective,
        lbfgs_minco_evaluate,
        lbfgs_minco_stepbound,
        lbfgs_minco_progress,
        &context,
        params);
  }
  const auto solve_end = std::chrono::steady_clock::now();

  clip_to_bounds(variables, context.bounds);
  const ObjectiveEvalResult after = evaluate_minco_objective(
      variables, context.head_pva, context.tail_pva, context.opt, context.penalty,
      &context.quadrature_alphas, &context.quadrature_weights);
  const MincoTrajectory traj = build_trajectory_from_eval(
      after, context.head_pva, context.tail_pva, context.opt);

  Eigen::VectorXd times;
  Eigen::MatrixXd z;
  Eigen::MatrixXd z_dot;
  Eigen::MatrixXd z_ddot;
  Eigen::MatrixXd nu;
  Eigen::MatrixXd nu_dot;
  Eigen::MatrixXd tau;
  Eigen::MatrixXd thrust;
  py::dict diagnostics = sample_minco_reference(
      traj, context.penalty.params, dt, times, z, z_dot, z_ddot, nu, nu_dot,
      tau, thrust);
  dict_update(diagnostics, diagnostics_dict(after));
  diagnostics["feasible_reference_exported"] = 1.0;
  diagnostics["minco_energy"] = trajectory_energy(traj);
  diagnostics["minco_duration"] = trajectory_duration(traj);
  diagnostics["minco_piece_count"] = static_cast<double>(traj.piece_count);
  diagnostics["minco_min_segment_time"] = traj.ts.size() > 0 ? traj.ts.minCoeff() : 0.0;
  diagnostics["minco_max_segment_time"] = traj.ts.size() > 0 ? traj.ts.maxCoeff() : 0.0;
  diagnostics["minco_penalty_total"] = after.penalty_result.total;
  diagnostics["minco_objective_before"] = before.objective;
  diagnostics["minco_objective_after"] = after.objective;
  diagnostics["minco_optimizer_success"] = status_code >= 0 ? 1.0 : 0.0;
  diagnostics["minco_optimizer_iterations"] = static_cast<double>(context.iterations);
  diagnostics["minco_cpp_planner_exported"] = 1.0;
  diagnostics["optimizer_backend_cpp_lbfgs_lite_plan_export"] = 1.0;
  diagnostics["optimizer_eval_count"] = static_cast<double>(context.eval_count);
  diagnostics["optimizer_line_search_evals"] = static_cast<double>(
      context.line_search_evals);
  diagnostics["optimizer_grad_inf_norm"] =
      after.gradient.size() > 0 ? after.gradient.cwiseAbs().maxCoeff() : 0.0;
  diagnostics["optimizer_status_code"] = static_cast<double>(status_code);
  dict_update(diagnostics, minco_residuals_dict(traj, context.head_pva, context.tail_pva));

  const auto wall_end = std::chrono::steady_clock::now();
  diagnostics["minco_cpp_planner_solve_time_ms"] =
      std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
  diagnostics["minco_cpp_planner_wall_time_ms"] =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  py::dict result;
  result["success"] = status_code >= 0;
  result["status_code"] = status_code;
  result["message"] =
      context.failure_message.empty()
          ? std::string(lbfgs::lbfgs_strerror(status_code))
          : context.failure_message;
  result["objective_before"] = before.objective;
  result["objective_after"] = after.objective;
  result["optimizer_fun"] = final_objective;
  result["iterations"] = context.iterations;
  result["eval_count"] = context.eval_count;
  result["variables"] = vector_to_numpy(variables);
  result["gradient"] = vector_to_numpy(after.gradient);
  result["inPs"] = matrix_to_numpy(after.inPs);
  result["theta"] = vector_to_numpy(after.theta);
  result["ts"] = vector_to_numpy(after.ts);
  result["coefficients"] = matrix_to_numpy(traj.coeffs);
  result["t"] = vector_to_numpy(times);
  result["z"] = matrix_to_numpy(z);
  result["z_dot"] = matrix_to_numpy(z_dot);
  result["z_ddot"] = matrix_to_numpy(z_ddot);
  result["nu"] = matrix_to_numpy(nu);
  result["nu_dot"] = matrix_to_numpy(nu_dot);
  result["tau"] = matrix_to_numpy(tau);
  result["thrust"] = matrix_to_numpy(thrust);
  result["diagnostics"] = diagnostics;
  return result;
}

}  // namespace

PYBIND11_MODULE(_robotx_safe_docking_minco_cpp, m) {
  m.doc() = "C++ hot-path helpers for RobotX safe docking MINCO planning.";
  m.def(
      "minco_objective_gradient", &minco_objective_gradient,
      py::arg("variables"), py::arg("head_pva"), py::arg("tail_pva"),
      py::arg("params"), py::arg("optimizer_config"),
      py::arg("penalty_config"));
  m.def(
      "minco_optimize_lbfgs", &minco_optimize_lbfgs,
      py::arg("variables"), py::arg("head_pva"), py::arg("tail_pva"),
      py::arg("params"), py::arg("optimizer_config"),
      py::arg("penalty_config"));
  m.def(
      "plan_minco_reference", &plan_minco_reference_cpp,
      py::arg("head_pva"), py::arg("tail_pva"), py::arg("params"),
      py::arg("optimizer_config"), py::arg("penalty_config"), py::arg("dt"),
      py::arg("initial_inPs") = py::none(), py::arg("initial_ts") = py::none());
  m.def(
      "plan_lattice_terminal", &plan_lattice_terminal_cpp,
      py::arg("z"), py::arg("z_dot"), py::arg("params"), py::arg("config"));
}
