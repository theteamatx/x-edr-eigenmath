// Copyright 2023 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "spline_fit.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "quadratic_optimization.h"
#include "spline_coefficients.h"

namespace eigenmath {

QuinticSpline FitSplineToWaypoints(const std::vector<VectorXd>& waypoints,
                                   int num_segments) {
  return FitSplineToWaypoints(waypoints, num_segments, 1.0, {});
}

QuinticSpline FitSplineToWaypoints(const std::vector<VectorXd>& waypoints,
                                   int num_segments,
                                   double acceleration_cost_weight) {
  return FitSplineToWaypoints(waypoints, num_segments, acceleration_cost_weight,
                              {});
}

QuinticSpline FitSplineToWaypoints(
    const std::vector<VectorXd>& waypoints, int num_segments,
    double acceleration_cost_weight,
    const std::vector<double>& in_waypoint_weights) {
  return FitSplineToWaypoints(waypoints, num_segments, acceleration_cost_weight,
                              in_waypoint_weights, {}, {}, {});
}

// helper functions and function objects
namespace {

// A function object which produces coefficients for computing spline
// derivatives for a particular waypoint, based on the full spline
// parameterization.
class AssignWaypointCoefficients {
 public:
  // Constructs the function object.
  //
  // `num_waypoints`: Number of waypoints
  // `num_segments`: Number of quintic polynomial segments
  AssignWaypointCoefficients(int num_waypoints, int num_segments)
      : num_waypoints_(num_waypoints), num_segments_(num_segments) {}

  // Updates a row of spline coefficients corresponding to a particular
  // waypoint.
  //
  // `waypoint`: Index of the waypoint
  // `derivative`: Degree of the derivative to compute
  // `weight`: Multiplicative weight on the output coefficients
  // `row`: A row vector to update using the spline coefficients.
  //        (Only 6 parameters corresponding to a particular spline segment
  //        will be updated)
  template <typename Derived>
  void operator()(int waypoint, int derivative, double weight,
                  const Eigen::MatrixBase<Derived>& row) {
    CHECK_GE(waypoint, 0) << "Invalid via index " << waypoint;
    CHECK_LT(waypoint, num_waypoints_) << "Invalid via index " << waypoint;
    double spline_index =
        num_segments_ * (static_cast<double>(waypoint) /
                         static_cast<double>(num_waypoints_ - 1));
    int segment = static_cast<int>(spline_index);  // floor

    CHECK_LE(segment, num_segments_) << "Invalid segment index " << segment;
    if (segment == num_segments_) {  // this could happen for the last point
      segment = num_segments_ - 1;
    }
    double local_index = spline_index - segment;
    constexpr double kTolerance = 1e-10;
    CHECK_GE(local_index, 0.0) << "Local index should have been in [0, 1]";
    CHECK_LE(local_index, 1.0 + kTolerance)
        << "Local index should have been in [0, 1]";
    int num_cols = 3 * (num_segments_ + 1);
    int col_start = segment * 3;
    CHECK_LT(col_start + 5, num_cols) << "column index was invalid";
    Matrix<double, 6, 1> coefficients = Matrix<double, 6, 1>::Zero();
    if (derivative == 0) {
      coefficients = SplineCoefficients<0>(local_index);
    } else if (derivative == 1) {
      coefficients = SplineCoefficients<1>(local_index);
    } else if (derivative == 2) {
      coefficients = SplineCoefficients<2>(local_index);
    } else {
      LOG(FATAL) << "Unsupported derivative: " << derivative;
    }

    // This ugly const-cast is required to write directly into Eigen expressions
    // (see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html)
    const_cast<Eigen::MatrixBase<Derived>&>(row).template segment<6>(
        col_start) = weight * coefficients;
  }

  template <typename Derived>
  void operator()(int waypoint, int derivative,
                  const Eigen::MatrixBase<Derived>& row) {
    return operator()(waypoint, derivative, 1.0, row);
  }

 private:
  int num_waypoints_;
  int num_segments_;
};

}  // namespace

QuinticSpline FitSplineToWaypoints(
    const std::vector<VectorXd>& waypoints, int num_segments,
    double acceleration_cost_weight,
    const std::vector<double>& in_waypoint_weights,
    const std::vector<VariableSubstitution>& position_constraints,
    const std::vector<VariableSubstitution>& velocity_constraints,
    const std::vector<VariableSubstitution>& acceleration_constraints) {
  const int num_waypoints = waypoints.size();
  CHECK_GE(num_waypoints, 2) << "Need at least two waypoints to fit";
  const int num_dof = waypoints[0].size();
  CHECK_GE(num_dof, 1) << "Need at least one DOF to fit";
  CHECK_GE(num_segments, 1) << "Need at least one spline segment to fit";

  // fill up via weights if they don't exist
  auto waypoint_weights_sqrt = in_waypoint_weights;  // copy to modify locally
  if (waypoint_weights_sqrt.empty()) {
    waypoint_weights_sqrt.resize(num_waypoints, 1.0);
  } else {
    CHECK_EQ(static_cast<int>(waypoint_weights_sqrt.size()), num_waypoints)
        << "Waypoint weights is not the right size";
    // actually compute sqrt
    for (auto& weight : waypoint_weights_sqrt) {
      weight = std::sqrt(weight);
    }
  }
  auto assignWaypointCoefficients =
      AssignWaypointCoefficients(num_waypoints, num_segments);

  // X is a matrix that maps the spline control points to the waypoints.
  // The spline will have (num_segments + 1) control points, each of which
  // consists of a position, velocity and acceleration. This results in 3 *
  // (num_segments + 1) columns. The number of rows is equal to the number of
  // waypoints in the input. w = X * p, where w is the vector of waypoints, and
  // p is the spline control point vector.

  int num_cols = 3 * (num_segments + 1);
  MatrixXd X = MatrixXd::Zero(num_waypoints, num_cols);
  for (int via = 0; via < num_waypoints; ++via) {
    assignWaypointCoefficients(via, 0, waypoint_weights_sqrt[via], X.row(via));
  }

  // We wish to minimize
  //
  // arg min_p (z - Xp)^T W (z - Xp) + p^T k R p
  //
  // where p are the spline parameters, z are the measurements, X are the spline
  // basis functions, p^T R p is the integral of squared acceleration of the
  // spline, k is the weight on this acceleration cost, and W is the diagonal
  // waypoint weight matrix.
  //
  // Putting this into a familiar form it becomes:
  //     min.   0.5 p^T G p + p^T c
  //     where:
  //            G = X^T W X + k R
  //            c = -X^T W z
  // (the notation is the same as that used in QuadMinimize*)
  // Note: in our implementation, sqrt(W) has already been absorbed into X
  MatrixXd G = X.transpose() * X;
  // add the squared acceleration costs to G:
  Matrix<double, 6, 6> R_local = SplineIntegralSquaredDerivative<2>();
  for (int i = 0; i < num_segments; ++i) {
    G.block<6, 6>(i * 3, i * 3) += acceleration_cost_weight * R_local;
  }

  // construct the C (measurement) matrix
  MatrixXd Z(num_waypoints, num_dof);
  for (int t = 0; t < num_waypoints; ++t) {
    CHECK_EQ(waypoints[t].size(), num_dof)
        << "Waypoint had wrong number of DOFs";
    Z.row(t) = waypoint_weights_sqrt[t] * waypoints[t];
  }
  MatrixXd C = -X.transpose() * Z;

  // setup problem constraints (equality constraints: Ap = b)
  auto augmented_position_constraints = position_constraints;
  // augment position constraints with start and end
  augmented_position_constraints.push_back({0, waypoints.front()});
  augmented_position_constraints.push_back(
      {num_waypoints - 1, waypoints.back()});
  int num_constraints = augmented_position_constraints.size() +
                        velocity_constraints.size() +
                        acceleration_constraints.size();
  MatrixXd constraints_lhs = MatrixXd::Zero(num_constraints, num_cols);
  MatrixXd constraints_rhs = MatrixXd::Zero(num_constraints, num_dof);

  int next_constraint = 0;

  // convenience function to check and add all user-defined constraints
  auto add_constraints =
      [&constraints_lhs, &constraints_rhs, &next_constraint,
       &assignWaypointCoefficients](
          const std::vector<VariableSubstitution>& constraints_in,
          int derivative) {
        absl::flat_hash_set<int> constraint_set;
        for (const auto& constraint : constraints_in) {
          CHECK(constraint_set.find(constraint.index) == constraint_set.end())
              << "Duplicate constraint index found: " << constraint.index;
          constraint_set.insert(constraint.index);
          assignWaypointCoefficients(constraint.index, derivative,
                                     constraints_lhs.row(next_constraint));
          constraints_rhs.row(next_constraint) = constraint.value;
          ++next_constraint;
        }
      };

  add_constraints(augmented_position_constraints, 0);
  add_constraints(velocity_constraints, 1);
  add_constraints(acceleration_constraints, 2);

  CHECK_EQ(next_constraint, num_constraints)
      << "Didn't fill up all constraints (" << next_constraint
      << " != " << num_constraints << ")";

  return QuinticSpline(
      QuadMinimizeConstrainedSchur(G, C, constraints_lhs, constraints_rhs));
}

bool FitSplineToWaypointsSafe(
    const std::vector<VectorXd>& waypoints, int num_segments,
    const PathWithVelocityValidityChecker& validity_checker,
    double acceleration_cost_weight, std::vector<double> waypoint_weights,
    const std::vector<VariableSubstitution>& position_constraints,
    const std::vector<VariableSubstitution>& velocity_constraints,
    const std::vector<VariableSubstitution>& acceleration_constraints,
    QuinticSpline& spline) {
  const double via_weight_increase_factor = 2.0;
  const unsigned int max_iterations = 50;

  size_t input_path_length = waypoints.size();
  if (waypoint_weights.empty()) {
    waypoint_weights.resize(input_path_length, 1.0);
  } else {
    CHECK_EQ(waypoint_weights.size(), input_path_length)
        << "Waypoint weights is not the right size";
  }

  std::vector<VectorXd> spline_path;
  std::vector<VectorXd> spline_velocities;

  bool path_valid = false;
  std::vector<bool> per_via_validity(input_path_length);
  unsigned int iteration = 0;

  do {
    spline = FitSplineToWaypoints(
        waypoints, num_segments, acceleration_cost_weight, waypoint_weights,
        position_constraints, velocity_constraints, acceleration_constraints);
    spline_path = spline.Discretize<0>(input_path_length);
    spline_velocities = spline.Discretize<1>(input_path_length);
    per_via_validity = validity_checker(spline_path, spline_velocities);
    CHECK_EQ(per_via_validity.size(), input_path_length)
        << "Path validity checker returned wrong number of results";
    path_valid = true;
    for (size_t i = 0; i < input_path_length; ++i) {
      if (!per_via_validity[i]) {
        waypoint_weights[i] *= via_weight_increase_factor;
        path_valid = false;
      }
    }
    ++iteration;
  } while (!path_valid && iteration < max_iterations);

  return path_valid;
}

bool FitSplineToWaypointsSafe(
    const std::vector<VectorXd>& waypoints, int num_segments,
    const PathValidityChecker& validity_checker,
    double acceleration_cost_weight, std::vector<double> waypoint_weights,
    const std::vector<VariableSubstitution>& position_constraints,
    const std::vector<VariableSubstitution>& velocity_constraints,
    const std::vector<VariableSubstitution>& acceleration_constraints,
    QuinticSpline& spline) {
  return FitSplineToWaypointsSafe(
      waypoints, num_segments,
      [&validity_checker](const std::vector<VectorXd>& positions,
                          const std::vector<VectorXd>&  // velocities
                          /*unused*/) { return validity_checker(positions); },
      acceleration_cost_weight, waypoint_weights, position_constraints,
      velocity_constraints, acceleration_constraints, spline);
}

bool FitSplineToWaypointsSafe(
    const std::vector<VectorXd>& waypoints, int num_segments,
    const PathWithVelocityValidityChecker& validity_checker,
    QuinticSpline& spline) {
  return FitSplineToWaypointsSafe(waypoints, num_segments, validity_checker,
                                  1.0, {}, {}, {}, {}, spline);
}

bool FitSplineToWaypointsSafe(const std::vector<VectorXd>& waypoints,
                              int num_segments,
                              const PathValidityChecker& validity_checker,
                              QuinticSpline& spline) {  // NOLINT
  return FitSplineToWaypointsSafe(waypoints, num_segments, validity_checker,
                                  1.0, {}, {}, {}, {}, spline);
}

}  // namespace eigenmath
