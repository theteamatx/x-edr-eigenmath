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

#include "spline_factories.h"

#include <vector>

#include "Eigen/Cholesky"
#include "absl/log/check.h"

namespace eigenmath {

QuinticSpline MakeSplineConstantVelocity(const Eigen::VectorXd& start,
                                         const Eigen::VectorXd& end) {
  int num_dof = start.size();
  CHECK_EQ(num_dof, end.size()) << "start and end must have the same size";
  Eigen::MatrixXd control_points(6, num_dof);
  control_points.setZero();
  control_points.row(0) = start;        // start position
  control_points.row(3) = end;          // end position
  control_points.row(1) = end - start;  // start velocity
  control_points.row(4) = end - start;  // end velocity
  return QuinticSpline(control_points);
}

QuinticSpline MakeSplineZeroStartEndVelAcc(const Eigen::VectorXd& start,
                                           const Eigen::VectorXd& end) {
  int num_dof = start.size();
  CHECK_EQ(num_dof, end.size()) << "start and end must have the same size";
  Eigen::MatrixXd control_points(6, num_dof);
  control_points.setZero();
  control_points.row(0) = start;    // start position
  control_points.row(3) = end;      // end position
  control_points.row(1).setZero();  // start velocity
  control_points.row(4).setZero();  // end velocity
  return QuinticSpline(control_points);
}

QuinticSpline MakeSplineNaturalWithDirections(const Eigen::VectorXd& start,
                                              const Eigen::VectorXd& start_dir,
                                              const Eigen::VectorXd& end,
                                              const Eigen::VectorXd& end_dir) {
  const int num_dof = start.size();
  CHECK_EQ(num_dof, start_dir.size())
      << "start and start_dir must have the same size";
  CHECK_EQ(num_dof, end.size()) << "start and end must have the same size";
  CHECK_EQ(num_dof, end_dir.size())
      << "start and end_dir must have the same size";
  Eigen::MatrixXd control_points(6, num_dof);
  control_points.setZero();
  control_points.row(0) = start;  // start position
  control_points.row(3) = end;    // end position

  // The start and end velocity and acceleration will be aligned with the
  // given start and end direction vectors, we must solve for the magnitudes.
  // We are solving for 4 scaling factors.
  // We are solving for the minimum integral of the norm of the second
  // derivative, which is not ideal (solving for minimum curvature would
  // be better), but this produces a simple quadratic problem and resulting
  // curves are smooth enough.

  // Construct the quadratic part of the cost function:
  const double t1t1 = start_dir.dot(start_dir);
  const double t1t2 = start_dir.dot(end_dir);
  const double t2t2 = end_dir.dot(end_dir);
  Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
  Q << 8.0 / 35.0 * t1t1, t1t1 / 60.0, -t1t2 / 70.0, t1t2 / 210.0, t1t1 / 60.0,
      t1t1 / 630.0, -t1t2 / 210.0, t1t2 / 1260.0, -t1t2 / 70.0, -t1t2 / 210.0,
      8.0 / 35.0 * t2t2, -t2t2 / 60.0, t1t2 / 210.0, t1t2 / 1260.0,
      -t2t2 / 60.0, t2t2 / 630.0;

  // Construct the linear part of the cost function:
  const Eigen::VectorXd p1_minus_p2 = start - end;
  const double p1p2_dot_t1 = p1_minus_p2.dot(start_dir);
  const double p1p2_dot_t2 = p1_minus_p2.dot(end_dir);
  Eigen::Vector4d c = Eigen::Vector4d::Zero();
  c << 3.0 / 14.0 * p1p2_dot_t1, 1.0 / 84.0 * p1p2_dot_t1,
      3.0 / 14.0 * p1p2_dot_t2, -1.0 / 84.0 * p1p2_dot_t2;

  // Solve for the scaling factors and ensure velocities are in good direction.
  Eigen::Vector4d lambda = Q.llt().solve(c);
  if (lambda[0] < 0) {
    lambda[0] = -lambda[0];
  }
  if (lambda[0] * lambda[1] > 0) {
    lambda[1] = -lambda[1];
  }
  if (lambda[2] < 0) {
    lambda[2] = -lambda[2];
  }
  if (lambda[2] * lambda[3] > 0) {
    lambda[3] = -lambda[3];
  }

  // Apply the scale factors that we have solved for.
  control_points.row(1) = start_dir * lambda[0];  // start velocity
  control_points.row(2) = start_dir * lambda[1];  // start acceleration
  control_points.row(4) = end_dir * lambda[2];    // end velocity
  control_points.row(5) = end_dir * lambda[3];    // end acceleration

  return QuinticSpline(control_points);
}

QuinticSpline MakeSplineFromWaypoints(
    const std::vector<Eigen::VectorXd>& waypoints) {
  int num_waypoints = waypoints.size();
  CHECK_GE(num_waypoints, 2) << "Need at least 2 waypoints";
  int num_dof = waypoints[0].size();
  Eigen::MatrixXd control_points(num_waypoints * 3, num_dof);
  control_points.setZero();
  for (int i = 0; i < num_waypoints; ++i) {
    CHECK_EQ(waypoints[i].size(), num_dof)
        << "All waypoints must have the same size";
    control_points.row(i * 3) = waypoints[i];
  }
  return QuinticSpline(control_points);
}

QuinticSpline MakeSplineFromWaypointsAndDirections(
    const std::vector<Eigen::VectorXd>& waypoints,
    const std::vector<Eigen::VectorXd>& directions) {
  const int num_waypoints = waypoints.size();
  CHECK_GE(num_waypoints, 2) << "Need at least 2 waypoints";
  CHECK_EQ(num_waypoints, directions.size())
      << "waypoints and directions must have the same size";
  const int num_dof = waypoints[0].size();
  CHECK_EQ(num_dof, directions[0].size())
      << "waypoints and directions must have the same dof";
  Eigen::MatrixXd control_points(num_waypoints * 3, num_dof);
  control_points.setZero();
  for (int i = 0; i < num_waypoints; ++i) {
    CHECK_EQ(waypoints[i].size(), num_dof)
        << "All waypoints must have the same size";
    CHECK_EQ(directions[i].size(), num_dof)
        << "All directions must have the same size";
    control_points.row(i * 3) = waypoints[i];
  }

  // The waypoint velocity and acceleration will be aligned with the
  // given waypoint direction vectors, we must solve for their magnitudes.
  // We are solving for 2*num_waypoints scaling factors.
  // We are solving for the minimum integral of the norm of the second
  // derivative, which is not ideal (solving for minimum curvature would
  // be better), but this produces a simple quadratic problem and resulting
  // curves are smooth enough.

  Eigen::MatrixXd Q(num_waypoints * 2, num_waypoints * 2);
  Q.setZero();
  Eigen::VectorXd c(num_waypoints * 2);
  c.setZero();
  double t1t1 = directions[0].dot(directions[0]);
  for (int i = 1; i < num_waypoints; ++i) {
    // Construct the quadratic part of the cost function:
    const double t1t2 = directions[i - 1].dot(directions[i]);
    const double t2t2 = directions[i].dot(directions[i]);
    Eigen::Matrix4d sub_Q = Eigen::Matrix4d::Zero();
    sub_Q << 8.0 / 35.0 * t1t1, t1t1 / 60.0, -t1t2 / 70.0, t1t2 / 210.0,
        t1t1 / 60.0, t1t1 / 630.0, -t1t2 / 210.0, t1t2 / 1260.0, -t1t2 / 70.0,
        -t1t2 / 210.0, 8.0 / 35.0 * t2t2, -t2t2 / 60.0, t1t2 / 210.0,
        t1t2 / 1260.0, -t2t2 / 60.0, t2t2 / 630.0;
    Q.block<4, 4>(2 * (i - 1), 2 * (i - 1)) += sub_Q;
    // For next iteration:
    t1t1 = t2t2;

    // Construct the linear part of the cost function:
    const Eigen::VectorXd p1_minus_p2 = waypoints[i - 1] - waypoints[i];
    const double p1p2_dot_t1 = p1_minus_p2.dot(directions[i - 1]);
    const double p1p2_dot_t2 = p1_minus_p2.dot(directions[i]);
    Eigen::Vector4d sub_c = Eigen::Vector4d::Zero();
    sub_c << 3.0 / 14.0 * p1p2_dot_t1, 1.0 / 84.0 * p1p2_dot_t1,
        3.0 / 14.0 * p1p2_dot_t2, -1.0 / 84.0 * p1p2_dot_t2;
    c.block<4, 1>(2 * (i - 1), 0) += sub_c;
  }

  // Solve for the scaling factors and ensure velocities are in good direction.
  // Apply the scale factors that we have solved for.
  Eigen::VectorXd lambda = Q.llt().solve(c);
  for (int i = 0; i < num_waypoints; ++i) {
    if (lambda[2 * i] < 0) {
      lambda[2 * i] = -lambda[2 * i];
    }
    if (lambda[2 * i] * lambda[2 * i + 1] > 0) {
      lambda[2 * i + 1] = -lambda[2 * i + 1];
    }
    // Set velocity and acceleration.
    control_points.row(3 * i + 1) = directions[i] * lambda[2 * i];
    control_points.row(3 * i + 2) = directions[i] * lambda[2 * i + 1];
  }

  return QuinticSpline(control_points);
}

QuinticSpline MakeSplineFromControlPoints(
    const std::vector<Eigen::VectorXd>& control_points) {
  int num_dof = control_points.size();
  CHECK_GE(num_dof, 1);

  const int num_control_points = control_points[0].size() / 3;  // q, qd, qdd
  Eigen::MatrixXd control_points_matrix(num_control_points * 3, num_dof);

  for (int dof = 0; dof < num_dof; ++dof) {
    const int control_point_vector_length = control_points[dof].size();
    CHECK_EQ(control_point_vector_length % 3, 0);
    CHECK_GE(control_point_vector_length, 6);
    CHECK_EQ(control_point_vector_length, num_control_points * 3)
        << "All control point vectors must be the same size!";
    control_points_matrix.col(dof) = control_points[dof];
  }
  return QuinticSpline(control_points_matrix);
}

QuinticSpline MakeSplineFromControlPoints(
    const Eigen::MatrixXd& control_points) {
  return QuinticSpline(control_points);
}

}  // namespace eigenmath
