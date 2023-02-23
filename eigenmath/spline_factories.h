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

#ifndef EIGENMATH_EIGENMATH_SPLINE_FACTORIES_H_
#define EIGENMATH_EIGENMATH_SPLINE_FACTORIES_H_

#include <vector>

#include "quintic_spline.h"

namespace eigenmath {

// Creates a spline representing a straight line from start to end, with
// constant velocity.
//
// `start`: The start position
// `end`: The end position (must have same dimensionality as start)
// Returns a QuinticSpline which has a single polynomial segment per DOF,
// starting at `start` and ending at `end`, with constant velocity throughout
// and zero accelerations at either end.
QuinticSpline MakeSplineConstantVelocity(const Eigen::VectorXd& start,
                                         const Eigen::VectorXd& end);

// Creates a spline representing a straight line from start to end, that starts
// and ends at zero velocity and acceleration.
//
// `start`: The start position
// `end`: The end position (must have same dimensionality as start)
// Returns a QuinticSpline which has a single polynomial segment per DOF,
// starting at `start` and ending at `end`, with zero velocities and
// accelerations at either end.
QuinticSpline MakeSplineZeroStartEndVelAcc(const Eigen::VectorXd& start,
                                           const Eigen::VectorXd& end);

// Creates a natural spline representing from start to end, that starts
// and ends with the given direction vector and with zero curvature.
//
// `start`: The start position
// `start_dir`: The start direction (magnitude is ignored)
// `end`: The end position (must have same dimensionality as start)
// `end_dir`: The end direction (magnitude is ignored)
// Returns a QuinticSpline which has a single polynomial segment per DOF,
// starting at `start` and ending at `end`, with given directions and
// zero curvature at either end.
QuinticSpline MakeSplineNaturalWithDirections(const Eigen::VectorXd& start,
                                              const Eigen::VectorXd& start_dir,
                                              const Eigen::VectorXd& end,
                                              const Eigen::VectorXd& end_dir);

// Create a quintic spline from a vector of waypoints.
//
// The resulting spline will have zero velocities and accelerations at each
// waypoint.
//
// `waypoints`: A vector of waypoints, each of which has the same
// dimensionality.
// Returns a QuinticSpline which passes through each of the waypoints exactly,
// with zero velocities and accelerations.
QuinticSpline MakeSplineFromWaypoints(
    const std::vector<Eigen::VectorXd>& waypoints);

// Create a quintic spline from a vector of waypoints and directions.
//
// The resulting spline will have continuous direction, continuous velocity,
// continuous acceleration, and zero curvature at each waypoint.
//
// `waypoints`: A vector of waypoints, each of which has the same
// dimensionality.
// `directions`: A vector of directions, each of which has the same
// dimensionality. Magnitudes are ignored.
// Returns a QuinticSpline which passes through each waypoint exactly,
// with continuous direction, continuous velocity, continuous acceleration,
// and zero curvature at each waypoint.
QuinticSpline MakeSplineFromWaypointsAndDirections(
    const std::vector<Eigen::VectorXd>& waypoints,
    const std::vector<Eigen::VectorXd>& directions);

// Create a quintic spline from a raw set of control points.
//
// `control_points`: Each element of ctrl_points describes a 1 DOF spline,
// with M control points. A control point consists of the position, and its 1st
// and 2nd derivatives (q, qd, qdd). Each 1 DOF control point vector is of the
// form [ q_0, qd_0, qdd_0, q_1, qd_1, qdd_2,... q_M, qd_M, qdd_M ]'. Each
// vector must have the same size (M * 3). The resulting spline contains M-1
// fifth order polynomial segments for each DOF, each of which is formed by
// taking two consecutive control points.
// Returns a QuinticSpline corresponding to the control points.
QuinticSpline MakeSplineFromControlPoints(
    const std::vector<Eigen::VectorXd>& control_points);

// Create a quintic spline from a raw set of control points.
//
// This function simply forwards its argument to the QuinticSpline
// constructor, it's kept here for consistency.
//
// `control_points`: Each column of ctrl_points describes a 1 DOF spline,
// with M control points. A control point consists of the position, and its 1st
// and 2nd derivatives (q, qd, qdd). Each column vector is of the form [ q_0,
// qd_0, qdd_0, q_1, qd_1, qdd_2,... q_M, qd_M, qdd_M ]'. The number of rows is
// thus M * 3. The number of columns corresponds to the number of DOFs. The
// resulting spline contains M-1 fifth order polynomial segments for each DOF,
// each of which is formed by taking two consecutive control points.
// Returns a QuinticSpline corresponding to the control points.
QuinticSpline MakeSplineFromControlPoints(
    const Eigen::MatrixXd& control_points);

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SPLINE_FACTORIES_H_
