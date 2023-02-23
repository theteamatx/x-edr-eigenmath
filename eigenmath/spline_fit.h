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

#ifndef EIGENMATH_EIGENMATH_SPLINE_FIT_H_
#define EIGENMATH_EIGENMATH_SPLINE_FIT_H_

#include <functional>
#include <vector>

#include "quadratic_optimization.h"
#include "quintic_spline.h"
#include "types.h"

namespace eigenmath {

// Fits a quintic spline to a set of waypoints, with optional constraints on
// derivatives.
//
// (see below for shorter, commonly used overloads)
//
// This function fits a set of quintic polynomial segments to the provided
// waypoints, by minimizing the sum of squared distances between each waypoint
// and a point on the spline. The waypoints are uniformly distributed to spline
// index points between [0, num_segments]. The fit is regularized by adding a
// cost on the integral squared acceleration of the entire spline. Each waypoint
// can be assigned a different weight on its least squares cost term, if needed.
// The start and end position will always be satisfied exactly. Finally, the
// function also accepts optional constraints at each waypoint for its desired
// position, velocity, and acceleration, if any of these quantities need to be
// achieved exactly.
//
// Say we have s spline segments, with which to fit N points (\f$ z_1 \ldots z_N
// \f$). The s spline segments are represented by s+1 control points, each of
// which consists of a position, velocity and acceleration, resulting in (s+1)*3
// parameters, which we stack into a column vector called p. We also stack the
// points \f$ z_1 \ldots z_N \f$ into a single column vector z. Now we solve the
// following least squares minimization problem: \f$ \min_p (z - Xp)^T W (z -
// Xp) + p^T kR p \f$, where X is an N by (s+1)*3 matrix that maps the spline
// control parameter vector p onto the positions sampled by the spline, W is an
// N by N diagonal weight matrix for weights per waypoint, and R is a square
// matrix such that \f$p^T R p\f$ computes the integral of squared accelerations
// of the entire spline. k is the weight on the acceleration cost term: a higher
// value will produce a smoother spline, at the expense of not fitting the
// points exactly. 1.0 might be a good starting point for this parameter, but
// this needs to be tuned for each application. Alternatively, consider using
// `FitSplineToWaypointsSafe` which automatically tunes the weights in
// conjunction with a validity checker.
//
// Each set of constraints (for positions, velocities, and accelerations
// respectively) must contain unique waypoint indices, e.g., specifying two
// position constraints for the same waypoint index is invalid. Moreover,
// constraints for indices 0 and waypoints.size()-1 cannot be specified in the
// position_constraints vector, because the start and end are implicitly
// constrained to the first and last waypoints.
//
// A reference on this algorithm can be found here:
// Sprunk, Lau, Burgard, "Improved Non-linear Spline Fitting for Teaching
// Trajectories to Mobile Robots", ICRA 2012.
// http://ais.informatik.uni-freiburg.de/publications/papers/sprunk12icra.pdf
// Our implementation is based on the above method, but adds regularization,
// per-via weights, and position and derivative constraints.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `acceleration_cost_weight`: The weight of the squared acceleration cost
// `waypoint_weights`: A vector of weights per waypoint. If empty, weights
// are considered to be 1.0 for every waypoint.
// `position_constraints`: A vector of position constraints to apply when
// fitting the spline. Each constraint specifies an index (waypoint number), and
// the desired position at that waypoint.
// `velocity_constraints`: A vector of velocity constraints to apply when
// fitting the spline. Each constraint specifies an index (waypoint number), and
// the desired velocity at that waypoint.
// `acceleration_constraints`: A vector of acceleration constraints to apply
// when fitting the spline. Each constraint specifies an index (waypoint
// number), and the desired acceleration at that
// waypoint.
// Returns QuinticSpline containing `num_segments` segments, which fits
// `waypoints`, using the given weights, satisfying the constraints.
QuinticSpline FitSplineToWaypoints(
    const std::vector<VectorXd>& waypoints, int num_segments,
    double acceleration_cost_weight,
    const std::vector<double>& waypoint_weights,
    const std::vector<VariableSubstitution>& position_constraints,
    const std::vector<VariableSubstitution>& velocity_constraints,
    const std::vector<VariableSubstitution>& acceleration_constraints);

// Fits a quintic spline to a set of waypoints.
//
// This is an overload of `FitSplineToWaypoints` which omits derivative
// constraints.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `acceleration_cost_weight`: The weight of the squared acceleration cost
// `waypoint_weights`: A vector of weights per waypoint. If empty, weights
// are considered to be 1.0 for every waypoint.
// Returns QuinticSpline containing `num_segments` segments, which fits
// `waypoints` using the given weights.
QuinticSpline FitSplineToWaypoints(const std::vector<VectorXd>& waypoints,
                                   int num_segments,
                                   double acceleration_cost_weight,
                                   const std::vector<double>& waypoint_weights);

// Fits a quintic spline to a set of waypoints.
//
// This is an overload of `FitSplineToWaypoints` which omits derivative
// constraints and waypoint weights.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `acceleration_cost_weight`: The weight of the squared acceleration cost
// Returns QuinticSpline containing `num_segments` segments, which fits
// `waypoints` using the given acceleration cost weight.
QuinticSpline FitSplineToWaypoints(const std::vector<VectorXd>& waypoints,
                                   int num_segments,
                                   double acceleration_cost_weight);

// Fits a quintic spline to a set of waypoints.
//
// This is an overload of `FitSplineToWaypoints` which omits derivative
// constraints and weights.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// Returns QuinticSpline containing `num_segments` segments, which fits
// `waypoints`.
QuinticSpline FitSplineToWaypoints(const std::vector<VectorXd>& waypoints,
                                   int num_segments);

// Signature for a function that accepts a vector of waypoints, and outputs a
// bool for each waypoint, indicating whether the waypoint is valid or not. This
// is used in `FitSplineToWaypointsSafe`.
using PathValidityChecker =
    std::function<std::vector<bool>(const std::vector<VectorXd>&)>;

// Signature for a function that accepts a vector of waypoints and velocities
// respectively, and outputs a bool for each waypoint, indicating whether the
// waypoint is valid or not. This is used in `FitSplineToWaypointsSafe`.
using PathWithVelocityValidityChecker = std::function<std::vector<bool>(
    const std::vector<VectorXd>&, const std::vector<VectorXd>&)>;

// Fits a quintic spline to a set of waypoints while ensuring validity.
//
// (see below for shorter, commonly used overloads)
//
// This is an iterative loop around `FitSplineToWaypoints`. It starts by
// fitting a spline with the provided weights. It then computes the validity of
// the path that the spline achieves. If there are points on the spline found to
// be invalid, their weights are bumped up, and the spline fit is recomputed.
// This algorithm is useful to smooth out jerky paths while still ensuring
// validity. Ideally, the validity checker should compute validity in between
// pairs of points, and not just at the points themselves. This also assumes
// that the input waypoints, and straight lines between them are perfectly valid
// (otherwise this algorithm could never succeed).
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `validity_checker`: A function that checks the validity of a sampled path
// and its velocities and returns a bool per waypoint.
// `acceleration_cost_weight`: The weight of the squared acceleration cost
// `waypoint_weights`: A vector of weights per waypoint. If empty, weights
// are considered to be 1.0 for every waypoint.
// `position_constraints`: A vector of position constraints to apply when
// fitting the spline. Each constraint specifies an index (waypoint number), and
// the desired position at that waypoint.
// `velocity_constraints`: A vector of velocity constraints to apply when
// fitting the spline. Each constraint specifies an index (waypoint number), and
// the desired velocity at that waypoint.
// `acceleration_constraints`: A vector of acceleration constraints to apply
// when fitting the spline. Each constraint specifies an index (waypoint
// number), and the desired acceleration at that
// waypoint.
// `spline`: [out] QuinticSpline containing `num_segments` segments, which
// fits `waypoints`.
// Returns true if the fit was successful and valid, false otherwise.
bool FitSplineToWaypointsSafe(
    const std::vector<VectorXd>& waypoints, int num_segments,
    const PathWithVelocityValidityChecker& validity_checker,
    double acceleration_cost_weight, std::vector<double> waypoint_weights,
    const std::vector<VariableSubstitution>& position_constraints,
    const std::vector<VariableSubstitution>& velocity_constraints,
    const std::vector<VariableSubstitution>& acceleration_constraints,
    QuinticSpline& spline);

// Fits a quintic spline to a set of waypoints while ensuring validity.
//
// Same as above, but accepts a PathValidityChecker without velocities.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `validity_checker`: A function that checks the validity of a sampled path
// and returns a bool per waypoint.
// `acceleration_cost_weight`: The weight of the squared acceleration cost
// `waypoint_weights`: A vector of weights per waypoint. If empty, weights
// are considered to be 1.0 for every waypoint.
// `position_constraints`: A vector of position constraints to apply when
// fitting the spline. Each constraint specifies an index (waypoint number), and
// the desired position at that waypoint.
// `velocity_constraints`: A vector of velocity constraints to apply when
// fitting the spline. Each constraint specifies an index (waypoint number), and
// the desired velocity at that waypoint.
// `acceleration_constraints`: A vector of acceleration constraints to apply
// when fitting the spline. Each constraint specifies an index (waypoint
// number), and the desired acceleration at that
// waypoint.
// `spline`: [out] QuinticSpline containing `num_segments` segments, which
// fits `waypoints`.
// Returns true if the fit was successful and valid, false otherwise.
bool FitSplineToWaypointsSafe(
    const std::vector<VectorXd>& waypoints, int num_segments,
    const PathValidityChecker& validity_checker,
    double acceleration_cost_weight, std::vector<double> waypoint_weights,
    const std::vector<VariableSubstitution>& position_constraints,
    const std::vector<VariableSubstitution>& velocity_constraints,
    const std::vector<VariableSubstitution>& acceleration_constraints,
    QuinticSpline& spline);

// Fits a quintic spline to a set of waypoints.
//
// This is a simpler overload of `FitSplineToWaypointsSafe`.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `validity_checker`: A function that checks the validity of a sampled path
// and its velocities and returns a bool per waypoint.
// `spline`: [out] QuinticSpline containing `num_segments` segments, which
// fits `waypoints`.
// Returns true if the fit was successful and valid, false otherwise.
bool FitSplineToWaypointsSafe(
    const std::vector<VectorXd>& waypoints, int num_segments,
    const PathWithVelocityValidityChecker& validity_checker,
    QuinticSpline& spline);

// Fits a quintic spline to a set of waypoints.
//
// This is a simpler overload of `FitSplineToWaypointsSafe`.
//
// `waypoints`: A vector of waypoints to fit the spline to
// `num_segments`: Number of spline segments to use
// `validity_checker`: A function that checks the validity of a sampled path
// and returns a bool per waypoint.
// `spline`: [out] QuinticSpline containing `num_segments` segments, which
// fits `waypoints`.
// Returns true if the fit was successful and valid, false otherwise.
bool FitSplineToWaypointsSafe(const std::vector<VectorXd>& waypoints,
                              int num_segments,
                              const PathValidityChecker& validity_checker,
                              QuinticSpline& spline);

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SPLINE_FIT_H_
