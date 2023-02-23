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

#ifndef EIGENMATH_EIGENMATH_QUINTIC_SPLINE_SEGMENT_H_
#define EIGENMATH_EIGENMATH_QUINTIC_SPLINE_SEGMENT_H_

#include "Eigen/Core"
#include "absl/log/check.h"
#include "spline_coefficients.h"
#include "types.h"

namespace eigenmath {

// Represents a single quintic spline segment for N independent dimensions.
//
// This class stores a quintic polynomial per dimension, represented in terms of
// its Hermite coefficients (start and end-point position, velocity, and
// acceleration). It also encapsulates the duration of this segment, to allow
// for correct sampling of derivatives within the segment.
//
// All vectors and matrices are fixed-size, hence this type and all of its
// functions are safe for use within a real-time context.
//
// `N_DOF`: Number of dimensions
template <int N_DOF>
class QuinticSplineSegment {
 public:
  using Vector = eigenmath::Vector<double, N_DOF>;

  // Construct a QuinticSplineSegment from its start and end position, velocity,
  // and acceleration, and the duration of the segment.
  //
  // `start_position`: Start position (size must be N_DOF).
  // `start_velocity`: Start velocity (size must be N_DOF).
  // `start_acceleration`: Start acceleration (size must be N_DOF).
  // `end_position`: End position (size must be N_DOF).
  // `end_velocity`: End velocity (size must be N_DOF).
  // `end_acceleration`: End acceleration (size must be N_DOF).
  // `duration`: Duration of the spline segment (derivatives will be
  // returned in terms of this unit).
  template <typename Derived>
  QuinticSplineSegment(const Eigen::MatrixBase<Derived>& start_position,
                       const Eigen::MatrixBase<Derived>& start_velocity,
                       const Eigen::MatrixBase<Derived>& start_acceleration,
                       const Eigen::MatrixBase<Derived>& end_position,
                       const Eigen::MatrixBase<Derived>& end_velocity,
                       const Eigen::MatrixBase<Derived>& end_acceleration,
                       double duration)
      : duration_(duration) {
    hermite_parameters_.row(0) = start_position;
    hermite_parameters_.row(1) = start_velocity * duration;
    hermite_parameters_.row(2) = start_acceleration * duration * duration;
    hermite_parameters_.row(3) = end_position;
    hermite_parameters_.row(4) = end_velocity * duration;
    hermite_parameters_.row(5) = end_acceleration * duration * duration;
  }

  QuinticSplineSegment(const QuinticSplineSegment& other) = default;
  QuinticSplineSegment(QuinticSplineSegment&& other) = default;
  QuinticSplineSegment& operator=(const QuinticSplineSegment& other) = default;
  QuinticSplineSegment& operator=(QuinticSplineSegment&& other) = default;

  // Get the duration of this spline segment
  //
  // Returns the duration of this spline segment.
  double GetDuration() const { return duration_; }

  // Sample the position of each variable at time t within the spline segment.
  //
  // `t`: Time within the segment at which to sample (must be >= 0 and <=
  // duration).
  // Returns a vector of size N_DOF containing the positions at time t.
  Vector SamplePosition(double t) const {
    AssertTimeIsValid(t);
    return hermite_parameters_.transpose() *
           SplineCoefficients<0>(t / duration_);
  }

  // Sample the velocity of each variable at time t within the spline segment.
  //
  // `t`: Time within the segment at which to sample (must be >= 0 and <=
  // duration).
  // Returns a vector of size N_DOF containing the velocities at time t.
  Vector SampleVelocity(double t) const {
    AssertTimeIsValid(t);
    return hermite_parameters_.transpose() *
           SplineCoefficients<1>(t / duration_) / duration_;
  }

  // Sample the acceleration of each variable at time t within the spline
  // segment.
  //
  // `t`: Time within the segment at which to sample (must be >= 0 and <=
  // duration)
  // Returns a vector of size N_DOF containing the accelerations at time t.
  Vector SampleAcceleration(double t) const {
    AssertTimeIsValid(t);
    return hermite_parameters_.transpose() *
           SplineCoefficients<2>(t / duration_) / (duration_ * duration_);
  }

 private:
  // Check that the time t is within bounds.
  void AssertTimeIsValid(double t) const {
    CHECK_GE(t, 0) << "t must be greater than zero.";
    CHECK_LE(t, duration_) << "t must be <= spline duration.";
  }

  double duration_;  // Duration of this spline segment.
  Matrix<double, 6, N_DOF> hermite_parameters_;  // Hermite parameters.
};

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_QUINTIC_SPLINE_SEGMENT_H_
