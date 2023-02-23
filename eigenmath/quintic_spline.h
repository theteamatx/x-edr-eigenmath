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

#ifndef EIGENMATH_EIGENMATH_QUINTIC_SPLINE_H_
#define EIGENMATH_EIGENMATH_QUINTIC_SPLINE_H_

#include <vector>

#include "Eigen/Core"

namespace eigenmath {

// Representation of a quintic spline.
//
// Consists of a number of consecutive fifth order polynomial segments,
// independently for each degree of freedom. The spline is represented as a set
// of control points, each of which consists of a position, velocity and
// acceleration for each degree of freedom. Every point on the curve and its
// derivatives are completely determined by its two neighboring control points.
//
// This class can only be constructed using a raw Eigen::MatrixXd representing
// the control points. More convenient ways of creating splines or fitting
// splines to data can be found in spline_factories.h and spline_fit.h.
//
// Internal details: Each polynomial segment for a single dof is defined as
// follows: @f$ q = k_5 u^5 + k_4 u^4 + k_3 u^3 + k_2 u^2 + k_1 u + k_0 @f$,
// where @f$q@f$ is the value of the spline curve at position @f$u@f$ within a
// particular segment. @f$u@f$ ranges from 0 to 1 within a segment (also called
// `local_index` at function interfaces). @f$k_0@f$ through @f$k_5@f$ are the
// polynomial coefficients. Since we want to ensure that neighboring segments
// share the same position, velocity and acceleration, we use these quantities
// as the internal storage mechanism instead of storing the coefficients
// directly.
class QuinticSpline {
 public:
  QuinticSpline() = default;

  // Construct a quintic spline from a raw set of control points.
  //
  // More convenient spline creation functions are in spline_factories.h.
  //
  // `control_points`: Each column of ctrl_points describes a 1 DOF spline,
  // with M control points. A control point consists of the position, and its
  // 1st and 2nd derivatives (q, qd, qdd). Each column vector is of the form [
  // q_0, qd_0, qdd_0, q_1, qd_1, qdd_2,... q_M, qd_M, qdd_M ]'. The number of
  // rows is thus M * 3. The number of columns corresponds to the number of
  // DOFs. The resulting spline contains M-1 fifth order polynomial segments for
  // each DOF, each of which is formed by taking two consecutive control points.
  explicit QuinticSpline(const Eigen::MatrixXd& control_points);

  // Sample a point within the spline.
  //
  // `degree`: The degree of the derivative to sample [0, 3].
  // `spline_index`: The spline_index (also called s for short) is mapped
  // to a segment number by taking the floor of s. The local index into the
  // polynomial (u) is then obtained as u = s - floor(s). spline_index must be
  // in the range [0, maxSplineIndex()].
  //
  // Returns a vector of size numDof(), holding the 'degree'-th derivative of
  // the spline at spline_index.
  template <int degree>
  Eigen::VectorXd Sample(double spline_index) const;

  // Sample a point within the spline (in-place form).
  //
  // `degree`: The degree of the derivative to sample [0, 3].
  // `spline_index`: Index into the spline [0, maxSplineIndex()].
  // `value`: The 'degree'-th derivative of the spline at spline_index.
  template <int degree>
  void Sample(double spline_index, Eigen::VectorXd& value) const;

  // Samples the spline at each point in the given vector of spline_index
  // parameters.
  //
  // `degree`: The degree of the derivative to sample [0, 3].
  // `spline_indices`: A vector of indices into the spline, each of which
  // must be in the range [0, maxSplineIndex()].
  // Returns a vector of 'degree'-th derivative of the spline at each index.
  template <int degree>
  std::vector<Eigen::VectorXd> Sample(
      const std::vector<double>& spline_indices) const;

  // Returns a uniformly discretized version of the spline, with num_samples
  // elements.
  //
  // `degree`: The degree of the derivative to sample [0, 3].
  // `num_samples`: Number of samples in the returned path. (must be >= 2)
  // Returns the vector of num_samples points representing the 'degree'-th
  // derivative of the spline.
  template <int degree>
  std::vector<Eigen::VectorXd> Discretize(int num_samples) const;

  // Returns the integral of squared derivative of the spline.
  //
  // Computes @f$ \int_{s=0}^{max_s}\left( \frac{d^n q(s)}{d q^n} \right)^2 @f$,
  // where max_s is @c maxSplineIndex(), @f$q(s)@f$ is the value of the spline
  // at index @f$s@f$, and @f$n@f$ is the degree of the derivative.
  //
  // `degree`: The degree of the derivative [1, 3].
  // Returns the integral squared derivative of the spline, one value per DOF.
  template <int degree>
  Eigen::VectorXd IntegralSquaredDerivative() const;

  // Maximum value for the spline index parameter (s).
  double MaxSplineIndex() const;

  // Number of degrees of freedom.
  int NumDof() const;

  // Number of contained polynomial segments.
  int NumSegments() const;

 private:
  // Represents a location within the spline, as a combination of segment number
  // and a position within the segment.
  struct LocalIndex {
    int segment;         // The index of the polynomial segment in the spline.
    double local_index;  // Index [0,1] within the segment.
  };

  // Converts a global spline index to the local segment number and index.
  LocalIndex GetLocalIndex(double spline_index) const;

  Eigen::MatrixXd control_points_;
};

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_QUINTIC_SPLINE_H_
