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

#ifndef EIGENMATH_EIGENMATH_SPLINE_COEFFICIENTS_H_
#define EIGENMATH_EIGENMATH_SPLINE_COEFFICIENTS_H_

// WARNING: DO NOT EDIT!
// This file was auto-generated by generate_spline_coefficients.py

#include "types.h"

namespace eigenmath {

// Computes the coefficients vector of the spline parameters
// (q0, qd0, qdd0, q1, qd1, qdd1), which may be used to sample the `Degree`-th
// derivative of the spline at a given u value.
//
// The spline is parametrized as
//   S(u) = SplineCoefficients<Degree>(u) * (q0, qd0, qdd0, q1, qd1, qdd1).
// where `u` is the spline index parameter in [0,1].
template <int Degree>
Vector6d SplineCoefficients(const double u);

// Specialization for Degree = 0
template <>
inline Vector6d SplineCoefficients<0>(const double u) {
  const double u2 = u * u;
  const double u3 = u * u2;
  const double u4 = u * u3;
  const double u5 = u * u4;
  return {-10 * u3 + 15 * u4 - 6 * u5 + 1,            // q0
          u - 6 * u3 + 8 * u4 - 3 * u5,               // qd0
          u2 / 2 - 3 * u3 / 2 + 3 * u4 / 2 - u5 / 2,  // qdd0
          10 * u3 - 15 * u4 + 6 * u5,                 // q1
          -4 * u3 + 7 * u4 - 3 * u5,                  // qd1
          u3 / 2 - u4 + u5 / 2};                      // qdd1
}

// Specialization for Degree = 1
template <>
inline Vector6d SplineCoefficients<1>(const double u) {
  const double u2 = u * u;
  const double u3 = u * u2;
  const double u4 = u * u3;
  return {-30 * u2 + 60 * u3 - 30 * u4,          // q0
          -18 * u2 + 32 * u3 - 15 * u4 + 1,      // qd0
          u - 9 * u2 / 2 + 6 * u3 - 5 * u4 / 2,  // qdd0
          30 * u2 - 60 * u3 + 30 * u4,           // q1
          -12 * u2 + 28 * u3 - 15 * u4,          // qd1
          3 * u2 / 2 - 4 * u3 + 5 * u4 / 2};     // qdd1
}

// Specialization for Degree = 2
template <>
inline Vector6d SplineCoefficients<2>(const double u) {
  const double u2 = u * u;
  const double u3 = u * u2;
  return {-60 * u + 180 * u2 - 120 * u3,   // q0
          -36 * u + 96 * u2 - 60 * u3,     // qd0
          -9 * u + 18 * u2 - 10 * u3 + 1,  // qdd0
          60 * u - 180 * u2 + 120 * u3,    // q1
          -24 * u + 84 * u2 - 60 * u3,     // qd1
          3 * u - 12 * u2 + 10 * u3};      // qdd1
}

// Specialization for Degree = 3
template <>
inline Vector6d SplineCoefficients<3>(const double u) {
  const double u2 = u * u;
  return {360 * u - 360 * u2 - 60,   // q0
          192 * u - 180 * u2 - 36,   // qd0
          36 * u - 30 * u2 - 9,      // qdd0
          -360 * u + 360 * u2 + 60,  // q1
          168 * u - 180 * u2 - 24,   // qd1
          -24 * u + 30 * u2 + 3};    // qdd1
}

// Computes a square matrix R [6x6], such that x' * R * x is the integral of the
// squared derivative between 0 and 1.  x here corresponds to the spline
// coefficient vector [q0, qd0, qdd0, q1, qd1, qdd1]. Eg, for velocities:
// \\f$ x^T R x = \int_0^1 ( \dot{s} )^2 \\f$
//
// Here, `Degree` is the derivative of spline
template <int Degree>
Matrix6d SplineIntegralSquaredDerivative();

// Specialization for Degree = 1
template <>
inline Matrix6d SplineIntegralSquaredDerivative<1>() {
  Matrix6d kernel = Matrix6d::Zero();
  kernel << 10.0 / 7.0, 3.0 / 14.0, 1.0 / 84.0, -10.0 / 7.0, 3.0 / 14.0,
      -1.0 / 84.0, 3.0 / 14.0, 8.0 / 35.0, 1.0 / 60.0, -3.0 / 14.0, -1.0 / 70.0,
      1.0 / 210.0, 1.0 / 84.0, 1.0 / 60.0, 1.0 / 630.0, -1.0 / 84.0,
      -1.0 / 210.0, 1.0 / 1260.0, -10.0 / 7.0, -3.0 / 14.0, -1.0 / 84.0,
      10.0 / 7.0, -3.0 / 14.0, 1.0 / 84.0, 3.0 / 14.0, -1.0 / 70.0,
      -1.0 / 210.0, -3.0 / 14.0, 8.0 / 35.0, -1.0 / 60.0, -1.0 / 84.0,
      1.0 / 210.0, 1.0 / 1260.0, 1.0 / 84.0, -1.0 / 60.0, 1.0 / 630.0;
  return kernel;
}

// Specialization for Degree = 2
template <>
inline Matrix6d SplineIntegralSquaredDerivative<2>() {
  Matrix6d kernel = Matrix6d::Zero();
  kernel << 120.0 / 7.0, 60.0 / 7.0, 3.0 / 7.0, -120.0 / 7.0, 60.0 / 7.0,
      -3.0 / 7.0, 60.0 / 7.0, 192.0 / 35.0, 11.0 / 35.0, -60.0 / 7.0,
      108.0 / 35.0, -4.0 / 35.0, 3.0 / 7.0, 11.0 / 35.0, 3.0 / 35.0, -3.0 / 7.0,
      4.0 / 35.0, 1.0 / 70.0, -120.0 / 7.0, -60.0 / 7.0, -3.0 / 7.0,
      120.0 / 7.0, -60.0 / 7.0, 3.0 / 7.0, 60.0 / 7.0, 108.0 / 35.0, 4.0 / 35.0,
      -60.0 / 7.0, 192.0 / 35.0, -11.0 / 35.0, -3.0 / 7.0, -4.0 / 35.0,
      1.0 / 70.0, 3.0 / 7.0, -11.0 / 35.0, 3.0 / 35.0;
  return kernel;
}

// Specialization for Degree = 3
template <>
inline Matrix6d SplineIntegralSquaredDerivative<3>() {
  Matrix6d kernel = Matrix6d::Zero();
  kernel << 720, 360, 60, -720, 360, -60, 360, 192, 36, -360, 168, -24, 60, 36,
      9, -60, 24, -3, -720, -360, -60, 720, -360, 60, 360, 168, 24, -360, 192,
      -36, -60, -24, -3, 60, -36, 9;
  return kernel;
}

}  // namespace eigenmath
#endif  // EIGENMATH_EIGENMATH_SPLINE_COEFFICIENTS_H_
