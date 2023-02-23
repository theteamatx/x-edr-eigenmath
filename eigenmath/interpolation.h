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

#ifndef EIGENMATH_EIGENMATH_INTERPOLATION_H_
#define EIGENMATH_EIGENMATH_INTERPOLATION_H_

#include <cmath>

#include "pose2.h"
#include "pose3.h"
#include "types.h"

namespace eigenmath {

// Computes the relative percentage of `value` in interval [`left`, `right`].
//
// Assumes that the value is in the interval.
template <typename Parameter>
double Percentage(const Parameter& value, const Parameter& left,
                  const Parameter& right) {
  return left == right ? Parameter{0}
                       : static_cast<double>(value - left) /
                             static_cast<double>(right - left);
}

// Returns the linear interpolation (1-t) * value0 + t * value1.
template <typename Value>
Value InterpolateLinear(double t, const Value& value0, const Value& value1) {
  return (1.0 - t) * value0 + t * value1;
}

// Interpolates linearly in the bounding box given by min and max
// (linear interpolation per coordinate).  Assumes that the parameter vector has
// coefficients in [0, 1].
template <typename Scalar, int Dimension>
Vector<Scalar, Dimension> InterpolateLinearInBox(
    const Vector<Scalar, Dimension>& parameter,
    const Vector<Scalar, Dimension>& min,
    const Vector<Scalar, Dimension>& max) {
  return (Scalar{1} - parameter.array()) * min.array() +
         parameter.array() * max.array();
}

// Interpolates linearly between `value0` and `value1`, using the
// parameterization of value(parameter1) = value1, value(parameter2) = value2.
template <typename Parameter, typename Value>
Value InterpolateLinear(Parameter parameter, const Parameter& parameter0,
                        const Parameter& parameter1, const Value& value0,
                        const Value& value1) {
  return InterpolateLinear(Percentage(parameter, parameter0, parameter1),
                           value0, value1);
}

// Quadratic interpolation
template <typename Value>
Value InterpolateQuadratic(double t, const Value& value0, const Value& value1,
                           const Value& value2) {
  const double t_minus_1_half = 0.5 * (t - 1.0);
  const double t_minus_2 = t - 2.0;
  return (t_minus_1_half * t_minus_2) * value0 - (t_minus_2 * t) * value1 +
         (t_minus_1_half * t) * value2;
}

// Quadratic interpolation
template <typename Parameter, typename Value>
Value InterpolateQuadratic(double t, const Parameter& t0, const Parameter& t1,
                           const Parameter& t2, const Value& v0,
                           const Value& v1, const Value& v2) {
  const double c_t_t0 = t - t0;
  const double c_t_t1 = t - t1;
  const double c_t_t2 = t - t2;
  const double c_t0_t1 = t0 - t1;
  const double c_t0_t2 = t0 - t2;
  const double c_t1_t2 = t1 - t2;
  const double coefficient0 = +c_t_t1 * c_t_t2 / (c_t0_t1 * c_t0_t2);
  const double coefficient1 = -c_t_t0 * c_t_t2 / (c_t0_t1 * c_t1_t2);
  const double coefficient2 = +c_t_t0 * c_t_t1 / (c_t0_t2 * c_t1_t2);
  return coefficient0 * v0 + coefficient1 * v1 + coefficient2 * v2;
}

// Computes cubic hermite spline coefficients for the unit interval [0,1]
//
// See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
inline void CubicHermiteSplineCoefficients(double t, double& coefficientValue0,
                                           double& coefficientDerivative0,
                                           double& coefficientValue1,
                                           double& coefficientDerivative1) {
  const double t2 = t * t;
  coefficientValue0 = (2.0 * t - 3.0) * t2 + 1.0;
  coefficientDerivative0 = (t - 2.0) * t2 + t;
  coefficientValue1 = t2 * (3.0 - 2.0 * t);
  coefficientDerivative1 = t2 * (t - 1.0);
}

// Cubic hermite spline interpolation on the unit interval [0,1]
//
// See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
template <typename Value>
Value InterpolateCubicHermite(double t, const Value& value0,
                              const Value& derivative0, const Value& value1,
                              const Value& derivative1) {
  double coefficientValue0, coefficientDerivative0;
  double coefficientValue1, coefficientDerivative1;
  CubicHermiteSplineCoefficients(t, coefficientValue0, coefficientDerivative0,
                                 coefficientValue1, coefficientDerivative1);
  return coefficientValue0 * value0 + coefficientDerivative0 * derivative0 +
         coefficientValue1 * value1 + coefficientDerivative1 * derivative1;
}

// Cubic hermite spline interpolation (using finite differences for tangents).
//
// Interpolates using four data points (ym,y0,y1,y2) with corresponding
// parameters (tm,t0,t1,t2). Control points must be sorted: tm <= t0 <= t <= t1
// <= t2 The postfix _m stands for _{-1} in the Wikipedia notation.
//
// See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
template <typename Parameter, typename Value>
Value InterpolateCubicHermite(const Parameter& t, const Parameter& tm,
                              const Parameter& t0, const Parameter& t1,
                              const Parameter& t2, const Value& ym,
                              const Value& y0, const Value& y1,
                              const Value& y2) {
  const double trel = Percentage(t, t0, t1);
  double coefficientValue0, coefficientDerivative0;
  double coefficientValue1, coefficientDerivative1;
  CubicHermiteSplineCoefficients(trel, coefficientValue0,
                                 coefficientDerivative0, coefficientValue1,
                                 coefficientDerivative1);
  const double t1_minus_t0_half = 0.5 * (t1 - t0);
  const double t0m = t1_minus_t0_half / static_cast<double>(t0 - tm);
  const double t10 = t1_minus_t0_half / static_cast<double>(t1 - t0);
  const double t21 = t1_minus_t0_half / static_cast<double>(t2 - t1);
  return coefficientValue0 * y0 + (coefficientDerivative0 * t0m) * (y0 - ym) +
         ((coefficientDerivative0 + coefficientDerivative1) * t10) * (y1 - y0) +
         (coefficientDerivative1 * t21) * (y2 - y1) + coefficientValue1 * y1;
}

// Cubic hermite spline interpolation (using Catmull-Rom for tangents).
//
// Interpolates using four data points (ym,y0,y1,y2) with corresponding
// parameters (tm,t0,t1,t2). Control points must be sorted: tm <= t0 <= t <= t1
// <= t2 The postfix _m stands for _{-1} in the Wikipedia notation.
//
// See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
template <typename Parameter, typename Value>
Value InterpolateCatmullRom(const Parameter& t, const Parameter& tm,
                            const Parameter& t0, const Parameter& t1,
                            const Parameter& t2, const Value& ym,
                            const Value& y0, const Value& y1, const Value& y2) {
  double trel = Percentage(t, t0, t1);
  double coefficientValue0, coefficientDerivative0;
  double coefficientValue1, coefficientDerivative1;
  CubicHermiteSplineCoefficients(trel, coefficientValue0,
                                 coefficientDerivative0, coefficientValue1,
                                 coefficientDerivative1);
  double t1_minus_t0 = (t1 - t0);
  double t1m = t1_minus_t0 / static_cast<double>(t1 - tm);
  double t20 = t1_minus_t0 / static_cast<double>(t2 - t0);
  return coefficientValue0 * y0 + (coefficientDerivative0 * t1m) * (y1 - ym) +
         (coefficientDerivative1 * t20) * (y2 - y0) + coefficientValue1 * y1;
}

// Interpolates linearly between two Euclidean vectors.
template <typename Scalar, int Dimension>
Vector<Scalar, Dimension> Interpolate(Scalar p,
                                      const Vector<Scalar, Dimension>& t_a,
                                      const Vector<Scalar, Dimension>& t_b) {
  return InterpolateLinear(p, t_a, t_b);
}

// Interpolates between two elements in SO(2).
//
// If the angle between the two rotations is exactly 180 degrees there are
// theoretically two possible solutions. In this case this implementation
// deterministically chooses one.
template <typename Scalar, int OptionsA, int OptionsB>
SO2<Scalar> Interpolate(Scalar p, const SO2<Scalar, OptionsA>& ref_q_a,
                        const SO2<Scalar, OptionsB>& ref_q_b) {
  const Scalar delta_angle = (ref_q_a.inverse() * ref_q_b).angle();
  return ref_q_a * SO2<Scalar>{p * delta_angle};
}

// Interpolates between two elements in SO(3).
//
// If the angle between the two rotations is exactly 180 degrees there are
// theoretically two possible solutions. In this case this implementation
// deterministically chooses one.
//
// This implementation does not work for extrapolation.
template <typename Scalar, int OptionsA, int OptionsB>
Quaternion<Scalar, kDefaultOptions> Interpolate(
    Scalar p, const Quaternion<Scalar, OptionsA>& ref_q_a,
    const Quaternion<Scalar, OptionsB>& ref_q_b) {
  using std::abs;
  using std::acos;
  using std::sin;
  Scalar p0, p1;
  Scalar cos_omega = ref_q_a.dot(ref_q_b);
  Scalar abs_cos_omega = abs(cos_omega);
  // If angle is too small do only linear interpolation.
  if (abs_cos_omega > Scalar(1) - Eigen::NumTraits<Scalar>::epsilon()) {
    p0 = Scalar(1) - p;
    p1 = p;
  } else {
    // We are all good and can do slerp.
    Scalar omega = acos(abs_cos_omega);
    Scalar sin_omega = sin(omega);
    p0 = sin((Scalar(1) - p) * omega) / sin_omega;
    p1 = sin(p * omega) / sin_omega;
  }
  // Pick the right direction.
  if (cos_omega < 0) {
    p1 = -p1;
  }
  return Quaternion<Scalar, kDefaultOptions>{p0 * ref_q_a.coeffs() +
                                             p1 * ref_q_b.coeffs()};
}

// Interpolates between two Pose2 transformations.
//
// This interpolates translation and rotation independently from each other.
template <typename Scalar, int OptionsA, int OptionsB>
Pose2<Scalar> Interpolate(Scalar p, const Pose2<Scalar, OptionsA>& ref_pose_a,
                          const Pose2<Scalar, OptionsB>& ref_pose_b) {
  return Pose2<Scalar>{
      Interpolate(p, ref_pose_a.translation(), ref_pose_b.translation()),
      Interpolate(p, ref_pose_a.so2(), ref_pose_b.so2())};
}

// Interpolates between two Pose3 transformations
//
// This interpolates translation and rotation independently from each other.
template <typename Scalar, int OptionsA, int OptionsB>
Pose3<Scalar> Interpolate(Scalar p, const Pose3<Scalar, OptionsA>& ref_pose_a,
                          const Pose3<Scalar, OptionsB>& ref_pose_b) {
  return Pose3<Scalar>{
      Interpolate(p, ref_pose_a.quaternion(), ref_pose_b.quaternion()),
      Interpolate(p, ref_pose_a.translation(), ref_pose_b.translation())};
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_INTERPOLATION_H_
