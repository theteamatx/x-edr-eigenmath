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

#ifndef EIGENMATH_EIGENMATH_SCALAR_UTILS_H_
#define EIGENMATH_EIGENMATH_SCALAR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/base/macros.h"

namespace eigenmath {

// Saturate value to -min_max, min_max.
template <typename T>
inline constexpr T Saturate(const T& val, const T& min_max) {
  return std::clamp(val, -min_max, min_max);
}

// Converts the argument from degrees to radians
template <typename T>
inline constexpr auto Radians(T deg) {
  if constexpr (sizeof(T) > sizeof(float)) {
    return static_cast<double>(M_PI) * (static_cast<double>(deg) / 180.0);
  } else {
    return static_cast<float>(M_PI) * (static_cast<float>(deg) / 180.0f);
  }
}

// Converts the argument from radians to degrees
template <typename T>
inline constexpr auto Degrees(T rad) {
  if constexpr (sizeof(T) > sizeof(float)) {
    return 180.0 * (static_cast<double>(rad) / static_cast<double>(M_PI));
  } else {
    return 180.0f * (static_cast<float>(rad) / static_cast<float>(M_PI));
  }
}

// Computes the cardinal sine function "sin(x) / x".
// This implementation avoids the singularity at 0.
template <typename T>
inline T Sinc(T x) {
  using std::abs;
  using std::sin;
  if (abs(x) < std::numeric_limits<T>::epsilon()) {
    return T(1.0);
  }
  return sin(x) / x;
}

// Computes the function "(1 - cos(x)) / x".
// This implementation avoids the singularity at 0.
template <typename T>
inline T OneMinusCosOverX(T x) {
  using std::abs;
  using std::cos;
  if (abs(x) < std::numeric_limits<T>::epsilon()) {
    return T(0.0);
  }
  return (T(1.0) - cos(x)) / x;
}

// Approximate exponential function on restricted input range [-inf, 0].
// The approximation is within 1e-3 of std::exp(), and the output is guaranteed
// to lie in [0, 1] for valid input values.
//
// It is about 4x faster than std::exp(), see benchmarks.
inline constexpr double ApproximateExp(double x) {
  constexpr int kExponent = 10;
  constexpr double kApproximationOrder = 1 << kExponent;
  double result =
      1.0 + std::clamp(x, -kApproximationOrder, 0.0) / kApproximationOrder;
  // Calculate result ^ kApproximationOrder.
  for (int i = 0; i < kExponent; ++i) {
    result *= result;
  }
  return result;
}

// Wraps to the open interval [0, upper_bound).
// Wrap for floating point types.
template <typename T>
inline constexpr T Wrap(T x, T upper_bound) {
  static_assert(std::is_floating_point_v<T>,
                "Only floating point values are supported.");
  return x - (upper_bound * std::floor(x / upper_bound));
}

// Wraps to the open interval [lower,upper).
template <typename T>
inline constexpr T Wrap(T x, T lower_bound, T upper_bound) {
  return lower_bound + Wrap(x - lower_bound, upper_bound - lower_bound);
}

// Return the square of the input.
template <class T>
inline constexpr T Square(T a) {
  return a * a;
}

// Combine two probabilities assuming that their causes are uncorrelated.
// For example:
//  p1 = P(A|B)*P(B)   &   p2 = P(A|C)*P(C)
// with B and C being independent (corr(B,C) ~= 0). The result is:
//  p_result = P(A|B or C)*P(B or C) ~= (1 - p1) * p2 + p1
template <typename T>
constexpr T CombineIndependentProbabilities(T p1, T p2) {
  // Max function is needed to account for p1 or p2 being greater than 1.
  using std::max;
  return max(p1, max(p2, p1 + p2 - p1 * p2));
}

// Combine two probabilities assuming that their causes are nearly perfectly
// correlated. For example:
//  p1 = P(A|B)*P(B)   &   p2 = P(A|C)*P(C)
// with B and C being dependent (corr(B,C) ~= 1). The result is:
//  p_result = P(A|B or C)*P(B or C) ~= max(p1, p2)
template <typename T>
constexpr T CombineDependentProbabilities(T p1, T p2) {
  using std::max;
  return max(p1, p2);
}

// Compute the roots of a quadratic expression.
// Solves: a*x^2 + b*x + c = 0
// Returns the number of real roots found:
//  - -1 if the quadratic is ill-formed (a == 0 and b == 0)
//  - 0 if roots are complex (root1_re +- root_im * i)
//  - 1 if there are real repeated roots
//  - 2 if there are 2 real roots
// If the imaginary parts are set to null (default), then complex roots are
// not calculated.
template <typename T>
int ComputeQuadraticRoots(T a, T b, T c, T* root1_re, T* root2_re,
                          T* root_im = nullptr) {
  // Some ADL-friendly using clauses (because we are in a function template).
  using std::abs;
  using std::sqrt;
  using std::swap;
  if (root_im != nullptr) {
    *root_im = 0.0;
  }
  if (abs(a) < std::numeric_limits<T>::epsilon()) {
    // Linear equation.
    if (abs(b) < std::numeric_limits<T>::epsilon()) {
      // Infeasible equation (0*x^2 + 0*x + c = 0).
      return -1;
    }
    *root1_re = -c / b;
    *root2_re = *root1_re;
    return 1;
  }
  const T discriminant = b * b - 4.0 * a * c;
  if (discriminant < -std::numeric_limits<T>::epsilon()) {
    // Complex roots, root1_re +- root_im * i.
    if (root_im != nullptr) {
      *root1_re = -b / (2.0 * a);
      *root2_re = *root1_re;
      *root_im = sqrt(-discriminant) / (2.0 * a);
    }
    return 0;
  }
  if (discriminant > std::numeric_limits<T>::epsilon()) {
    // Two real roots.
    const T sqrt_disc = sqrt(discriminant);
    if (b > 0.0) {
      *root1_re = (-b - sqrt_disc) / (2.0 * a);
    } else {
      *root1_re = (-b + sqrt_disc) / (2.0 * a);
    }
    *root2_re = c / (a * (*root1_re));
    if (*root1_re > *root2_re) {
      swap(*root1_re, *root2_re);
    }
    return 2;
  }
  // One real repeated root.
  *root1_re = -b / (2.0 * a);
  *root2_re = *root1_re;
  return 1;
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SCALAR_UTILS_H_
