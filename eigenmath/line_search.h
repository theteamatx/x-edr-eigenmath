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

#ifndef EIGENMATH_EIGENMATH_LINE_SEARCH_H_
#define EIGENMATH_EIGENMATH_LINE_SEARCH_H_

#include <cmath>
#include <type_traits>
#include <utility>

namespace eigenmath {

// This is an implementation of the Golden-section search algorithm.
// Searches for the minimum of a given function within a given interval.
// This algorithm assumes that the function f is strictly unimodal.
//
// This function will never fail and is guaranteed to find a solution in
// a finite number of steps, determined by the ratio between the initial
// interval size and the tolerance.
//
// Returns a std::pair<Scalar, Scalar>, where `second` is the smallest function
// value and `first` the corresponding function argument.
template <typename Functor, typename Scalar>
std::pair<Scalar, Scalar> GoldenSectionSearchMinimize(Scalar left, Scalar right,
                                                      Functor f,
                                                      Scalar x_tolerance) {
  static_assert(std::is_floating_point_v<Scalar>,
                "Scalar must be a floating point type.");

  constexpr Scalar kGoldenRatio = Scalar{1.618033988};

  if (right < left) {
    // std::swap is not supported in cuda, so swap left and right manually.
    const Scalar tmp = left;
    left = right;
    right = tmp;
  }

  Scalar mid_left = (right + kGoldenRatio * left) / (Scalar{1} + kGoldenRatio);
  Scalar mid_right = (left + kGoldenRatio * right) / (Scalar{1} + kGoldenRatio);

  Scalar mid_left_f = f(mid_left);
  Scalar mid_right_f = f(mid_right);

  x_tolerance = std::abs(x_tolerance);
  while (std::abs(right - left) >= x_tolerance) {
    if (mid_right_f < mid_left_f) {
      left = mid_left;
      mid_left = mid_right;
      mid_left_f = mid_right_f;
      mid_right = (left + kGoldenRatio * right) / (Scalar{1} + kGoldenRatio);
      mid_right_f = f(mid_right);
    } else {
      right = mid_right;
      mid_right = mid_left;
      mid_right_f = mid_left_f;
      mid_left = (right + kGoldenRatio * left) / (Scalar{1} + kGoldenRatio);
      mid_left_f = f(mid_left);
    }
  }

  const Scalar best_x = (mid_right + mid_left) * Scalar{0.5};
  const Scalar best_f = (mid_left_f + mid_right_f) * Scalar{0.5};
  return {best_x, best_f};
}

// This is an implementation of the bisection search algorithm.
// Searches for the change of sign of a given function within a given interval.
// This algorithm assumes that there is only a single change of sign.
//
// This function will never fail and is guaranteed to find a solution in
// a finite number of steps, determined by the ratio between the initial
// interval size and the tolerance.
//
// Returns:
//  -1 if the function value at each side is negative.
//   1 if the function value at each side is positive.
//   0 otherwise, and a solution was found.
template <typename Functor>
int BisectionSearchZeroCross(double left, double right, Functor f,
                             double x_tolerance, double *best_left_x,
                             double *best_right_x) {
  double left_f = f(left);
  double right_f = f(right);

  if (std::signbit(left_f) == std::signbit(right_f)) {
    return (std::signbit(left_f) ? -1 : 1);
  }

  x_tolerance = std::abs(x_tolerance);
  while (std::abs(left - right) > x_tolerance) {
    const double mid_x = (left + right) / 2.0;
    const double mid_f = f(mid_x);
    if (std::signbit(mid_f) == std::signbit(left_f)) {
      left = mid_x;
      left_f = mid_f;
    } else {
      right = mid_x;
      right_f = mid_f;
    }
  }
  if (best_left_x) {
    *best_left_x = left;
  }
  if (best_right_x) {
    *best_right_x = right;
  }
  return 0;
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_LINE_SEARCH_H_
