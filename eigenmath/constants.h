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

#ifndef EIGENMATH_EIGENMATH_CONSTANTS_H_
#define EIGENMATH_EIGENMATH_CONSTANTS_H_

#include <cmath>
#include <type_traits>

namespace eigenmath {

// Converts `degree` to radian.
template <typename T>
constexpr T RadianFromDegree(T degree) {
  static_assert(std::is_floating_point_v<T>,
                "RadianFromDegree should use floating point type");
  constexpr T one_degree_in_radian{M_PI / 180};
  return degree * one_degree_in_radian;
}

// Converts `radian` to degree.
template <typename T>
constexpr T DegreeFromRadian(T radian) {
  static_assert(std::is_floating_point_v<T>,
                "DegreeFromRadian should use floating point type");
  constexpr T one_radian_in_degree{180 / M_PI};
  return radian * one_radian_in_degree;
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_CONSTANTS_H_
