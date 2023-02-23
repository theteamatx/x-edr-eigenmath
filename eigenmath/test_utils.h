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

#ifndef EIGENMATH_EIGENMATH_TEST_UTILS_H_
#define EIGENMATH_EIGENMATH_TEST_UTILS_H_

#include "manifolds.h"
#include "pose3.h"
#include "so3.h"
#include "types.h"

namespace eigenmath {

// Compares 2 vectors by subtracting and calculating the norm.
//
// Note: this avoids Eigen::Matrix::isApprox() which returns false if
// the sign of any component is different or if the value is very close to 0.
//
template <typename T, typename DerivedValue, typename DerivedExpected>
static inline bool isNormApprox(
    const Eigen::MatrixBase<DerivedValue> &value,
    const Eigen::MatrixBase<DerivedExpected> &expected, T epsilon) {
  const T delta = (value - expected).norm();
  return delta <= epsilon;
}

// Compare 2 SO3 by comparing logSO3 of first times inverse of second.
//
// Note: this avoids Eigen::Matrix::isApprox() which return false if
// the sign of any component is different or if the value is very close to 0.
//
template <typename T, int OptionsValue, int OptionsExpected>
static inline bool isNormApprox(const SO3<T, OptionsValue> &value,
                                const SO3<T, OptionsExpected> &expected,
                                T epsilon) {
  return isNormApprox(LogSO3(value * expected.inverse()), Vector3<T>(0, 0, 0),
                      epsilon);
}

// Compare 2 SO3 by comparing logSO3 of first times inverse of second.
//
// Note: this avoids Eigen::Matrix::isApprox() which return false if
// the sign of any component is different or if the value is very close to 0.
template <typename T, int OptionsValue, int OptionsExpected>
static inline bool isNormApprox(const Pose3<T, OptionsValue> &value,
                                const Pose3<T, OptionsExpected> &expected,
                                T epsilon) {
  return isNormApprox(value.so3(), expected.so3(), epsilon) &&
         isNormApprox(value.translation(), expected.translation(), epsilon);
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_TEST_UTILS_H_
