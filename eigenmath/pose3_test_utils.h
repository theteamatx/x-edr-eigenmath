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

#ifndef EIGENMATH_EIGENMATH_POSE3_TEST_UTILS_H_
#define EIGENMATH_EIGENMATH_POSE3_TEST_UTILS_H_

#include <iomanip>
#include <sstream>
#include <string>

#include "pose3.h"

namespace eigenmath {

template <class T, int Options_a, int Options_b>
std::string Pose3ComparisonToString(const Pose3<T, Options_a>& a,
                                    const Pose3<T, Options_b>& b) {
  std::stringstream out;
  const Pose3<T> combined = a.inverse() * b;

  out << std::setprecision(4);
  out << "translation (diff norm: " << combined.translation().norm()
      << " meters)\n";
  for (int i = 0; i < 3; i++) {
    out << a.translation()(i) << "\t| " << b.translation()(i) << std::endl;
  }

  Eigen::AngleAxis<T> rotation;
  rotation.fromRotationMatrix(combined.rotationMatrix());
  out << "rotation (diff angle: " << rotation.angle() << " rad)\n";
  for (int i = 0; i < 4; i++) {
    out << a.quaternion().coeffs()(i) << "\t| " << b.quaternion().coeffs()(i)
        << std::endl;
  }
  return out.str();
}

template <typename T>
Matrix6<T> TestCovariance(const T t) {
  return Eigen::DiagonalMatrix<T, 6>(t, 2 * t, 3 * t, 4 * t, 5 * t, 6 * t);
}

template <typename T>
Pose3<T> TestPose(const T t) {
  return {QuaternionFromRPY(t, t / 2, t / 3), Vector3d(t, 2 * t, 3 * t)};
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_POSE3_TEST_UTILS_H_
