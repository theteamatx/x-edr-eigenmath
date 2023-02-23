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

#ifndef EIGENMATH_EIGENMATH_SIMILARITY3_H_
#define EIGENMATH_EIGENMATH_SIMILARITY3_H_

#include "absl/log/check.h"
#include "pose3.h"
#include "types.h"

namespace eigenmath {

// Similarity transform in 3 dimensions. This is a shape preserving transform.
//
// The scale factor must be positive.
//
// In this implementation, Similarity3 is simply a pair of a Pose3 and a scale
// factor.
template <typename T, int Options = kDefaultOptions>
class Similarity3 {
 public:
  Similarity3() : scale_(1) {}

  // Conversion operator for other Similarity3 types with different
  // Eigen::Options
  template <int OtherOptions>
  Similarity3(T scale, const Pose3<T, OtherOptions>& pose)
      : scale_(scale), pose_(pose) {
    CHECK_GT(scale, 0) << "scale factor must be positive";
  }

  Vector3<T> operator*(const Vector3<T>& point_a) const {
    Vector3<T> scaledpoint_a = scale_ * point_a;
    return pose_ * scaledpoint_a;
  }

  const Pose3<T, Options>& pose() const { return pose_; }

  T scale() const { return scale_; }

  Matrix<T, 4, 4> matrix() const {
    Matrix<T, 4, 4> m = pose_.matrix();
    m.template topLeftCorner<3, 3>() *= scale_;
    return m;
  }

  // Checks if identical to another similarity.
  template <int OtherOptions>
  bool isApprox(const Similarity3<T, OtherOptions>& other, T tolerance) const {
    return matrix().isApprox(other.matrix(), tolerance);
  }

  // Checks if identical to another similarity using default tolerance.
  template <int OtherOptions>
  bool isApprox(const Similarity3<T, OtherOptions>& other) const {
    return matrix().isApprox(other.matrix());
  }

 private:
  // Scale factor.
  T scale_;

  // Rigid body pose in three dimensions.
  Pose3<T, Options> pose_;
};

using Similarity3d = Similarity3<double>;
using Similarity3f = Similarity3<float>;

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SIMILARITY3_H_
