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

#ifndef EIGENMATH_EIGENMATH_POSE3_H_
#define EIGENMATH_EIGENMATH_POSE3_H_

#include <iostream>
#include <ostream>

#include "Eigen/Core"      // IWYU pragma: keep
#include "Eigen/Geometry"  // IWYU pragma: keep
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "rotation_utils.h"  // IWYU pragma: export
#include "so3.h"
#include "types.h"

namespace eigenmath {

// Transformation for 3D space
template <typename Scalar, int Options = kDefaultOptions>
class Pose3 {
 public:
  template <int OtherOptions>
  using Quaternion = Quaternion<Scalar, OtherOptions>;

  // Identity pose
  EIGEN_DEVICE_FUNC Pose3() : translation_(ZeroTranslation()) {}

  // Creates a pose with given translation
  EIGEN_DEVICE_FUNC explicit Pose3(const Vector3<Scalar>& translation)
      : translation_{translation}, rotation_{} {}

  // Creates a pose with given rotation
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC explicit Pose3(const SO3<Scalar, OtherOptions>& rotation)
      : translation_{ZeroTranslation()}, rotation_{rotation} {}

  // Creates a pose with given rotation
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC explicit Pose3(const Quaternion<OtherOptions>& rotation)
      : translation_{ZeroTranslation()}, rotation_{rotation} {}

  // Creates a pose from a 3 x 3 rotation matrix
  EIGEN_DEVICE_FUNC explicit Pose3(const Matrix3<Scalar>& rotation_matrix)
      : translation_(ZeroTranslation()), rotation_(rotation_matrix) {}

  // Creates a pose with given translation and rotation
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC Pose3(const SO3<Scalar, OtherOptions>& rotation,
                          const Vector3<Scalar>& translation)
      : translation_{translation}, rotation_{rotation} {}

  // Creates a pose with given translation and rotation.
  // The \p policy provides control over whether the rotation quaternion should
  // be normalized again or not. This can be relevant, e.g., to guarantee
  // bit-true representations on deserialization.
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC Pose3(const Quaternion<OtherOptions>& rotation,
                          const Vector3<Scalar>& translation,
                          const NormalizationPolicy policy = kNormalize)
      : translation_{translation}, rotation_(rotation, policy == kNormalize) {}

  // Creates a pose from a 3 x 3 rotation matrix and a translation vector
  EIGEN_DEVICE_FUNC Pose3(const Matrix3<Scalar>& rotation_matrix,
                          const Vector3<Scalar>& translation)
      : translation_(translation), rotation_(rotation_matrix) {}

  // Creates a pose from a 4 x 4 affine transformation matrix
  template <int OtherOptions = kDefaultOptions>
  EIGEN_DEVICE_FUNC explicit Pose3(const Matrix4<Scalar, OtherOptions>& affine)
      : translation_(affine.template topRightCorner<3, 1>().eval()),
        rotation_(affine.template topLeftCorner<3, 3>().eval()) {}

  // Conversion operator for other Pose3 types with different
  // Eigen::Options
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC Pose3(const Pose3<Scalar, OtherOptions>& other)  // NOLINT
      : translation_{other.translation()}, rotation_{other.so3()} {}

  // Assignment operator for other Pose3 types with different Eigen::Options.
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC Pose3& operator=(const Pose3<Scalar, OtherOptions>& other) {
    translation_ = other.translation();
    rotation_ = other.so3();
    return *this;
  }

  // Gives identity pose, same as default constructor, more readable.
  EIGEN_DEVICE_FUNC static Pose3<Scalar, Options> Identity() {
    return Pose3<Scalar, Options>{};
  }

  // 3D translation vector
  EIGEN_DEVICE_FUNC const Vector3<Scalar>& translation() const {
    return translation_;
  }
  EIGEN_DEVICE_FUNC Vector3<Scalar>& translation() { return translation_; }

  // 3D rotation
  EIGEN_DEVICE_FUNC const SO3<Scalar, Options>& so3() const {
    return rotation_;
  }
  EIGEN_DEVICE_FUNC SO3<Scalar, Options>& so3() { return rotation_; }

  // Quaternion
  EIGEN_DEVICE_FUNC const Quaternion<Options>& quaternion() const {
    return rotation_.quaternion();
  }

  // Sets the rotation using quaternion
  template <int OtherOptions = kDefaultOptions>
  EIGEN_DEVICE_FUNC void setQuaternion(
      const Quaternion<OtherOptions>& quaternion) {
    rotation_ = quaternion;
  }

  // Gives the 3D rotation matrix
  EIGEN_DEVICE_FUNC Matrix3<Scalar> rotationMatrix() const {
    return rotation_.matrix();
  }

  // Sets the 3D rotation matrix
  EIGEN_DEVICE_FUNC void setRotationMatrix(
      const Matrix3<Scalar>& rotation_matrix) {
    rotation_ = SO3<Scalar, Options>(rotation_matrix);
  }

  // x-axis of the coordinate frame
  EIGEN_DEVICE_FUNC Vector3<Scalar> xAxis() const {
    return rotation_.matrix().template block<3, 1>(0, 0);
  }

  // y-axis of the coordinate frame
  EIGEN_DEVICE_FUNC Vector3<Scalar> yAxis() const {
    return rotation_.matrix().template block<3, 1>(0, 1);
  }

  // z-axis of the coordinate frame
  EIGEN_DEVICE_FUNC Vector3<Scalar> zAxis() const {
    return rotation_.matrix().template block<3, 1>(0, 2);
  }

  // Inverse transformation
  EIGEN_DEVICE_FUNC Pose3<Scalar> inverse() const {
    SO3<Scalar> rotation_inverse = rotation_.inverse();
    return Pose3<Scalar>{rotation_inverse, -(rotation_inverse * translation_)};
  }

  // Returns an affine 4x4 transformation matrix
  EIGEN_DEVICE_FUNC Matrix4<Scalar> matrix() const {
    Matrix4<Scalar> M = Matrix4<Scalar>::Identity();
    M.template topLeftCorner<3, 3>() = rotationMatrix();
    M.template topRightCorner<3, 1>() = translation();
    return M;
  }

  // Cast Pose3 instance to other scalar type
  template <typename OtherScalar>
  EIGEN_DEVICE_FUNC Pose3<OtherScalar> cast() const {
    return Pose3<OtherScalar>(rotation_.template cast<OtherScalar>(),
                              translation_.template cast<OtherScalar>());
  }

  // Checks if identical to another pose under a given tolerance
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC bool isApprox(const Pose3<Scalar, OtherOptions>& other,
                                  Scalar tolerance) const {
    return isApprox(other, tolerance, tolerance);
  }

  // Checks if identical to another pose under a given tolerance
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC bool isApprox(const Pose3<Scalar, OtherOptions>& other,
                                  Scalar linear_tolerance,
                                  Scalar angular_tolerance) const {
    return ((translation() - other.translation()).norm() < linear_tolerance) &&
           so3().isApprox(other.so3(), angular_tolerance);
  }

  // Checks if identical to another pose under default tolerance
  template <int OtherOptions>
  EIGEN_DEVICE_FUNC bool isApprox(
      const Pose3<Scalar, OtherOptions>& other) const {
    return isApprox(other, Eigen::NumTraits<Scalar>::dummy_precision());
  }

  // Compose poses in-place
  template <typename OtherScalar, int OtherOptions>
  EIGEN_DEVICE_FUNC Pose3<Scalar, Options>& operator*=(
      const Pose3<OtherScalar, OtherOptions>& rhs) {
    translation_ += rotation_ * rhs.translation();
    rotation_ *= rhs.so3();
    return *this;
  }

  // Composes two poses
  template <typename OtherScalar, int OtherOptions>
  EIGEN_DEVICE_FUNC Pose3<Scalar> operator*(
      const Pose3<OtherScalar, OtherOptions>& rhs) const {
    Pose3<Scalar> result(*this);
    return result *= rhs;
  }

 private:
  EIGEN_DEVICE_FUNC static Vector3<Scalar> ZeroTranslation() {
    return Vector3<Scalar>::Zero();
  }

  Vector3<Scalar> translation_;
  // Padding makes aligned and unaligned types match in memory layout.
  Scalar padding_ = Scalar{0.0};
  SO3<Scalar, Options> rotation_;
};

template <typename Sink, typename T, int Options>
void AbslStringify(Sink& sink, const Pose3<T, Options>& a_pose_b) {
  absl::Format(&sink, "translation: %v quaternion: %v",
               eigenmath::AbslStringified(a_pose_b.translation()),
               eigenmath::AbslStringified(a_pose_b.quaternion().coeffs()));
}

template <typename T, int Options>
std::ostream& operator<<(std::ostream& os, const Pose3<T, Options>& a_pose_b) {
  os << absl::StrCat(a_pose_b);
  return os;
}

// Transforms a 3D point with a 3D transformation
template <typename Scalar, int Options, typename Derived>
EIGEN_DEVICE_FUNC Vector<Scalar, 3> operator*(
    const Pose3<Scalar, Options>& pose,
    const Eigen::MatrixBase<Derived>& point) {
  return pose.translation() + pose.so3() * point;
}

//  Pose3 class using doubles
using Pose3d = Pose3<double>;
//  Pose3 class using floats
using Pose3f = Pose3<float>;

//  Creates a pose from angle axis representation
template <typename T>
Pose3<T> CreateAngleAxisPose(T angle, const Vector3<T>& axis,
                             const Vector3<T>& position) {
  return Pose3<T>(Eigen::AngleAxis<T>(angle, axis).matrix(), position);
}

//  Creates a pose from angle axis representation
template <typename T>
Pose3<T> CreateAngleAxisPose(T angle, const Vector3<T>& axis) {
  return Pose3<T>(Eigen::AngleAxis<T>(angle, axis).matrix());
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_POSE3_H_
