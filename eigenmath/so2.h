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

#ifndef EIGENMATH_EIGENMATH_SO2_H_
#define EIGENMATH_EIGENMATH_SO2_H_

#include <cmath>
#include <ostream>
#include <sstream>
#include <type_traits>

#include "Eigen/Core"
#include "absl/log/check.h"
#include "rotation_utils.h"
#include "types.h"
#include "utils.h"

namespace eigenmath {

namespace details {

// This is essentially a workaround for ceres::Jet<>. While jets do
// provide a check for finite values, it is named IsFinite instead of
// isfinite. So, in ceresutils/poses.h there is a specialization of
// this struct which calls ceres::IsFinite<> instead.
template <typename Scalar>
struct IsFinite {
  bool operator()(const Scalar& value) const {
    using std::isfinite;
    return isfinite(value);
  }
};

template <typename Scalar>
bool isFinite(const Scalar& value) {
  return IsFinite<Scalar>()(value);
}

}  // namespace details

// A representation of 2D rotations using complex numbers
//
// The SO2 object for a rotation angle \f$\theta\f$ is defined as
//    @f$ q = cos(\theta) + I sin(\theta) @f$
//
// In comparison to an angle representation this avoids the need to wrap angles
// and most operations can be performed with minimal runtime costs.
//
// Only the following functions have non-trivial runtime costs:
//    - SO2(angle)
//    - SO2(Matrix2)
//    - SO2(sin_angle, cos_angle)
//    - angle()
template <typename Scalar, int Options = kDefaultOptions>
class SO2 {
 public:
  // Initializes to the identity rotation
  SO2() : SO2{Scalar(1), Scalar(0), false} {}

  // Initializes with the given rotation angle
  //
  // This function calls sine and cosine.
  explicit SO2(Scalar angle) {
    CHECK(details::isFinite(angle));
    using std::cos;
    using std::sin;
    cos_sin_angle_[0] = cos(angle);
    cos_sin_angle_[1] = sin(angle);
  }

  // Initializes with a rotation matrix
  //
  // This function performs a singular value decomposition on the given matrix.
  template <int OtherOptions>
  explicit SO2(const Matrix2<Scalar, OtherOptions>& matrix) {
    CHECK(details::isFinite(matrix.sum()));
    Matrix2<Scalar> R = OrthogonalizeRotationMatrix(matrix);
    cos_sin_angle_[0] = R(0, 0);
    cos_sin_angle_[1] = R(1, 0);
    CHECK(details::isFinite(cos_sin_angle_[0]));
    CHECK(details::isFinite(cos_sin_angle_[1]));
  }

  // Initializes using cosine and sine of a rotation angle
  //
  // Either do_normalize must be true, or [cos_angle, sin_angle] must be
  // normalized.
  //
  // By default the given cosine and sine are normalized.
  SO2(Scalar cos_angle, Scalar sin_angle, bool do_normalize = true)
      : cos_sin_angle_(cos_angle, sin_angle) {
    CHECK(details::isFinite(cos_angle));
    CHECK(details::isFinite(sin_angle));
    if (do_normalize) {
      cos_sin_angle_.normalize();
    }
    CHECK(isNormalized()) << "Must be normalized after constructor.";
  }

  // Conversion operator for other SO2 types with different Eigen::Options.
  template <int OtherOptions>
  SO2(const SO2<Scalar, OtherOptions>& other) {  // NOLINT
    // Do not assert IsFinite. The only way to get values into the
    // other instance is via ctors which do check, or via
    // const_cast<> in which case people are on their own.
    cos_sin_angle_[0] = other.cos_angle();
    cos_sin_angle_[1] = other.sin_angle();
  }

  // Assignment operator for other SO2 types with different Eigen::Options.
  template <int OtherOptions>
  SO2& operator=(const SO2<Scalar, OtherOptions>& other) {
    // Do not assert IsFinite. The only way to get values into the
    // other instance is via ctors which do check, or via
    // const_cast<> in which case people are on their own.
    cos_sin_angle_[0] = other.cos_angle();
    cos_sin_angle_[1] = other.sin_angle();
    return *this;
  }

  // The cosine of the rotation angle.
  constexpr Scalar cos_angle() const { return cos_sin_angle_[0]; }

  // The sine of the rotation angle.
  constexpr Scalar sin_angle() const { return cos_sin_angle_[1]; }

  // Computes and returns the rotation angle in the interval
  // \f$[-\pi|+\pi]\f$
  Scalar angle() const {
    using std::atan2;
    return atan2(sin_angle(), cos_angle());
  }

  // Computes and returns the magnitude of the rotation in radians.
  Scalar norm() const {
    using std::abs;
    return abs(angle());
  }

  // Returns the corresponding 2D rotation matrix.
  Matrix2<Scalar> matrix() const {
    return MakeMatrix<Scalar, 2, 2>(
        {{cos_angle(), -sin_angle()}, {sin_angle(), cos_angle()}});
  }

  // The inverse rotation.
  constexpr SO2<Scalar> inverse() const {
    return SO2<Scalar>{cos_angle(), -sin_angle(), kUnsafeCtor};
  }

  // Normalize the representation.
  void normalize() { cos_sin_angle_.normalize(); }

  // Check whether the representation is normalized.
  bool isNormalized() const {
    using std::abs;
    return abs(cos_sin_angle_.squaredNorm() - Scalar(1)) <
           Eigen::NumTraits<Scalar>::dummy_precision();
  }

  // Returns a vector holding cosine and sine of the rotation angle.
  const Vector2<Scalar, Options>& coeffs() const { return cos_sin_angle_; }

  // Returns the x-axis of the coordinate frame.
  const Vector2<Scalar, Options>& xAxis() const { return cos_sin_angle_; }

  // Returns the y-axis of the coordinate frame.
  Vector2<Scalar> yAxis() const {
    return Vector2<Scalar>(-sin_angle(), cos_angle());
  }

  // Cast SO2 instance to other scalar type.
  template <typename OtherScalar>
  SO2<OtherScalar> cast() const {
    if constexpr (std::is_same_v<OtherScalar, Scalar>) {
      return *this;
    } else {
      // force normalize call inside constructor
      // (crucial when going from lower precision to higher precision)
      return SO2<OtherScalar>(static_cast<OtherScalar>(cos_sin_angle_[0]),
                              static_cast<OtherScalar>(cos_sin_angle_[1]));
    }
  }

  // Checks if identical to another pose under a given tolerance.
  bool isApprox(const SO2& other, Scalar tolerance) const {
    return (inverse() * other).norm() < tolerance;
  }

  // Checks if identical to another pose under default tolerance.
  bool isApprox(const SO2& other) const {
    return isApprox(other, Eigen::NumTraits<Scalar>::dummy_precision());
  }

 private:
  Vector2<Scalar, Options> cos_sin_angle_;

  enum UnsafeCtorSignal { kUnsafeCtor };

  // Used to more efficiently construct an SO2 object when the invariants are
  // already guaranteed to hold by construction.
  constexpr SO2(Scalar cos_angle, Scalar sin_angle, UnsafeCtorSignal /*unused*/)
      : cos_sin_angle_(cos_angle, sin_angle) {}
};

// outputs a SO2 to an ostream.
template <typename T, int Options>
std::ostream& operator<<(std::ostream& os,
                         const SO2<T, Options>& a_rotation_b) {
  os << "angle: " << a_rotation_b.angle();
  return os;
}

// Multiplies two rotations.
template <typename LhsScalar, int LhsOptions, typename RhsScalar,
          int RhsOptions>
auto operator*(const SO2<LhsScalar, LhsOptions>& lhs,
               const SO2<RhsScalar, RhsOptions>& rhs) {
  using ResultScalar = std::common_type_t<LhsScalar, RhsScalar>;
  SO2<ResultScalar> product = {
      lhs.cos_angle() * rhs.cos_angle() - lhs.sin_angle() * rhs.sin_angle(),
      rhs.cos_angle() * lhs.sin_angle() + lhs.cos_angle() * rhs.sin_angle()};
  // Assuming the length of cos_sin_angle_ is 1 + epsilon due to
  // floating point rounding errors, this code reduces the error to
  // epsilon^3 / 32.
  const ResultScalar nsq = product.coeffs().squaredNorm();
  using std::abs;
  if (abs(nsq - ResultScalar(1)) >
      Eigen::NumTraits<ResultScalar>::dummy_precision()) {
    const_cast<Vector2<ResultScalar>&>(product.coeffs()) *=
        (ResultScalar(3) + nsq) / (ResultScalar(1) + ResultScalar(3) * nsq);
  }
  return product;
}

// Rotates a 2D vector.
template <typename Scalar, int Options, typename Derived>
constexpr Vector2<Scalar> operator*(const SO2<Scalar, Options>& rotation,
                                    const Eigen::MatrixBase<Derived>& point) {
  static_assert(
      Derived::RowsAtCompileTime == 2 && Derived::ColsAtCompileTime == 1,
      "Vector must be two-dimensional");
  const Scalar point_x = point.x();
  const Scalar point_y = point.y();
  return {rotation.cos_angle() * point_x - point_y * rotation.sin_angle(),
          rotation.cos_angle() * point_y + point_x * rotation.sin_angle()};
}

// Checks if angle falls within a given interval. The lower angle must
// be lower than the upper angle (no wrap around), i.e., for an angular range
// going from pi/2 to 3*pi/2, DO NOT set the upper angle to -pi/2.
template <typename Scalar, int Options>
bool IsInInterval(const SO2<Scalar, Options>& so2, Scalar lower_angle,
                  Scalar upper_angle) {
  if (upper_angle < lower_angle) {
    return false;
  }
  Scalar self_angle = so2.angle();
  // Bring self_angle to within 2*M_PI above lower_angle.
  constexpr Scalar two_pi(2.0 * M_PI);
  while (self_angle < lower_angle) {
    self_angle += two_pi;
  }
  while (self_angle > lower_angle + two_pi) {
    self_angle -= two_pi;
  }
  return self_angle <= upper_angle;
}

using SO2d = SO2<double>;
using SO2f = SO2<float>;

// Computes a unit quaternion with Z as rotation axis from an SO2 element.
template <typename Scalar, int Options>
Quaternion<Scalar> QuaternionZ(const SO2<Scalar, Options>& so2) {
  using std::sqrt;
  // Get sin^2(half angle) and cos^2(half angle).
  const Scalar squared_sin_cos[] = {(1 - so2.cos_angle()) / 2,
                                    (1 + so2.cos_angle()) / 2};
  const Scalar sin_sign = std::signbit(so2.sin_angle()) ? -1 : 1;
  const Scalar sin_cos[] = {sqrt(squared_sin_cos[0]) * sin_sign,
                            sqrt(squared_sin_cos[1])};
  return Quaternion<Scalar>(sin_cos[1], 0, 0, sin_cos[0]);
}

// Wraps an angle to the interval \f$[-\pi,+\pi\f$.
template <typename Scalar>
Scalar WrapAngle(Scalar angle) {
  // We make the assumption that the angle is actually almost inside the
  // interval. In this case using while should be faster than calling fmod.
  constexpr Scalar Range = Scalar(2.0 * M_PI);
  while (angle > +M_PI) {
    angle -= Range;
  }
  while (angle < -M_PI) {
    angle += Range;
  }
  return angle;
}

// Gets the angle difference between two rotations
template <typename Scalar, int LhsOptions, int RhsOptions>
Scalar DeltaAngle(const SO2<Scalar, LhsOptions>& from,
                  const SO2<Scalar, RhsOptions>& to) {
  return (from.inverse() * to).angle();
}

// Computes the derivative of SO2(theta).matrix() wrt. angle theta.
template <class Scalar>
Matrix2<Scalar> SO2DerivativeTheta(Scalar theta) {
  const SO2<Scalar> rotation(theta);
  Matrix2<Scalar> R;
  R << -rotation.sin_angle(), -rotation.cos_angle(), rotation.cos_angle(),
      -rotation.sin_angle();
  return R;
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SO2_H_
