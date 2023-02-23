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

#ifndef EIGENMATH_EIGENMATH_MANIFOLDS_H_
#define EIGENMATH_EIGENMATH_MANIFOLDS_H_

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/log/check.h"
#include "pose2.h"
#include "pose3.h"
#include "so2.h"
#include "types.h"
#include "utils.h"

namespace eigenmath {

namespace manifolds_internal {

// Exponential map of the special unitary group SU(2). This function is
// an implementation detail for ExpSO3.
//
// Let \f$ \exp \f$ be the matrix exponential and \f$ \hat{\cdot} \f$ be the
// function which maps a tangent vector in SU(2) to its corresponding (2x2)
// matrix representation. It holds hat:
//
// \f$ \exp_{SU(2)}(\delta) = exp(\hat{delta}) \f$
//
// The delta parameter is the tangent vector of SU(2).
// Returns a unit quaternion, a member of SU(2).
template <class T, int Options>
Quaternion<T, Options> ExpSU2Impl(const Vector3<T, Options>& delta) {
  // Employ "using" statement here, since we need to find "sin" and friends
  // using ADL if they are not defined in the std namespace for type T.
  using std::cos;
  using std::sin;
  using std::sqrt;

  Quaternion<T, Options> q_delta;
  const T theta_squared = delta.squaredNorm();
  if (theta_squared > Eigen::NumTraits<T>::dummy_precision()) {
    const T theta = sqrt(theta_squared);
    q_delta.w() = cos(theta);
    q_delta.vec() = (sin(theta) / theta) * delta;
  } else {
    // taylor expansions around theta == 0
    q_delta.w() = T(1.) - T(0.5) * theta_squared;
    q_delta.vec() = (T(1.) - T(1. / 6.) * theta_squared) * delta;
  }
  return q_delta;
}

// Logarithmic map of the special unitary group SU(2). This function is
// an implementation detail for LogSO3.
//
// This is the inverse of ExpSU2Impl.
//
// The q parameter is the quaternion as element of SU(2).
// The quaternion q must of unit length.
// Returns corresponding vector in the tangent space of SU(2).
template <class T, int Options>
Vector3<T> LogSU2Impl(const Quaternion<T, Options>& q) {
  // Employ using statement here, since we need to find "abs" and friends using
  // ADL if they are not defined in the std namespace for type T.
  using std::abs;
  using std::acos;
  using std::asin;
  using std::pow;
  using std::sqrt;

  // Implementation of the logarithmic map of SU(2) using atan.
  // This follows Hertzberg et al. "Integrating Generic Sensor Fusion Algorithms
  // with Sound State Representations through Encapsulation of Manifolds", Eq.
  // (31)
  // We use asin and acos instead of atan to enable the use of Eigen Autodiff
  // with SU2.
  const T sign_of_w = q.w() < T(0.0) ? T(-1.0) : T(1.0);
  const T abs_w = sign_of_w * q.w();
  const Vector3<T> v = sign_of_w * q.vec();
  const T squared_norm_of_v = v.squaredNorm();

  CHECK_LT(abs(T(1.) - abs_w * abs_w - squared_norm_of_v),
           Eigen::NumTraits<T>::dummy_precision())
      << "quaternion q must be approx. of unit length";

  if (squared_norm_of_v > Eigen::NumTraits<T>::dummy_precision()) {
    const T norm_of_v = sqrt(squared_norm_of_v);
    if (abs_w > Eigen::NumTraits<T>::dummy_precision()) {
      // asin(x) = acos(x) at x = 1/sqrt(2).
      if (norm_of_v <= T(M_SQRT1_2)) {
        return (asin(norm_of_v) / norm_of_v) * v;
      }
      return (acos(abs_w) / norm_of_v) * v;
    }
    return (M_PI_2 / norm_of_v) * v;
  }

  // Taylor expansion at squared_norm_of_v == 0
  return (T(1.) / abs_w - squared_norm_of_v / (T(3.) * pow(abs_w, 3))) * v;
}

// Matrix exponential using taylor series.
//
// This is a implementation of the matrix exponential using its
// definition. To increase robustness for matrices with higher magnitudes, this
// method should not be called directly but used in terms of
// MatrixExpScalingAndSquaringImpl.
//
// The A parameter is a square matrix.
// The num_iter parameter is the number of iterations to use.
// Returns the matrix exponential of A, which is an invertible matrix.
template <class T, int dim, int Options>
Matrix<T, dim, dim> MatrixExpTaylorImpl(const Matrix<T, dim, dim, Options>& A,
                                        int num_iter) {
  static_assert(dim >= 1, "dim must be >= 1");
  Matrix<T, dim, dim> result;
  Matrix<T, dim, dim> A_to_j;
  A_to_j.setIdentity();
  result = A_to_j;
  T j_fac = T(1.);

  for (int j = 1; j < num_iter; ++j) {
    j_fac *= j;
    A_to_j *= A;
    result += T(1.) / j_fac * A_to_j;
  }
  return result;
}

// Matrix exponential using scaling and squaring.
//
// The A parameter is a square matrix.
// The num_iter parameter is the number of iterations to use.
// Returns the matrix exponential of A, which is an invertible matrix.
template <class T, int dim, int Options>
Matrix<T, dim, dim> MatrixExpScalingAndSquaringImpl(
    const Matrix<T, dim, dim, Options>& A, int num_iter) {
  // Scaling and squaring method
  // Following Moler and Van Loan: "Nineteen Dubious Ways to Compute the
  // Exponential of a Matrix", SIAM Review, 1978, p. 12-15

  using std::log2;
  using std::pow;
  static_assert(dim >= 1, "dim must be >= 1");
  Matrix<T, dim, dim> result;

  // calculate scaling factor and corresponding number of squaring operations
  const T max_abs_val = A.template lpNorm<Eigen::Infinity>();
  if (max_abs_val == 0) {
    result.setIdentity();
    return result;
  }
  const T log2_max_abs_val = std::max(T(0.), log2(max_abs_val));
  const int num_squaring = std::ceil(log2_max_abs_val);
  const T m = pow(T(2.), num_squaring);

  // scale matrix by factor m
  Matrix<T, dim, dim> scaled_A = (T(1.) / m) * A;

  // calculate matrix exponential for scaled matrix
  result = MatrixExpTaylorImpl(scaled_A, num_iter);

  // perform squaring of the exponential
  for (int i = 0; i < num_squaring; ++i) {
    result *= result;
  }
  return result;
}

}  // namespace manifolds_internal

// This is just an overload for ExpRiemann on euclidean vectors.
template <typename T, int N, int Options>
Vector<T, N, Options> ExpRiemann(const Vector<T, N, Options>& base,
                                 const Vector<T, N, Options>& delta) {
  return base + delta;
}

// This is just an overload for LogRiemann on euclidean vectors.
template <typename T, int N, int Options>
Vector<T, N, Options> LogRiemann(
    const Vector<T, N, Options>& ref_translation_t,
    const Vector<T, N, Options>& ref_translation_a) {
  return ref_translation_a - ref_translation_t;
}

// Exponential map of the special orthogonal group SO(3).
//
// It maps a tangent element of SO(3) at the identity, which is equivalent to
// the rotation vector (axis times angle), to the corresponing rotation in
// SO(3). Here we choose to represent the rotation as a unit quaternion instead
// of a rotation matrix.
//
// Let \f$ \exp \f$ be the matrix exponential, \f$ R \f$ be a function which
// maps a quaternion to the corresponing rotation matrix and \f$ \hat{\cdot} \f$
// be the function which maps a 3-vector the corresponding skew-symmetric (3x3)
// matrix representation. It holds hat:
//
// \f$ R(\exp_{SO3}(\delta)) = exp(\hat{delta}) \f$
//
// The delta parameter is the rotation vector (axis times angle).
// Returns the corresponding unit quaternion representing rotation in 3D.
template <class T, int Options>
SO3<T> ExpSO3(const Vector3<T, Options>& delta) {
  //  SU(2) is a double cover of SO(3), thus we have to half the tangent vector
  //  delta
  const Vector3<T, Options> half_delta = T(0.5) * delta;
  return SO3<T>(manifolds_internal::ExpSU2Impl(half_delta), false);
}

// These are just overloads for ExpRiemann.
template <class T, int Options>
SO3<T> ExpRiemann(const Vector3<T, Options>& delta) {
  return ExpSO3(delta);
}

template <class T, int Options>
SO3<T> ExpRiemann(const SO3<T>& base, const Vector3<T, Options>& delta) {
  return base * ExpSO3(delta);
}

// Logarithmic map of the special orthogonal group SO(3).
//
// This is the inverse of exp.
//
// The so3 parameter is a 3d rotation (represented as unit quaternion).
// Returns the corresponding rotation vector (angle times axis).
template <class T, int Options>
Vector3<T> LogSO3(const SO3<T, Options>& so3) {
  // SU(2) is a double cover of SO(3), thus we have to multiply the tangent
  // vector delta by two
  return T(2.) * manifolds_internal::LogSU2Impl(so3.quaternion());
}

// These are just overloads for LogRiemann.
template <class T, int Options>
Vector3<T> LogRiemann(const SO3<T, Options>& so3) {
  return LogSO3(so3);
}

template <class T, int Options>
Vector3<T> LogRiemann(const SO3<T, Options>& ref_so3_t,
                      const SO3<T, Options>& ref_so3_a) {
  return LogRiemann(ref_so3_t.inverse() * ref_so3_a);
}

// Jacobian of logarithmic map for SO(3).
//
// Computes the Jacobian for the logarithmic map of the special orthogonal group
// SO(3). This is the derivative with respect to all four coefficients of the
// given quaternion.
//
// The q parameters is the location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar, int Options>
Matrix<Scalar, 3, 4> LogSO3DerivativeManifold(
    const Quaternion<Scalar, Options>& q) {
  using std::abs;
  using std::atan2;
  using std::sqrt;
  const Scalar w = abs(q.w());
  const Scalar w_sign = (q.w() > 0 ? Scalar(1) : Scalar(-1));
  const Scalar x = w_sign * q.x();
  const Scalar y = w_sign * q.y();
  const Scalar z = w_sign * q.z();
  const Scalar vec_len_2 = q.vec().squaredNorm();
  const Scalar vec_len = sqrt(vec_len_2);
  constexpr Scalar two = Scalar(2);
  if (vec_len < Eigen::NumTraits<Scalar>::dummy_precision()) {
    // This is the Taylor expansion using terms where the combined order of x,
    // y, z is smaller than four.
    constexpr Scalar b = Scalar(1) / Scalar(3);
    const Scalar xx = x * x;
    const Scalar yy = y * y;
    const Scalar zz = z * z;
    constexpr Scalar c = -Scalar(4) / Scalar(3);
    const Scalar cxy = c * x * y;
    const Scalar cxz = c * x * z;
    const Scalar cyz = c * y * z;
    return MakeMatrix<Scalar, 3, 4>(
        {{-two * x, two - xx + b * (yy + zz), cxy, cxz},
         {-two * y, cxy, two - yy + b * (xx + zz), cyz},
         {-two * z, cxz, cyz, two - zz + b * (xx + yy)}});
  } else {
    const Scalar b = two * atan2(vec_len, w) / vec_len;
    const Scalar c = (two * w - b) / vec_len_2;
    const Scalar cx = c * x;
    const Scalar cy = c * y;
    const Scalar cz = c * z;
    const Scalar cxx = cx * x;
    const Scalar cyy = cy * y;
    const Scalar czz = cz * z;
    const Scalar cxy = cx * y;
    const Scalar cxz = cx * z;
    const Scalar cyz = cy * z;
    return w_sign * MakeMatrix<Scalar, 3, 4>({
                        {-two * x, b + cxx, cxy, cxz},
                        {-two * y, cxy, b + cyy, cyz},
                        {-two * z, cxz, cyz, b + czz},
                    });
  }
}

// Left Jacobian of SO(3) w.r.t. exponential coordinates.
//
// Computes the Jacobian for the special orthogonal group with respect to all
// three coefficients of the given LogSO3 (axis*angle) vector, z.
//
// This can be interpreted as satisfying J_left * inv(J_right) == R(z).
//
// Algorithms from: Chirikjian, G. Stochastic Models, Information Theory, and
// Lie Groups, Volume 2. 2010. p40.
//
// The z parameter is the location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar>
Matrix3<Scalar> LeftExponentialJacobian(const Vector3<Scalar>& z) {
  using std::abs;
  using std::cos;
  using std::sin;
  Matrix3<Scalar> jacobian = Matrix3<Scalar>::Identity();
  const Scalar alpha = z.norm();
  if (abs(alpha) > Eigen::NumTraits<Scalar>::dummy_precision()) {
    const Scalar alpha2 = alpha * alpha;
    const Scalar alpha3 = alpha2 * alpha;
    const Scalar coeff0 = (Scalar(1) - cos(alpha)) / alpha2;
    const Scalar coeff1 = (alpha - sin(alpha)) / alpha3;
    const Scalar xx = z.x() * z.x();
    const Scalar xy = z.x() * z.y();
    const Scalar xz = z.x() * z.z();
    const Scalar yy = z.y() * z.y();
    const Scalar yz = z.y() * z.z();
    const Scalar zz = z.z() * z.z();
    // jacobian = I + coeff0 * z_skew + coeff1 * z_skew * z_skew
    jacobian(0, 0) = Scalar(1) - coeff1 * (yy + zz);
    jacobian(0, 1) = -coeff0 * z.z() + coeff1 * xy;
    jacobian(0, 2) = coeff0 * z.y() + coeff1 * xz;

    jacobian(1, 0) = coeff0 * z.z() + coeff1 * xy;
    jacobian(1, 1) = Scalar(1) - coeff1 * (xx + zz);
    jacobian(1, 2) = -coeff0 * z.x() + coeff1 * yz;

    jacobian(2, 0) = -coeff0 * z.y() + coeff1 * xz;
    jacobian(2, 1) = coeff0 * z.x() + coeff1 * yz;
    jacobian(2, 2) = Scalar(1) - coeff1 * (xx + yy);
  }
  return jacobian;
}

// Inverse left Jacobian of SO(3) w.r.t. exponential coordinates.
//
// Computes the inverse of the Jacobian for the special orthogonal group with
//  respect to all three coefficients of the given LogSO3 (axis*angle)
//  vector, z.
//
// This can be interpreted as satisfying J_left * inv(J_right) == R(z).
//
// Algorithms from: Chirikjian, G. Stochastic Models, Information Theory, and
// Lie Groups, Volume 2. 2010. p40.
//
// The z parameter is the location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar>
Matrix3<Scalar> InverseLeftExponentialJacobian(const Vector3<Scalar>& z) {
  using std::abs;
  using std::cos;
  using std::sin;
  Matrix3<Scalar> jacobian = Matrix3<Scalar>::Identity();
  const Scalar alpha = z.norm();
  if (abs(alpha) > Eigen::NumTraits<Scalar>::dummy_precision()) {
    const Scalar alpha2 = alpha * alpha;
    const Scalar coeff0 = -Scalar(0.5);
    const Scalar coeff1 =
        Scalar(1) / alpha2 -
        (Scalar(1) + cos(alpha)) / (Scalar(2) * alpha * sin(alpha));
    const Scalar xx = z.x() * z.x();
    const Scalar xy = z.x() * z.y();
    const Scalar xz = z.x() * z.z();
    const Scalar yy = z.y() * z.y();
    const Scalar yz = z.y() * z.z();
    const Scalar zz = z.z() * z.z();
    // jacobian = I + coeff0 * z_skew + coeff1 * z_skew * z_skew
    jacobian(0, 0) = Scalar(1) - coeff1 * (yy + zz);
    jacobian(0, 1) = -coeff0 * z.z() + coeff1 * xy;
    jacobian(0, 2) = coeff0 * z.y() + coeff1 * xz;

    jacobian(1, 0) = coeff0 * z.z() + coeff1 * xy;
    jacobian(1, 1) = Scalar(1) - coeff1 * (xx + zz);
    jacobian(1, 2) = -coeff0 * z.x() + coeff1 * yz;

    jacobian(2, 0) = -coeff0 * z.y() + coeff1 * xz;
    jacobian(2, 1) = coeff0 * z.x() + coeff1 * yz;
    jacobian(2, 2) = Scalar(1) - coeff1 * (xx + yy);
  }
  return jacobian;
}

// Right Jacobian of SO(3) w.r.t. exponential coordinates.
//
// Computes the Jacobian for the special orthogonal group with respect to all
// three coefficients of the given LogSO3 (axis*angle) vector, z.
//
// This can be interpreted as satisfying J_left * inv(J_right) == R(z).
//
// Algorithms from: Chirikjian, G. Stochastic Models, Information Theory, and
// Lie Groups, Volume 2. 2010. p40.
//
// The z parameter is the location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar>
Matrix3<Scalar> RightExponentialJacobian(const Vector3<Scalar>& z) {
  return LeftExponentialJacobian(z).transpose();
}

// Inverse right Jacobian of SO(3) w.r.t. exponential coordinates.
//
// Computes the inverse of the Jacobian for the special orthogonal group with
// respect to all three coefficients of the given logSO3 (axis*angle)
// vector, z.
//
// This can be interpreted as satisfying J_left * inv(J_right) == R(z).
//
// Algorithms from: Chirikjian, G. Stochastic Models, Information Theory, and
// Lie Groups, Volume 2. 2010. p40.
//
// The z parameter is the location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar>
Matrix3<Scalar> InverseRightExponentialJacobian(const Vector3<Scalar>& z) {
  return InverseLeftExponentialJacobian(z).transpose();
}

// Exponential map of the special orthogonal group SO(2).
//
// For the two dimensional space the tangent space simply consists of the
// rotation angle.
//
// The delta parameter is the rotation angle.
// Returns an SO(2) object representing the rotation in 2D.
template <typename Scalar>
SO2<Scalar> ExpSO2(Scalar delta) {
  return SO2<Scalar>{delta};
}

// These are just overloads for ExpRiemann.
template <class T, int Options>
SO2<T> ExpRiemann(const Vector<T, 1, Options>& delta) {
  return ExpSO2(delta.x());
}

template <class T, int Options>
SO2<T> ExpRiemann(const SO2<T>& base, const Vector<T, 1, Options>& delta) {
  return base * ExpSO2(delta.x());
}

// Logarithmic map of the special orthogonal group SO(2).
//
// The q parameter is the element in SO(2).
// Returns the rotation angle.
template <typename Scalar, int Options>
Scalar LogSO2(const SO2<Scalar, Options>& q) {
  return q.angle();
}

// These are just overloads for LogRiemann.
template <class T, int Options>
Vector<T, 1> LogRiemann(const SO2<T, Options>& so2) {
  return Vector<T, 1>{LogSO2(so2)};
}

template <class T, int Options>
Vector<T, 1> LogRiemann(const SO2<T, Options>& ref_so2_t,
                        const SO2<T, Options>& ref_so2_a) {
  return LogRiemann(ref_so2_t.inverse() * ref_so2_a);
}

// Jacobian of logarithmic map for SO(2).
//
// Computes the Jacobian for the logarithmic map of the special orthogonal group
// SO(2). The Jacobian is the matrix of all first-order derivatives.
//
// The q parameter is the location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar, int Options>
Matrix<Scalar, 1, 2> LogSO2Derivative(const SO2<Scalar, Options>& q) {
  return MakeMatrix<Scalar, 1, 2>({{-q.sin_angle(), q.cos_angle()}});
}

// Exponential map for the special Euclidean group SE(2)
//
// Convention for tangent space is: (delta_x, delta_y, delta_rotation).
//
// The tangent parameter is a direction in tangent space se(2).
// Returns a 2D transformation in SE(2).
template <typename Derived>
Pose2<typename Derived::RealScalar> ExpSE2(
    const Eigen::MatrixBase<Derived>& tangent) {
  static_assert(Derived::RowsAtCompileTime == 3,
                "ExpSE2 expects a Vector3<Scalar>");
  static_assert(Derived::ColsAtCompileTime == 1,
                "ExpSE2 expects a Vector3<Scalar>");
  auto&& delta = tangent.eval();
  using Scalar = typename Derived::RealScalar;
  using std::abs;
  using std::cos;
  using std::sin;
  const Scalar dx = delta[0];
  const Scalar dy = delta[1];
  const Scalar theta = delta[2];
  const Scalar cos_theta = cos(theta);
  const Scalar sin_theta = sin(theta);
  // Evaluate the following functions for x = theta:
  //    sinc(x) = sin(x) / x
  //    cinc(x) = (1 - cos(x)) / x
  // The Taylor expansions around x=0 are:
  //    sinc(x) = 1 - 1/6*x*x
  //    cinc(x) = 1/2*x
  Scalar sinc, cinc;
  if (abs(theta) < Eigen::NumTraits<Scalar>::dummy_precision()) {
    sinc = Scalar(1) - (Scalar(1) / Scalar(6)) * theta * theta;
    cinc = (Scalar(1) / Scalar(2)) * theta;
  } else {
    sinc = sin_theta / theta;
    cinc = (Scalar(1) - cos_theta) / theta;
  }
  return Pose2<Scalar>{
      Vector2<Scalar>{dx * sinc - dy * cinc, dx * cinc + dy * sinc},
      SO2<Scalar>{cos_theta, sin_theta, false}};
}

// Logarithmic map of the special Euclidean group SE(2).
//
// Convention for tangent space is: (delta_x, delta_y, delta_rotation).
//
// The pose parameter is a 2D transformation in SE(2).
// Returns a direction in tangent space of se(2).
template <typename Scalar, int Options>
Vector3<Scalar> LogSE2(const Pose2<Scalar, Options>& pose) {
  using std::abs;
  using std::cos;
  using std::sin;
  const Scalar x = pose.translation().x();
  const Scalar y = pose.translation().y();
  const Scalar theta = pose.angle();
  const Scalar theta_half = theta / Scalar(2);
  // Evaluate the following function for x = theta:
  //    coto(x) = x/2 * sin(x) / (1 - cos(x))
  // The Taylor expansion around x=0 is:
  //    coto_0(x) = 1 - 1/12*x*x
  Scalar coto;
  if (pose.so2().cos_angle() >
      Scalar(1) - Eigen::NumTraits<Scalar>::dummy_precision()) {
    coto = Scalar(1) - (Scalar(1) / Scalar(12)) * theta * theta;
  } else {
    const Scalar cos_theta = pose.so2().cos_angle();
    const Scalar sin_theta = pose.so2().sin_angle();
    coto = theta_half * sin_theta / (Scalar(1) - cos_theta);
  }
  const Scalar dx = coto * x + theta_half * y;
  const Scalar dy = -theta_half * x + coto * y;
  return Vector3<Scalar>{dx, dy, theta};
}

// Derivative of group logarithm on SE(2) in tangent space.
//
// Let \f$ f: \mathbb{R}^3 \rightarrow \mathbb{R}^3, t \mapsto log(a exp(t)) \f$
// for a fixed \f$ a \in SE(2) \f$. Here $log$ and $exp$ are the group
// logarithm/exponential.
//
// This function computes the Jacobian \f$ J_f(0) \f$.
//
// The pose parameter is the 2D transform for which to compute the derivative.
// Returns the Jacobian of the function.
template <typename Scalar, int Options>
Matrix3<Scalar> LogSE2DerivativeTangent0(const Pose2<Scalar, Options>& pose) {
  const Scalar half = Scalar(1) / Scalar(2);
  const Scalar x_half = half * pose.translation().x();
  const Scalar y_half = half * pose.translation().y();
  const Scalar theta = pose.angle();
  const Scalar theta_half = half * theta;
  // Evaluate the following functions for x = theta:
  //    coto(x) = x/2 * sin(x) / (1 - cos(x))
  //    osca(x) = (sin(x) - x) / (1 - cos(x))
  // The Taylor expansions around x=0 are:
  //    coto_0(x) = 1 - 1/12*x*x
  //    osca_0(x) = -1/3*x
  Scalar coto, osca;
  if (pose.so2().cos_angle() >
      Scalar(1) - Eigen::NumTraits<Scalar>::dummy_precision()) {
    coto = Scalar(1) - (Scalar(1) / Scalar(12)) * theta * theta;
    osca = -(Scalar(1) / Scalar(3)) * theta;
  } else {
    const Scalar cos_theta = pose.so2().cos_angle();
    const Scalar sin_theta = pose.so2().sin_angle();
    coto = theta_half * sin_theta / (Scalar(1) - cos_theta);
    osca = (sin_theta - theta) / (Scalar(1) - cos_theta);
  }
  return MakeMatrix<Scalar, 3, 3>({{coto, -theta_half, x_half * osca + y_half},
                                   {theta_half, coto, -x_half + y_half * osca},
                                   {Scalar(0), Scalar(0), Scalar(1)}});
}

// Riemannian manifold exponential map on SE(2) at the identity.
//
// Convention for tangent space is: (delta_x, delta_y, delta_rotation).
//
// The tangent parameter is the direction in tangent space.
// Return a 2D transformation.
template <typename Scalar>
Pose2<Scalar> ExpRiemann(const Vector3<Scalar>& tangent) {
  return Pose2<Scalar>{Vector2<Scalar>(tangent.template topLeftCorner<2, 1>()),
                       SO2<Scalar>(ExpSO2(tangent[2]))};
}

// Riemannian manifold log on SE(2) at the identity.
//
// Convention for tangent space is: (delta_x, delta_y, delta_rotation).
//
// The pose parameter is a 2D transformation.
// Returns the direction in tangent space.
template <typename Scalar, int Options>
Vector3<Scalar> LogRiemann(const Pose2<Scalar, Options>& pose) {
  return Vector3<Scalar>{pose.translation().x(), pose.translation().y(),
                         LogSO2(pose.so2())};
}

// Jacobian of logarithmic map for Riemannian manifold on SE(2).
//
// Computes the Jacobian for the logarithmic map of the special Euclidean group
// SE(2). The Jacobian is the matrix of all first-order derivatives.
//
// The pose parameter is a location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar, int Options>
Matrix<Scalar, 3, 4> LogRiemannDerivative(const Pose2<Scalar, Options>& pose) {
  Matrix<Scalar, 3, 4> jacobian = Matrix<Scalar, 3, 4>::Identity();
  jacobian.template bottomRightCorner<1, 2>() = LogSO2Derivative(pose.so2());
  return jacobian;
}

// Riemannian manifold exponential map on SE(2) around a frame.
//
// Convention for tangent space is: (delta_x, delta_y, delta_rotation).
//
// The ref_pose_t parameter is the pose of tangent space in reference frame.
// The tangent parameter is a tangent pointing to the new frame.
// Return the pose of new frame in reference frame.
template <typename Scalar, int Options>
Pose2<Scalar> ExpRiemann(const Pose2<Scalar, Options>& ref_pose_t,
                         const Vector3<Scalar>& tangent) {
  return ref_pose_t * ExpRiemann(tangent);
}

// Riemannian manifold log on SE(2) around a frame.
//
// Convention for tangent space is: (delta_x, delta_y, delta_rotation).
//
// The ref_pose_t parameter is the pose of tangent space in reference frame.
// The ref_pose_a parameter is a transformation from reference frame to frame A.
// Returns a tangent pointing from tangent space origin to corresponding
// reference frame A.
template <typename Scalar, int Options_t, int Options_a>
Vector3<Scalar> LogRiemann(const Pose2<Scalar, Options_t>& ref_pose_t,
                           const Pose2<Scalar, Options_a>& ref_pose_a) {
  return LogRiemann(ref_pose_t.inverse() * ref_pose_a);
}

// Jacobian of logarithmic map for Riemannian manifold on SE(3).
//
// Computes the Jacobian for the logarithmic map of the special Euclidean group
// SE(3). The Jacobian is the matrix of all first-order derivatives.
//
// The pose parameter is a location on manifold where to compute the jacobian.
// Returns a matrix with all first-order derivatives.
template <typename Scalar, int Options>
Matrix<Scalar, 6, 7> LogRiemannDerivative(const Pose3<Scalar, Options>& pose) {
  Matrix<Scalar, 6, 7> jacobian = Matrix<Scalar, 6, 7>::Identity();
  jacobian.template bottomRightCorner<3, 4>() =
      LogSO3DerivativeManifold(pose.quaternion());
  return jacobian;
}

// Riemannian manifold exponential map on SE(3) at the identity.
//
// Convention for tangent space is: (delta_x, delta_y, delta_z, delta_r1,
// delta_r2, delta_r3).
//
// The tangent parameter is a direction in tangent space.
// Returns a 3D transformation.
template <typename Scalar, int Options>
Pose3<Scalar> ExpRiemann(const Vector6<Scalar, Options>& tangent) {
  return Pose3<Scalar>{
      ExpSO3(Vector3<Scalar>(tangent.template bottomLeftCorner<3, 1>())),
      tangent.template topLeftCorner<3, 1>()};
}

// Riemannian manifold log on SE(3) at the identity.
//
// Convention for tangent space is: (delta_x, delta_y, delta_z, delta_r1,
// delta_r2, delta_r3).
//
// The pose parameter is a 3D transformation.
// Returns a direction in tangent space.
template <typename Scalar, int Options>
Vector6<Scalar> LogRiemann(const Pose3<Scalar, Options>& pose) {
  Vector6<Scalar> delta;
  delta.template topLeftCorner<3, 1>() = pose.translation();
  delta.template bottomLeftCorner<3, 1>() = LogSO3(pose.so3());
  return delta;
}

// Riemannian manifold exponential map on SE(3) around a frame.
//
// Convention for tangent space is: (delta_x, delta_y, delta_z, delta_r1,
// delta_r2, delta_r3).
//
// The ref_pose_t parameter is the pose of tangent space in reference frame.
// The tangent parameter is a tangent pointing to the new frame.
// Returns a pose of new frame in reference frame.
template <typename Scalar, int PoseOptions, int TangentOptions>
Pose3<Scalar> ExpRiemann(const Pose3<Scalar, PoseOptions>& ref_pose_t,
                         const Vector6<Scalar, TangentOptions>& tangent) {
  return ref_pose_t * ExpRiemann(tangent);
}

// Riemannian manifold log on SE(3) around a frame.
//
// Convention for tangent space is: (delta_x, delta_y, delta_z, delta_r1,
// delta_r2, delta_r3).
//
// The ref_pose_t parameter is the pose of tangent space in reference frame.
// The ref_pose_a parameter is a transformation from reference frame to frame A.
// Returns a tangent pointing from tangent space origin to corresponding
// reference frame A
template <typename Scalar, int Options_t, int Options_a>
Vector6<Scalar> LogRiemann(const Pose3<Scalar, Options_t>& ref_pose_t,
                           const Pose3<Scalar, Options_a>& ref_pose_a) {
  return LogRiemann(ref_pose_t.inverse() * ref_pose_a);
}

// Computes matrix exponential.
//
// Computes the matrix exponential \f$ exp(X) := \sum_{j=0}^\infty
// \frac{1}{j!}X^j \f$ numerically. In particular, this function is used to test
// the correctnes of symbolic exponential maps such as ExpSO3.
//
// The X parameter is a square matrix.
// The num_iter parameter is the number of iterations to use.
// Returns the matrix exponential of X, which is an invertible matrix.
template <class T, int dim, int Options>
Matrix<T, dim, dim> MatrixExp(const Matrix<T, dim, dim, Options>& X,
                              int num_iter) {
  return manifolds_internal::MatrixExpScalingAndSquaringImpl(X, num_iter);
}

// Computes matrix exponential.
//
// Overload of MatrixExp with num_iter set to 100.
//
// The X parameter is a square matrix.
// Returns the matrix exponential of X, which is an invertible matrix.
template <class T, int dim, int Options>
Matrix<T, dim, dim> MatrixExp(const Matrix<T, dim, dim, Options>& X) {
  return MatrixExp(X, 100);
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_MANIFOLDS_H_
