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

#ifndef EIGENMATH_EIGENMATH_MEAN_AND_COVARIANCE_H_
#define EIGENMATH_EIGENMATH_MEAN_AND_COVARIANCE_H_

#include <type_traits>

#include "absl/log/check.h"
#include "manifolds.h"
#include "pose2.h"
#include "pose3.h"
#include "type_checks.h"
#include "types.h"

namespace eigenmath {

// Combines mean and covariance.
//
// `mean` is an element in an arbitrary state space.  `covariance` is defined in
// the tangent space at `mean` (which is Euclidean).
template <typename Mean>
struct MeanAndCovariance {
  // Vector type representing the tangent (linear) space.
  using Scalar = ScalarTypeOf<Mean>;
  using TangentVector =
      decltype(LogRiemann(std::declval<Mean>(), std::declval<Mean>()));
  static constexpr int kEigenDimension = TangentVector::RowsAtCompileTime;
  // Matrix type representing the covariance in the tangent space.
  using Covariance =
      Eigen::Matrix<Scalar, kEigenDimension, kEigenDimension, kDefaultOptions>;

  Mean mean;
  Covariance covariance;

  MeanAndCovariance() = default;

  // Construction for compatible mean and covariance types.
  template <typename OtherMeantT, typename OtherCovaranceT>
  MeanAndCovariance(const OtherMeantT& m, const OtherCovaranceT& c)
      : mean{m}, covariance{c} {}

  // Conversion operator for other MeanAndCovariance types with different
  // Eigen::Options
  template <typename OtherMean>
  MeanAndCovariance(  // NOLINT
      const MeanAndCovariance<OtherMean>& other)
      : mean{other.mean}, covariance{other.covariance} {}

  // Assignment operator for other MeanAndCovariance types with different
  // Eigen::Options
  template <typename OtherMean>
  MeanAndCovariance& operator=(const MeanAndCovariance<OtherMean>& other) {
    mean = other.mean;
    covariance = other.covariance;
    return *this;
  }
};

// Euclidean vector with covariance
//
// The mean is an element in n-dimensional Euclidean space.
// Covariance is a nxn positive definite matrix.
template <typename Scalar, int EigenDimension, int Options = kDefaultOptions>
using EuclideanMeanAndCovariance =
    MeanAndCovariance<Eigen::Matrix<Scalar, EigenDimension, 1, Options>>;

typedef EuclideanMeanAndCovariance<double, 2> PointAndCovariance2d;
typedef EuclideanMeanAndCovariance<float, 2> PointAndCovariance2f;

typedef EuclideanMeanAndCovariance<double, 3> PointAndCovariance3d;
typedef EuclideanMeanAndCovariance<float, 3> PointAndCovariance3f;

typedef EuclideanMeanAndCovariance<double, 3> VelocityAndCovariance3d;
typedef EuclideanMeanAndCovariance<float, 3> VelocityAndCovariance3f;

typedef EuclideanMeanAndCovariance<double, 6> TwistAndCovariance3d;
typedef EuclideanMeanAndCovariance<float, 6> TwistAndCovariance3f;

// SE(n) with covariance matrix
//
// For n=2: This is Pose2 as mean and a three dimensional square matrix as
// covariance. For n=3: This is Pose3 as mean and a six dimensional square
// matrix as covariance.

// 2D transformation with covariance.
template <typename Scalar, int Options = kDefaultOptions>
using PoseAndCovariance2 = MeanAndCovariance<Pose2<Scalar, Options>>;

using PoseAndCovariance2d = PoseAndCovariance2<double>;
using PoseAndCovariance2f = PoseAndCovariance2<float>;

// 3D transformation with covariance.
template <typename Scalar, int Options = kDefaultOptions>
using PoseAndCovariance3 = MeanAndCovariance<Pose3<Scalar, Options>>;

using PoseAndCovariance3d = PoseAndCovariance3<double>;
using PoseAndCovariance3f = PoseAndCovariance3<float>;

// Creates a covariance matrix from a `rotation` and diagonal `scale` matrix.
template <typename Scalar, int EigenDimension,
          int RotationOptions = kDefaultOptions,
          int ScaleOptions = kDefaultOptions>
Eigen::Matrix<Scalar, EigenDimension, EigenDimension, kDefaultOptions>
CreateCovariance(
    const Eigen::Matrix<Scalar, EigenDimension, EigenDimension,
                        RotationOptions>& rotation,
    const Eigen::Matrix<Scalar, EigenDimension, 1, ScaleOptions>& scale) {
  const Eigen::DiagonalMatrix<Scalar, EigenDimension> scale_matrix_square(
      (scale.array() * scale.array()).matrix());
  return rotation * scale_matrix_square * rotation.transpose();
}

// Creates a 3D covariance matrix from a angle/axis representation of a
// rotation, and a diagonal scaling matrix.
template <typename Scalar, int AxisOptions = kDefaultOptions,
          int ScaleOptions = kDefaultOptions>
Eigen::Matrix<Scalar, 3, 3, kDefaultOptions> CreateCovarianceAngleAxis(
    Scalar angle, const Eigen::Matrix<Scalar, 3, 1, AxisOptions>& axis,
    const Eigen::Matrix<Scalar, 3, 1, ScaleOptions>& scale) {
  return CreateCovariance(
      Eigen::AngleAxis<Scalar>(angle, axis.normalized()).matrix(), scale);
}

// Creates a mean and covariance object for Euclidean space.
template <
    typename Scalar, int EigenDimension, int MeanOptions = kDefaultOptions,
    int RotationOptions = kDefaultOptions, int ScaleOptions = kDefaultOptions>
EuclideanMeanAndCovariance<Scalar, EigenDimension>
CreateEuclideanMeanAndCovariance(
    const Eigen::Matrix<Scalar, EigenDimension, 1, MeanOptions>& mean,
    const Eigen::Matrix<Scalar, EigenDimension, EigenDimension,
                        RotationOptions>& rotation,
    const Eigen::Matrix<Scalar, EigenDimension, 1, ScaleOptions>& scale) {
  return EuclideanMeanAndCovariance<Scalar, EigenDimension>{
      mean, CreateCovariance(rotation, scale)};
}

// Computes sample mean and covariance for a set of points in Euclidean space.
// The set of points is obtained from the iterator range [begin,end).
// Returns mean and covariance.
template <typename Iterator>
auto SampleMeanAndCovariance(Iterator begin, Iterator end) {
  using Vector = typename std::iterator_traits<Iterator>::value_type;
  using Scalar = typename Vector::RealScalar;
  constexpr int kEigenDimension = Vector::RowsAtCompileTime;
  using Matrix = Matrix<Scalar, kEigenDimension, kEigenDimension>;
  const int dimension = begin->rows();
  // sums x for mean and x * x^T for covariance
  Vector pre_mean = Vector::Zero(dimension);
  Matrix pre_covariance = Matrix::Zero(dimension, dimension);
  int num_points = 0;
  for (; begin != end; ++begin, ++num_points) {
    const Vector& p = *begin;
    pre_mean += p;
    pre_covariance += p * p.transpose();
  }
  CHECK_GE(num_points, 1) << "You need at least one point to estimate mean.";
  CHECK_GE(num_points, 2)
      << "You need at least two points for sample covariance.";
  const Vector mean = pre_mean / static_cast<Scalar>(num_points);
  // for covariance use 1/(N-1) if Bessel's correction is used and 1/N otherwise
  const Matrix covariance = (pre_covariance - static_cast<Scalar>(num_points) *
                                                  mean * mean.transpose()) /
                            static_cast<Scalar>(num_points - 1);
  return EuclideanMeanAndCovariance<Scalar, kEigenDimension>{mean, covariance};
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_MEAN_AND_COVARIANCE_H_
