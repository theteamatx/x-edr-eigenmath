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

#ifndef EIGENMATH_EIGENMATH_NORMAL_DISTRIBUTION_H_
#define EIGENMATH_EIGENMATH_NORMAL_DISTRIBUTION_H_

#include <algorithm>
#include <cmath>

#include "absl/log/check.h"
#include "covariance.h"
#include "manifolds.h"
#include "mean_and_covariance.h"
#include "type_checks.h"
#include "types.h"

namespace eigenmath {

// Represents a multivariate normal distribution
//
// The multi-variate normal distribution is a generalization of the
// one-dimensional normal distribution for arbitrary dimension.
//
// `MeanType` is an element in an arbitrary state space. The covariance is
// defined in the tangent space at the mean. The tangent space vectors are
// obtained through the LogRiemann function in manifolds.h.
template <typename MeanType>
class NormalDistribution {
 public:
  // Vector type representing the tangent (linear) space.
  using Scalar = ScalarTypeOf<MeanType>;
  using TangentVector =
      decltype(LogRiemann(std::declval<MeanType>(), std::declval<MeanType>()));
  static constexpr int kEigenDimension = TangentVector::RowsAtCompileTime;
  // Matrix type representing the covariance in the tangent space.
  using Matrix =
      Eigen::Matrix<Scalar, kEigenDimension, kEigenDimension, kDefaultOptions>;

  // Initializes to standard normal distribution
  explicit NormalDistribution(int dim = kEigenDimension)
      : mean_(ZeroMean(std::max(0, dim))),
        covariance_(Matrix::Identity(std::max(0, dim), std::max(0, dim))),
        information_sqrt_(covariance_),
        mean_probability_(NormalizationFactor(std::max(0, dim))) {}

  // Creates a normal distribution from mean and covariance.
  NormalDistribution(const MeanType& mean, const Matrix& covariance)
      : mean_(mean), covariance_(covariance) {
    SetCovariance(covariance_);
  }

  explicit NormalDistribution(
      const MeanAndCovariance<MeanType>& mean_and_covariance)
      : NormalDistribution(mean_and_covariance.mean,
                           mean_and_covariance.covariance) {}

  const int Dimension() const {
    return (kEigenDimension == Eigen::Dynamic) ? covariance_.rows()
                                               : kEigenDimension;
  }

  const MeanType& Mean() const { return mean_; }

  const Matrix& Covariance() const { return covariance_; }

  void SetMean(const MeanType& mean) { mean_ = mean; }

  void SetCovariance(const Matrix& covariance) {
    covariance_ = covariance;
    const int dim = Dimension();
    CHECK_EQ(covariance_.rows(), dim);
    CHECK_EQ(covariance_.cols(), dim);
    // Inverse square root of covariance (use LLT).
    auto information_sqrt = GetSqrtInformationLLT(covariance_);
    CHECK_OK(information_sqrt);
    information_sqrt_ = *information_sqrt;
    // Compute the mean probability.
    Scalar covariance_det_sqrt = Scalar(1);
    for (int i = 0; i < dim; i++) {
      covariance_det_sqrt /= information_sqrt_(i, i);
    }
    mean_probability_ = NormalizationFactor(dim) / covariance_det_sqrt;
  }

  // Computes the squared Mahalanobis distance for a point
  Scalar MahalanobisDistanceSquared(const MeanType& v) const {
    return (information_sqrt_ * LogRiemann(mean_, v)).squaredNorm();
  }

  // Computes the Mahalanobis distance for a point
  Scalar MahalanobisDistance(const MeanType& v) const {
    return std::sqrt(MahalanobisDistanceSquared(v));
  }

  // Computes probability for a given point
  Scalar Probability(const MeanType& v) const {
    return mean_probability_ *
           std::exp(-Scalar(0.5) * MahalanobisDistanceSquared(v));
  }

  static MeanType ZeroMean(int dim = kEigenDimension) {
    if constexpr (std::is_same_v<MeanType, TangentVector>) {
      return TangentVector::Zero(dim);
    } else {
      return MeanType{};
    }
  }

  // Returns a standard normal distribution with 0 mean and unit covariance
  static NormalDistribution CreateStandard(int dim = kEigenDimension) {
    CHECK_GT(dim, 0);
    return NormalDistribution{dim};
  }

 private:
  // Returns the constant (dimension dependent) distribution normalization
  // factor
  static constexpr Scalar NormalizationFactor(int dim) {
    return std::pow(Scalar(2.0 * M_PI),
                    -static_cast<Scalar>(dim) / Scalar(2.0));
  }

  MeanType mean_;
  Matrix covariance_;
  Matrix information_sqrt_;
  Scalar mean_probability_;
};

template <typename Scalar, int EigenDimension>
using EuclideanNormalDistribution = NormalDistribution<
    Eigen::Matrix<Scalar, EigenDimension, 1, kDefaultOptions>>;

template <typename Scalar>
using EuclideanNormalDistributionX = NormalDistribution<
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, kDefaultOptions>>;

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_NORMAL_DISTRIBUTION_H_
