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

#ifndef EIGENMATH_EIGENMATH_DISTRIBUTION_H_
#define EIGENMATH_EIGENMATH_DISTRIBUTION_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "pose2.h"
#include "pose3.h"
#include "so2.h"
#include "so3.h"
#include "types.h"

namespace eigenmath {

// Uniform distribution with coordinates in  [0, 1).  Note that this does not
// match the Eigen sampling range, which uses [-1, 1).
//
// When sampling uniformly in a different range, consider applying an rotation
// and translation to the result.   The following code example generates a
// sample in the bounding box defined by min and max.
//
// std::default_random_engine generator(std::random_device{}());
// const Vector3d min{-1, 0, 2};
// const Vector3d max{1, 1, 5};
// UniformDistributionVector3d dist;
// const Vector3d sample = InterpolateLinearInBox(dist(generator), min, max);
//
template <typename RealType, int Dimension>
class UniformDistributionVector {
 public:
  using result_type = Vector<RealType, Dimension>;

  UniformDistributionVector() {
    static_assert(Dimension > 0,
                  "The distribution is designed for fixed size vectors.");
  }

  explicit UniformDistributionVector(int n) : dimension_(n) {
    static_assert(
        Dimension == Eigen::Dynamic,
        "The distribution is designed for dynamically sized vectors.");
  }

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    result_type sample(ActualDimension(DimensionType{}));
    constexpr RealType kZero{0};
    constexpr RealType kOne{1};
    std::uniform_real_distribution<RealType> interval(kZero, kOne);
    std::generate_n(sample.data(), ActualDimension(DimensionType{}),
                    [&]() { return interval(generator); });
    return sample;  // NRVO
  }

 private:
  struct NoneType {};
  using DimensionType = typename std::conditional<Dimension == Eigen::Dynamic,
                                                  int, NoneType>::type;
  constexpr int ActualDimension(NoneType /*unused*/) const noexcept {
    return Dimension;
  }
  int ActualDimension(int /*unused*/) const { return dimension_; }
  DimensionType dimension_;
};

using UniformDistributionVector2d = UniformDistributionVector<double, 2>;
using UniformDistributionVector3d = UniformDistributionVector<double, 3>;
using UniformDistributionVectorXd =
    UniformDistributionVector<double, Eigen::Dynamic>;

// Normal distribution on R^n.
template <typename RealType, int Dimension>
class NormalDistributionVector {
 public:
  static_assert(
      Dimension > 0,
      "This class is not (yet) designed for dynamically sized vectors.");
  using result_type = Vector<RealType, Dimension>;

  // Creates a zero-mean distribution with covariance as the identity matrix.
  NormalDistributionVector() = default;

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    result_type sample;
    std::normal_distribution<RealType> dist(0, 1);
    std::generate_n(sample.data(), Dimension,
                    [&]() { return dist(generator); });
    return sample;  // NRVO
  }
};

using NormalDistributionVector2d = NormalDistributionVector<double, 2>;
using NormalDistributionVector3d = NormalDistributionVector<double, 3>;
using NormalDistributionVectorXd =
    NormalDistributionVector<double, Eigen::Dynamic>;

// Uniform distribution on unit vectors in R^n.
template <typename RealType, int Dimension>
class UniformDistributionUnitVector {
 public:
  static_assert(
      Dimension > 0,
      "This class is not (yet) designed for dynamically sized vectors.");
  using result_type = Vector<RealType, Dimension>;

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    // Use rotationally symmetric distribution around 0 and normalize sample to
    // obtain a uniform distribution.
    result_type sample = normal_dist_(generator);
    while (sample.norm() < std::numeric_limits<RealType>::epsilon()) {
      sample = normal_dist_(generator);
    }
    sample.normalize();
    return sample;
  }

 private:
  NormalDistributionVector<RealType, Dimension> normal_dist_;
};

using UniformDistributionUnitVector2d =
    UniformDistributionUnitVector<double, 2>;
using UniformDistributionUnitVector3d =
    UniformDistributionUnitVector<double, 3>;
using UniformDistributionUnitVectorXd =
    UniformDistributionUnitVector<double, Eigen::Dynamic>;

// Uniform distribution on unit vectors in R^2.  This specialization has better
// performance, and avoids resampling.
template <typename RealType>
class UniformDistributionUnitVector<RealType, 2> {
 public:
  using result_type = Vector<RealType, 2>;

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    constexpr RealType kZero{0.0};
    constexpr RealType kTwoPi{2.0 * M_PI};
    std::uniform_real_distribution<RealType> circle(kZero, kTwoPi);
    const RealType sample = circle(generator);
    using std::cos;
    using std::sin;
    return {sin(sample), cos(sample)};
  }
};

// Uniform distribution on unit vectors in R^4.  This specialization has better
// performance, and avoids resampling.
template <typename RealType>
class UniformDistributionUnitVector<RealType, 4> {
 public:
  using result_type = Vector<RealType, 4>;

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    // Uses the uniform sampling method mentioned by [LaValle, Planning
    // Algorithms, Cambridge University Press, 2006] in section 5.2.2,
    // which is taken from [Shoemake, Uniform random rotations, in Kirk
    // (ed), Graphics Gems III, pg 124–132, Academic, New York, 1992]
    constexpr RealType kZero{0.0};
    constexpr RealType kOne{1.0};
    constexpr RealType kTwoPi{2.0 * M_PI};
    std::uniform_real_distribution<RealType> unit_interval(kZero, kOne);
    std::uniform_real_distribution<RealType> circle(kZero, kTwoPi);
    const RealType samples[] = {unit_interval(generator), circle(generator),
                                circle(generator)};
    using std::cos;
    using std::sin;
    using std::sqrt;
    return {sqrt(kOne - samples[0]) * sin(samples[1]),
            sqrt(kOne - samples[0]) * cos(samples[1]),
            sqrt(samples[0]) * sin(samples[2]),
            sqrt(samples[0]) * cos(samples[2])};
  }
};

using UniformDistributionUnitVector2d =
    UniformDistributionUnitVector<double, 2>;
using UniformDistributionUnitVector3d =
    UniformDistributionUnitVector<double, 3>;
using UniformDistributionUnitVectorXd =
    UniformDistributionUnitVector<double, Eigen::Dynamic>;

// Uniform distribution on SO2.
template <typename RealType = double>
class UniformDistributionSO2 {
 public:
  using result_type = SO2<RealType>;

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    constexpr RealType kZero{0.0};
    constexpr RealType kTwoPi{2.0 * M_PI};
    std::uniform_real_distribution<RealType> circle(kZero, kTwoPi);
    return result_type(circle(generator));
  }
};
using UniformDistributionSO2d = UniformDistributionSO2<double>;

// Uniform distribution on SO3.
template <typename RealType = double>
class UniformDistributionSO3 {
 public:
  using result_type = SO3<RealType>;

  template <typename Generator>
  result_type operator()(Generator &generator) const {
    UniformDistributionUnitVector<RealType, 4> dist;
    return result_type{Quaternion<RealType, kDefaultOptions>(dist(generator))};
  }
};
using UniformDistributionSO3d = UniformDistributionSO3<double>;

// Distribution on SE2 where the rotation is uniform (in SO2), and the
// translation has components uniformly distributed in [-1, 1].
template <typename RealType = double>
class UniformDistributionPose2 {
 public:
  using result_type = Pose2<RealType>;

  template <typename Generator>
  result_type operator()(Generator &generator) {
    UniformDistributionSO2<RealType> dist_SO2;
    constexpr RealType kOne{1.0};
    std::uniform_real_distribution<RealType> interval(-kOne, kOne);
    // Ensure deterministic order of evaluation.
    const RealType translation[] = {interval(generator), interval(generator)};
    return {Vector2<RealType>(Eigen::Map<const Vector2<RealType>>(translation)),
            dist_SO2(generator)};
  }
};
using UniformDistributionPose2d = UniformDistributionPose2<double>;

// Distribution on SE3 where the rotation is uniform (in SO3), and the
// translation has components uniformly distributed in [-1, 1].
template <typename RealType = double>
class UniformDistributionPose3 {
 public:
  using result_type = Pose3<RealType>;

  template <typename Generator>
  result_type operator()(Generator &generator) {
    UniformDistributionSO3<RealType> dist_SO3;
    constexpr RealType kOne{1.0};
    std::uniform_real_distribution<RealType> interval(-kOne, kOne);
    // Ensure deterministic order of evaluation.
    const RealType translation[] = {interval(generator), interval(generator),
                                    interval(generator)};
    return {dist_SO3(generator),
            Eigen::Map<const Vector3<RealType>>(translation)};
  }
};
using UniformDistributionPose3d = UniformDistributionPose3<double>;

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_DISTRIBUTION_H_
