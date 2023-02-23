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

#include "distribution.h"

#include <limits>
#include <random>

#include "Eigen/Core"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "interpolation.h"
#include "matchers.h"

namespace eigenmath {
namespace {

using ::testing::DoubleNear;
using testing::IsApprox;
using ::testing::Not;

// Uses the non-default constructor of UniformDistributionVector3d.
class UniformDynamicVectorDistribution3d : public UniformDistributionVectorXd {
 public:
  UniformDynamicVectorDistribution3d() : UniformDistributionVectorXd(3) {}
};

// Basic tests for all distributions as type parametrized test.
template <typename T>
class DistributionTest : public ::testing::Test {};

using TestDistributions = ::testing::Types<
    UniformDistributionVector3d, UniformDynamicVectorDistribution3d,
    UniformDistributionSO2d, UniformDistributionPose2d, UniformDistributionSO3d,
    UniformDistributionPose3d, NormalDistributionVector3d,
    UniformDistributionUnitVector2d, UniformDistributionUnitVector3d,
    UniformDistributionUnitVector<double, 4>>;
TYPED_TEST_SUITE(DistributionTest, TestDistributions);

TYPED_TEST(DistributionTest, DeterministicGeneration) {
  constexpr int kSeed = 1346634;
  using ResultType = typename TypeParam::result_type;
  TypeParam dist;

  // Sample reference value from distribution.
  std::default_random_engine rnd(kSeed);
  const ResultType expected = dist(rnd);

  // Compare multiple samples generated with the identical generator.
  for (int i = 0; i < 100; ++i) {
    rnd = std::default_random_engine(kSeed);
    const ResultType sample = dist(rnd);
    EXPECT_THAT(sample, IsApprox(expected));
  }
}

TYPED_TEST(DistributionTest, SampleVariance) {
  constexpr int kSeed = 1346634;
  std::default_random_engine rnd(kSeed);
  TypeParam dist;

  // Generate a few samples.
  constexpr int kSamples = 10;
  std::vector<typename TypeParam::result_type> samples;
  samples.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    samples.push_back(dist(rnd));
  }

  // It is unlikely that two samples are very close.
  for (int i = 0; i < kSamples; ++i) {
    for (int j = i + 1; j < kSamples; ++j) {
      EXPECT_THAT(samples[i], Not(IsApprox(samples[j])));
    }
  }
}

// Norm checking tests for unit vector distributions as type parametrized test.
template <typename T>
class UnitVectorDistributionTest : public ::testing::Test {};

using UnitVectorDistributions =
    ::testing::Types<UniformDistributionUnitVector2d,
                     UniformDistributionUnitVector3d,
                     UniformDistributionUnitVector<double, 4>>;
TYPED_TEST_SUITE(UnitVectorDistributionTest, UnitVectorDistributions);

TYPED_TEST(UnitVectorDistributionTest, HasNormOne) {
  constexpr int kSeed = 1346634;
  std::default_random_engine rnd(kSeed);
  TypeParam dist;
  using Vector = typename TypeParam::result_type;
  using Scalar = typename Vector::Scalar;
  constexpr Scalar kEpsilon = std::numeric_limits<Scalar>::epsilon();

  // Generate a few samples.
  constexpr int kSamples = 100;
  for (int i = 0; i < kSamples; ++i) {
    const Vector sample = dist(rnd);
    EXPECT_THAT(sample.norm(), DoubleNear(1, kEpsilon)) << sample;
  }
}

}  // namespace
}  // namespace eigenmath
