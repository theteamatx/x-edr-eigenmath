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

#include "vector_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include "distribution.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "sampling.h"
#include "types.h"

namespace eigenmath {
namespace {

constexpr double kEpsilon = 1e-6;

using ::testing::DoubleNear;
using testing::IsApprox;
using ::testing::StrEq;

TEST(EigenMathVectorUtils, CrossProduct) {
  Vector<double, 2> v1{1.0, 2.0};
  Vector<double, 2> v2{3.0, -2.0};
  EXPECT_NEAR(CrossProduct(v1, v2), -8.0, kEpsilon);
  EXPECT_NEAR(CrossProduct(v1, v2), -CrossProduct(v2, v1), kEpsilon);
  Vector<double, 2> v3{-3.0, -2.0};
  EXPECT_NEAR(CrossProduct(v1, v3), 4.0, kEpsilon);
  EXPECT_NEAR(CrossProduct(v1, v3), -CrossProduct(v3, v1), kEpsilon);
}

TEST(EigenMathVectorUtils, DotProduct) {
  Vector<double, 2> v1{1.0, 2.0};
  Vector<double, 2> v2{3.0, -2.0};
  EXPECT_NEAR(DotProduct(v1, v2), -1.0, kEpsilon);
  EXPECT_NEAR(DotProduct(v1, v2), DotProduct(v2, v1), kEpsilon);
  Vector<double, 2> v3{-3.0, -2.0};
  EXPECT_NEAR(DotProduct(v1, v3), -7.0, kEpsilon);
  EXPECT_NEAR(DotProduct(v1, v3), DotProduct(v3, v1), kEpsilon);
}

TEST(EigenMathVectorUtils, RightOrthogonal) {
  Vector<double, 2> v1{1.0, 2.0};
  EXPECT_THAT(RightOrthogonal(v1),
              IsApprox(Vector<double, 2>(-2.0, 1.0), kEpsilon));
  EXPECT_THAT(
      RightOrthogonal(RightOrthogonal(RightOrthogonal(RightOrthogonal(v1)))),
      IsApprox(v1, kEpsilon));
  EXPECT_NEAR(DotProduct(RightOrthogonal(v1), v1), 0.0, kEpsilon);
  Vector<double, 2> v2{-3.0, -2.0};
  EXPECT_THAT(RightOrthogonal(v2),
              IsApprox(Vector<double, 2>(2.0, -3.0), kEpsilon));
  EXPECT_THAT(
      RightOrthogonal(RightOrthogonal(RightOrthogonal(RightOrthogonal(v2)))),
      IsApprox(v2, kEpsilon));
  EXPECT_NEAR(DotProduct(RightOrthogonal(v2), v2), 0.0, kEpsilon);
}

TEST(EigenMathVectorUtils, LeftOrthogonal) {
  Vector<double, 2> v1{1.0, 2.0};
  EXPECT_THAT(LeftOrthogonal(v1),
              IsApprox(Vector<double, 2>(2.0, -1.0), kEpsilon));
  EXPECT_THAT(
      LeftOrthogonal(LeftOrthogonal(LeftOrthogonal(LeftOrthogonal(v1)))),
      IsApprox(v1, kEpsilon));
  EXPECT_NEAR(DotProduct(LeftOrthogonal(v1), v1), 0.0, kEpsilon);
  Vector<double, 2> v2{-3.0, -2.0};
  EXPECT_THAT(LeftOrthogonal(v2),
              IsApprox(Vector<double, 2>(-2.0, 3.0), kEpsilon));
  EXPECT_THAT(
      LeftOrthogonal(LeftOrthogonal(LeftOrthogonal(LeftOrthogonal(v2)))),
      IsApprox(v2, kEpsilon));
  EXPECT_NEAR(DotProduct(LeftOrthogonal(v2), v2), 0.0, kEpsilon);
}

TEST(EigenMathVectorUtils, ExtendToOrthonormalBasis) {
  const std::vector<Vector3d> test_inputs = {
      Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ(),
      Vector3d(1, 2, 3), Vector3d(-1, 1, 0)};
  for (const Vector3d& x : test_inputs) {
    const auto [y, z] = ExtendToOrthonormalBasis(x);
    constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

    EXPECT_THAT(y.norm(), DoubleNear(1, kEpsilon));
    EXPECT_THAT(z.norm(), DoubleNear(1, kEpsilon));

    EXPECT_THAT(x.dot(y), DoubleNear(0, kEpsilon));
    EXPECT_THAT(x.dot(z), DoubleNear(0, kEpsilon));
    EXPECT_THAT(y.dot(z), DoubleNear(0, kEpsilon));
  }
}

TEST(EigenMathVectorUtils, ExtendToOrthonormalBasisRandomValues) {
  auto generator = TestGenerator(kGeneratorTestSeed);
  const auto unit_vector_dist = UniformDistributionUnitVector<double, 3>{};

  constexpr int kNumberOfTestCases = 100;
  for (int i = 0; i < kNumberOfTestCases; ++i) {
    Vector3d x = unit_vector_dist(generator);
    const auto [y, z] = ExtendToOrthonormalBasis(x);
    constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

    EXPECT_THAT(y.norm(), DoubleNear(1, kEpsilon));
    EXPECT_THAT(z.norm(), DoubleNear(1, kEpsilon));

    EXPECT_THAT(x.dot(y), DoubleNear(0, kEpsilon));
    EXPECT_THAT(x.dot(z), DoubleNear(0, kEpsilon));
    EXPECT_THAT(y.dot(z), DoubleNear(0, kEpsilon));
  }
}

TEST(EigenMathVectorUtilsDeathTest, ScaleVectorToLimits) {
  VectorXd min(5), max(5), vec(5);

  min.resize(3);
  EXPECT_DEATH(ScaleVectorToLimits(min, max, &vec), "lower\\.rows");
  min.resize(5);
  max.resize(4);
  EXPECT_DEATH(ScaleVectorToLimits(min, max, &vec), "upper\\.rows.*");
  vec.resize(3);
  max.resize(5);
  EXPECT_DEATH(ScaleVectorToLimits(min, max, &vec), "upper\\.rows.*");
}

TEST(EigenMathVectorUtils, ScaleVectorToLimitsUnchanged) {
  Vector3d min = Vector3d::Constant(-1);
  Vector3d max = Vector3d::Constant(2);
  Vector3d vec = Vector3d::Constant(1);

  // Cases within limits.
  EXPECT_EQ(ScaleVectorToLimits(min, max, &vec), ScaleVectorResult::UNCHANGED);
  EXPECT_DOUBLE_EQ(vec[0], 1);
  EXPECT_DOUBLE_EQ(vec[1], 1);
  EXPECT_DOUBLE_EQ(vec[2], 1);

  max.setConstant(0.006);
  min.setConstant(-0.006);
  vec << -8.30737e-06, 5.57897e-07, 2.65359e-06;
  EXPECT_EQ(ScaleVectorToLimits(min, max, &vec), ScaleVectorResult::UNCHANGED);
}

TEST(EigenMathVectorUtils, ScaleVectorToLimitsVectorScaled) {
  Vector3d min = Vector3d::Constant(-1);
  Vector3d max = Vector3d::Constant(2);
  Vector3d vec{2.0, 3.0, 4.0};

  // Cases requiring rescaling.
  EXPECT_EQ(ScaleVectorToLimits(min, max, &vec),
            ScaleVectorResult::VECTOR_SCALED);
  EXPECT_NEAR(vec[0], 2.0 * 2.0 / 4.0, kEpsilon);
  EXPECT_NEAR(vec[1], 3.0 * 2.0 / 4.0, kEpsilon);
  EXPECT_NEAR(vec[2], 4.0 * 2.0 / 4.0, kEpsilon);

  vec << -2.0, -3.0, -4.0;
  EXPECT_EQ(ScaleVectorToLimits(min, max, &vec),
            ScaleVectorResult::VECTOR_SCALED);
  EXPECT_NEAR(vec[0], -2.0 * 1.0 / 4.0, kEpsilon);
  EXPECT_NEAR(vec[1], -3.0 * 1.0 / 4.0, kEpsilon);
  EXPECT_NEAR(vec[2], -4.0 * 1.0 / 4.0, kEpsilon);
}

TEST(EigenMathVectorUtils, ScaleVectorToLimitsElementsScaled) {
  Vector3d min{1, 0, 1};
  Vector3d max{2, 1, 2};
  Vector3d vec{2, -2, 2};

  // Requires per-element scaling.
  EXPECT_EQ(ScaleVectorToLimits(min, max, &vec),
            ScaleVectorResult::ELEMENTS_SCALED)
      << "Expected ELEMENTS_SCALED for min= " << min.transpose()
      << " max= " << max.transpose() << " vec= " << vec.transpose();

  EXPECT_NEAR(vec[0], 2, kEpsilon);
  EXPECT_NEAR(vec[1], 0, kEpsilon);
  EXPECT_NEAR(vec[2], 2, kEpsilon);

  // Infeasible case with upper limit < lower limit.
  max << 0, 0, 1;
  min << 1, -1, 0;
  vec << 2, 2, 2;
  EXPECT_EQ(ScaleVectorToLimits(min, max, &vec), ScaleVectorResult::INFEASIBLE);

  EXPECT_NEAR(vec[0], 0.5, kEpsilon);
  EXPECT_NEAR(vec[1], 0, kEpsilon);
  EXPECT_NEAR(vec[2], 1, kEpsilon);
}

TEST(EigenMathVectorUtils, ScaleDownToLimitsNoOps) {
  std::vector<std::tuple<Vector2d, Vector2d>> test_cases = {
      {Vector2d(1.0, 0.0), Vector2d(3.0, 4.0)},
      {Vector2d(-1.0, 0.0), Vector2d(3.0, 4.0)},
      {Vector2d(0.0, 4.0), Vector2d(3.0, 4.0)},
      {Vector2d(-1.0, -2.0), Vector2d(3.0, 4.0)},
      {Vector2d(-1.0, 4.0), Vector2d(3.0, 4.0)},
      {Vector2d(3.0, 4.0), Vector2d(3.0, 4.0)},
      {Vector2d(0.0, 0.0), Vector2d(3.0, 4.0)}};

  for (const auto& [vector, limits] : test_cases) {
    Vector2d scaled_vector = ScaleDownToLimits(vector, limits);
    // The output should be the same input vector.
    EXPECT_THAT(scaled_vector, IsApprox(vector, kEpsilon));
  }
}

TEST(EigenMathVectorUtils, ScaleDownToLimitsNeedsScaling) {
  std::vector<std::tuple<Vector2d, Vector2d>> test_cases = {
      {Vector2d(0.0, 5.0), Vector2d(3.0, 4.0)},
      {Vector2d(-4.0, 0.0), Vector2d(3.0, 4.0)},
      {Vector2d(-2.0, -5.0), Vector2d(3.0, 4.0)},
      {Vector2d(0.0, 5.0), Vector2d(3.0, 4.0)},
      {Vector2d(6.0, 5.0), Vector2d(3.0, 4.0)},
      {Vector2d(-2.0, 0.0), Vector2d(1e-6, 4.0)}};

  for (const auto& [vector, limits] : test_cases) {
    Vector2d scaled_vector = ScaleDownToLimits(vector, limits);

    // The output should not be the same input vector.
    EXPECT_THAT(scaled_vector, ::testing::Not(IsApprox(vector, kEpsilon)));

    // Check that the resulting vector keeps direction.
    double cos = std::clamp(
        scaled_vector.dot(vector) / (scaled_vector.norm() * vector.norm()),
        -1.0, 1.0);
    double angle = std::acos(cos);
    EXPECT_LT(angle, kEpsilon);

    // Check that the resulting vector is within limits.
    EXPECT_GE((limits - scaled_vector).minCoeff(), 0.0);
    EXPECT_LE((scaled_vector - limits).maxCoeff(), 0.0);
  }
}

TEST(EigenMathVectorUtils, ScaleDownToLimitsExpectDeath) {
  Vector2d vector(1.0, 0.0);
  Vector2d limits(-1.0, 2.0);
  EXPECT_DEATH(ScaleDownToLimits(vector, limits), ".*");
}

TEST(EigenMathVectorUtils, ScaleVectorResultToString) {
  EXPECT_THAT(ToString(ScaleVectorResult::ELEMENTS_SCALED),
              StrEq("eigenmath::ScaleVectorResult::ELEMENTS_SCALED"));
  EXPECT_THAT(ToString(ScaleVectorResult::VECTOR_SCALED),
              StrEq("eigenmath::ScaleVectorResult::VECTOR_SCALED"));
  EXPECT_THAT(ToString(ScaleVectorResult::INFEASIBLE),
              StrEq("eigenmath::ScaleVectorResult::INFEASIBLE"));
  EXPECT_THAT(ToString(ScaleVectorResult::UNCHANGED),
              StrEq("eigenmath::ScaleVectorResult::UNCHANGED"));
}

TEST(EigenMathVectorUtils, ApplyDeadbandWorksForFixedSize) {
  Vector2d input(12, -2);
  const Vector2d deadband(1, 0.2);
  const Vector2d expected_output(11, -1.8);

  EXPECT_THAT(ApplyDeadband(input, deadband), IsApprox(expected_output, 0.0));
  EXPECT_THAT(ApplyDeadband(input, Vector2d(Vector2d::Zero())),
              IsApprox(input, 0.0));
}
TEST(EigenMathVectorUtils, ApplyDeadbandWorksForDynamicSize) {
  VectorXd input(2);
  VectorXd deadband(2);
  VectorXd expected_output(2);
  input << 12, -2;
  deadband << 1, 0.2;
  expected_output << 11, -1.8;
  EXPECT_THAT(ApplyDeadband(input, deadband), IsApprox(expected_output, 0.0));
  EXPECT_THAT(ApplyDeadband(input, VectorXd(VectorXd::Zero(2))),
              IsApprox(input, 0.0));
}

TEST(EigenMathVectorUtilsDeathTest, ApplyDeadbandPanicsIfSizesDontMatch) {
  EXPECT_DEATH(ApplyDeadband(VectorXd(2), VectorXd(3)), "input\\.size");
}

}  // namespace
}  // namespace eigenmath
