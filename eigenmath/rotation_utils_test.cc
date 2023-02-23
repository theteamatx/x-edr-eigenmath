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

#include "rotation_utils.h"

#include <tuple>

#include "gtest/gtest.h"
#include "matchers.h"
#include "types.h"

namespace eigenmath {
namespace {

template <typename RotationType>
class RotationUtilsTest : public ::testing::Test {};

class TwoVectorParameterizedDeathTest
    : public ::testing::TestWithParam<std::tuple<Vector3d, Vector3d>> {};

class TwoVectorParameterizedValidTest
    : public ::testing::TestWithParam<std::tuple<Vector3d, Vector3d>> {};

static const Vector3d kValidFromTwoVectors[] = {
    Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ(),
    Vector3d(0, 1, 2), Vector3d(0, 0, 2), Vector3d(0.1, 0.01, -0.12)};

static const Vector3d kInvalidFromTwoVectors0[] = {
    Vector3d::Zero(),
    Vector3d(3, 1, 2),
};

static const Vector3d kInvalidFromTwoVectors1[] = {
    Vector3d::Zero(),
    -kInvalidFromTwoVectors0[1],
};

using eigenmath::testing::IsApprox;
using RotationTypes = ::testing::Types<Matrix3d, Quaterniond>;
TYPED_TEST_SUITE(RotationUtilsTest, RotationTypes);

TYPED_TEST(RotationUtilsTest, ConvertToRPYIdentity) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(0.0, 0.0, 0.0);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, 0.0, 1e-12);
  EXPECT_NEAR(yaw, 0.0, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYYaw) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(0.0, 0.0, M_PI_4);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, 0.0, 1e-12);
  EXPECT_NEAR(yaw, M_PI_4, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYYaw2) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(0.0, 0.0, M_PI_2);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, 0.0, 1e-12);
  EXPECT_NEAR(yaw, M_PI_2, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYPitch) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(0.0, M_PI_4, 0.0);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, M_PI_4, 1e-12);
  EXPECT_NEAR(yaw, 0.0, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYPitch2) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(0.0, M_PI_2, 0.0);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, M_PI_2, 1e-12);
  EXPECT_NEAR(yaw, 0.0, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYRoll) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(M_PI_4, 0.0, 0.0);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, M_PI_4, 1e-12);
  EXPECT_NEAR(pitch, 0.0, 1e-12);
  EXPECT_NEAR(yaw, 0.0, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYRoll2) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(M_PI_2, 0.0, 0.0);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, M_PI_2, 1e-12);
  EXPECT_NEAR(pitch, 0.0, 1e-12);
  EXPECT_NEAR(yaw, 0.0, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYMixed) {
  using RotationType = TypeParam;
  RotationType rot = RotationFromRPY<RotationType>(M_PI_4, -M_PI_4, M_PI_2);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, M_PI_4, 1e-12);
  EXPECT_NEAR(pitch, -M_PI_4, 1e-12);
  EXPECT_NEAR(yaw, M_PI_2, 1e-12);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYExtremePitchBelowROEThreshold) {
  using RotationType = TypeParam;
  RotationType rot =
      RotationFromRPY<RotationType>(M_PI_4, M_PI_2 - 2.0e-7, -M_PI_4);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, M_PI_4, 1e-9);
  EXPECT_NEAR(pitch, M_PI_2, 2.0e-7);
  EXPECT_NEAR(yaw, -M_PI_4, 1e-9);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYExtremePitchBelowROEThreshold2) {
  using RotationType = TypeParam;
  RotationType rot =
      RotationFromRPY<RotationType>(M_PI_4, -M_PI_2 + 2.0e-7, M_PI_4);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, M_PI_4, 1e-9);
  EXPECT_NEAR(pitch, -M_PI_2, 2.0e-7);
  EXPECT_NEAR(yaw, M_PI_4, 1e-9);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYExtremePitchAboveROEThreshold) {
  using RotationType = TypeParam;
  RotationType rot =
      RotationFromRPY<RotationType>(M_PI_4, M_PI_2 - 1.0e-9, -M_PI_4);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, M_PI_2, 1e-12);
  EXPECT_NEAR(yaw, -M_PI_2, 1e-9);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYExtremePitchAboveROEThreshold2) {
  using RotationType = TypeParam;
  RotationType rot =
      RotationFromRPY<RotationType>(M_PI_4, -M_PI_2 + 1.0e-9, M_PI_4);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 0.0, 1e-12);
  EXPECT_NEAR(pitch, -M_PI_2, 1e-12);
  EXPECT_NEAR(yaw, M_PI_2, 1e-9);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYExtremePitchBelowROEThreshold3) {
  using RotationType = TypeParam;
  RotationType rot =
      RotationFromRPY<RotationType>(2.0e-7, M_PI_2 - 2.0e-7, 2.0e-7 - M_PI_2);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, 2.0e-7, 1e-9);
  EXPECT_NEAR(pitch, M_PI_2, 2.0e-7);
  EXPECT_NEAR(yaw, 2.0e-7 - M_PI_2, 1e-9);
}

TYPED_TEST(RotationUtilsTest, ConvertToRPYExtremePitchBelowROEThreshold4) {
  using RotationType = TypeParam;
  RotationType rot =
      RotationFromRPY<RotationType>(M_PI_2 - 2.0e-7, -M_PI_2 + 2.0e-7, 2.0e-7);
  double roll, pitch, yaw;
  RotationToRPY(rot, &roll, &pitch, &yaw);
  EXPECT_NEAR(roll, M_PI_2 - 2.0e-7, 1e-9);
  EXPECT_NEAR(pitch, -M_PI_2, 2.0e-7);
  EXPECT_NEAR(yaw, 2.0e-7, 1e-9);
}

TEST_P(TwoVectorParameterizedDeathTest, QuaternionFromTwoVectorsDeath) {
  const auto& [v0, v1] = GetParam();
  EXPECT_DEATH(QuaternionFromTwoVectorsNotAntiParallel(v0, v1), "Check failed");
}

TEST_P(TwoVectorParameterizedValidTest,
       QuaternionFromTwoVectorsEqualsEigenImplementation) {
  const auto& [v0, v1] = GetParam();
  const Quaterniond expected_quaternion(Quaterniond::FromTwoVectors(v0, v1));
  const Quaterniond quaternion(QuaternionFromTwoVectorsNotAntiParallel(v0, v1));

  EXPECT_THAT(quaternion.coeffs(), IsApprox(expected_quaternion.coeffs()));
}

INSTANTIATE_TEST_SUITE_P(
    FromTwoVectorsDeathTests, TwoVectorParameterizedDeathTest,
    ::testing::Combine(::testing::ValuesIn(kInvalidFromTwoVectors0),
                       ::testing::ValuesIn(kInvalidFromTwoVectors1)));

INSTANTIATE_TEST_SUITE_P(
    FromTwoVectorsValidTests, TwoVectorParameterizedValidTest,
    ::testing::Combine(::testing::ValuesIn(kValidFromTwoVectors),
                       ::testing::ValuesIn(kValidFromTwoVectors)));

}  // namespace
}  // namespace eigenmath
