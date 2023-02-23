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

#include "so3.h"

#include <random>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "distribution.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "manifolds.h"
#include "matchers.h"
#include "sampling.h"
#include "utils.h"

namespace eigenmath {
namespace {

using eigenmath::testing::IsApprox;
using ::testing::Not;

TEST(TestSO3, ConstructorIdentity) {
  SO3d p;
  EXPECT_DOUBLE_EQ(1, p.quaternion().w());
  EXPECT_DOUBLE_EQ(0, p.quaternion().x());
  EXPECT_DOUBLE_EQ(0, p.quaternion().y());
  EXPECT_DOUBLE_EQ(0, p.quaternion().z());
}

TEST(TestSO3, ConstructorMatrix) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionSO3<> so3_dist;
  for (int i = 0; i < 50; i++) {
    Eigen::Matrix3d R = so3_dist(generator).matrix();
    SO3d so3{R};
    EXPECT_THAT(R, IsApprox(so3.matrix(), 1e-12));
  }
}

TEST(TestSO3, Cast) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionSO3<> so3_dist;
  for (int i = 0; i < 50; i++) {
    const Eigen::Matrix3d R = so3_dist(generator).matrix();
    const Eigen::Matrix3f Rf = R.cast<float>();

    SO3d p{R};
    SO3f pf{Rf};

    EXPECT_THAT(Rf.matrix(), IsApprox(pf.matrix(), 1e-5));
    Eigen::Matrix3f R_out = p.matrix().eval().cast<float>();
    Eigen::Matrix3f Rf_out = p.cast<float>().matrix();
    EXPECT_THAT(R_out, IsApprox(Rf_out, 1e-5));
  }
}

TEST(TestSO3, ConstructorQuaternion) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionVector<double, 4> dist;
  for (int i = 0; i < 50; i++) {
    Quaterniond q(dist(generator));
    SO3d so3{q};
    EXPECT_DOUBLE_EQ(so3.quaternion().norm(), 1.);
    Quaterniond unit_q = q.normalized();
    EXPECT_THAT(unit_q.coeffs(),
                IsApprox(SO3d{unit_q}.quaternion().coeffs(), 1e-12));
  }
}

TEST(TestSO3, ConstructorInvalidQuaternion) {
  Quaterniond q(0.0, 0.0, 0.0, 0.0);
  EXPECT_DEATH(SO3d{q}, "");
}

TEST(TestSO3, AssignmentInvalidQuaternion) {
  Quaterniond q(0.0, 0.0, 0.0, 0.0);
  SO3d quaternion;
  EXPECT_DEATH(quaternion = q, "");
}

TEST(TestSO3, Inverse) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionSO3<> so3_dist;
  for (int i = 0; i < 50; i++) {
    const auto so3 = so3_dist(generator);
    const SO3d so3_inv = so3.inverse();
    EXPECT_THAT(so3_inv.matrix(), IsApprox(so3.matrix().transpose(), 1e-12));
  }
}

TEST(TestSO3, Norm) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionVector3d vec_dist;
  UniformDistributionSO3<> so3_dist;
  for (int i = 0; i < 50; i++) {
    auto so3 = so3_dist(generator);
    SO3d so3_inv = so3.inverse();
    EXPECT_NEAR(so3_inv.norm(), so3.norm(), 1e-12);
    EXPECT_NEAR((so3_inv * so3).norm(), 0.0, 1e-12);
  }
  for (int i = 0; i < 50; i++) {
    Vector3d v3 = vec_dist(generator);
    SO3d so3 = ExpSO3(v3);
    EXPECT_NEAR(so3.norm(), v3.norm(), 1e-12);
    EXPECT_NEAR(so3.inverse().norm(), v3.norm(), 1e-12);
    SO3d so3_flip = so3;
    so3_flip.quaternion().coeffs() = -so3_flip.quaternion().coeffs();
    EXPECT_NEAR(so3_flip.norm(), v3.norm(), 1e-12);
    EXPECT_NEAR(so3_flip.inverse().norm(), v3.norm(), 1e-12);
  }
}

TEST(TestSO3, MultiplyPoses) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionSO3<> so3_dist;
  for (int i = 0; i < 50; i++) {
    SO3d a_R_b = so3_dist(generator);
    SO3d b_R_c = so3_dist(generator);
    SO3d a_R_c = a_R_b * b_R_c;
    EXPECT_THAT(a_R_c.matrix(),
                IsApprox(a_R_b.matrix() * b_R_c.matrix(), 1e-12));
  }
}

TEST(TestSO3, MultiplyPoints) {
  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionVector3d vec_dist;
  UniformDistributionSO3<> so3_dist;
  for (int i = 0; i < 50; i++) {
    SO3d a_R_b = so3_dist(generator);
    Eigen::Vector3d point_b = vec_dist(generator);

    Eigen::Vector3d point_a = a_R_b * point_b;
    EXPECT_THAT(point_a, IsApprox(a_R_b.matrix() * point_b, 1e-12));
  }
}

TEST(TestSO3, ConvertToRPY) {
  {  // identity
    SO3d s(0.0, 0.0, 0.0);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, 0.0, 1e-12);
    EXPECT_NEAR(yaw, 0.0, 1e-12);
  }
  {  // yaw
    SO3d s(0.0, 0.0, M_PI_4);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, 0.0, 1e-12);
    EXPECT_NEAR(yaw, M_PI_4, 1e-12);
  }
  {  // yaw
    SO3d s(0.0, 0.0, M_PI_2);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, 0.0, 1e-12);
    EXPECT_NEAR(yaw, M_PI_2, 1e-12);
  }
  {  // pitch
    SO3d s(0.0, M_PI_4, 0.0);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, M_PI_4, 1e-12);
    EXPECT_NEAR(yaw, 0.0, 1e-12);
  }
  {  // pitch
    SO3d s(0.0, M_PI_2, 0.0);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, M_PI_2, 1e-12);
    EXPECT_NEAR(yaw, 0.0, 1e-12);
  }
  {  // roll
    SO3d s(M_PI_4, 0.0, 0.0);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, M_PI_4, 1e-12);
    EXPECT_NEAR(pitch, 0.0, 1e-12);
    EXPECT_NEAR(yaw, 0.0, 1e-12);
  }
  {  // roll
    SO3d s(M_PI_2, 0.0, 0.0);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, M_PI_2, 1e-12);
    EXPECT_NEAR(pitch, 0.0, 1e-12);
    EXPECT_NEAR(yaw, 0.0, 1e-12);
  }
  {  // mixed
    SO3d s(M_PI_4, -M_PI_4, M_PI_2);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, M_PI_4, 1e-12);
    EXPECT_NEAR(pitch, -M_PI_4, 1e-12);
    EXPECT_NEAR(yaw, M_PI_2, 1e-12);
  }
  {  // extreme pitch below ROE threshold
    SO3d s(M_PI_4, M_PI_2 - 2.0e-7, -M_PI_4);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, M_PI_4, 1e-9);
    EXPECT_NEAR(pitch, M_PI_2, 2.0e-7);
    EXPECT_NEAR(yaw, -M_PI_4, 1e-9);
  }
  {  // extreme pitch below ROE threshold
    SO3d s(M_PI_4, -M_PI_2 + 2.0e-7, M_PI_4);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, M_PI_4, 1e-9);
    EXPECT_NEAR(pitch, -M_PI_2, 2.0e-7);
    EXPECT_NEAR(yaw, M_PI_4, 1e-9);
  }
  {  // extreme pitch above ROE threshold
    SO3d s(M_PI_4, M_PI_2 - 1.0e-9, -M_PI_4);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, M_PI_2, 1e-12);
    EXPECT_NEAR(yaw, -M_PI_2, 1e-9);
  }
  {  // extreme pitch above ROE threshold
    SO3d s(M_PI_4, -M_PI_2 + 1.0e-9, M_PI_4);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 0.0, 1e-12);
    EXPECT_NEAR(pitch, -M_PI_2, 1e-12);
    EXPECT_NEAR(yaw, M_PI_2, 1e-9);
  }
  {  // extreme pitch below ROE threshold
    SO3d s(2.0e-7, M_PI_2 - 2.0e-7, 2.0e-7 - M_PI_2);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, 2.0e-7, 1e-9);
    EXPECT_NEAR(pitch, M_PI_2, 2.0e-7);
    EXPECT_NEAR(yaw, 2.0e-7 - M_PI_2, 1e-9);
  }
  {  // extreme pitch below ROE threshold
    SO3d s(M_PI_2 - 2.0e-7, -M_PI_2 + 2.0e-7, 2.0e-7);
    double roll, pitch, yaw;
    SO3ToRPY(s, &roll, &pitch, &yaw);
    EXPECT_NEAR(roll, M_PI_2 - 2.0e-7, 1e-9);
    EXPECT_NEAR(pitch, -M_PI_2, 2.0e-7);
    EXPECT_NEAR(yaw, 2.0e-7, 1e-9);
  }
}

TEST(TestSO3, MakeDotProductPositive) {
  Vector4d positive_vector = MakeVector({1.0, 0.0, 0.0, 0.0});
  Vector4d negative_vector = MakeVector({-1.0, 0.0, 0.0, 0.0});

  // NOTE: Eigen now has an explicit constructors
  SO3d positive_quaternion(Quaterniond(positive_vector.data()));
  SO3d positive_quaternion2(Quaterniond(positive_vector.data()));
  SO3d negative_quaternion(Quaterniond(negative_vector.data()));

  // Positive and negative quaternions should already be the same rotation
  // using isApprox()
  EXPECT_THAT(positive_quaternion, IsApprox(negative_quaternion));
  // However, their quaternion representations aren't the same.
  EXPECT_THAT(positive_quaternion.quaternion(),
              Not(IsApprox(negative_quaternion.quaternion())));

  // Flip the positive to match the sign of the negative.
  positive_quaternion2.MakeDotProductPositive(negative_quaternion);
  // Still the same rotation as before.
  EXPECT_THAT(positive_quaternion2, IsApprox(positive_quaternion));
  // Quaternion is also the same as negative_quaternion now.
  EXPECT_THAT(positive_quaternion2.quaternion(),
              IsApprox(negative_quaternion.quaternion()));

  // One more flip, and nothing should change this time.
  positive_quaternion2.MakeDotProductPositive(negative_quaternion);
  EXPECT_THAT(positive_quaternion2, IsApprox(positive_quaternion));
  EXPECT_THAT(positive_quaternion2.quaternion(),
              IsApprox(negative_quaternion.quaternion()));
}

TEST(SO3NormalizationCheckerTest, WorksForNormalizedQuaterniond) {
  Quaterniond q(1.0, 0.0, 0.0, 0.0);
  const internal::SO3NormalizationChecker checker(q);
  EXPECT_TRUE(checker.IsNormalized());
}

TEST(SO3NormalizationCheckerTest, WorksForNotNormalizedQuaterniond) {
  Quaterniond q(0.0, 0.0, 0.0, 0.0);
  const internal::SO3NormalizationChecker checker(q);
  EXPECT_FALSE(checker.IsNormalized());
}

}  // namespace
}  // namespace eigenmath
