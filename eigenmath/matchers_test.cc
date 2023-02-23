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

#include "matchers.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "line_utils.h"
#include "mean_and_covariance.h"
#include "types.h"

namespace eigenmath {
namespace testing {
namespace {

using ::testing::Not;
using ::testing::Pointwise;

TEST(IsApprox, CloseToZero) {
  const Vector3d kZeroVector = Vector3d::Zero();
  const Vector3d about_zero(0.0, 0.0, 1e-10);
  EXPECT_THAT(about_zero, IsApprox(kZeroVector, 1e-9));

  const Matrix3d kZeroMatrix = Matrix3d::Zero();
  Matrix3d close_to_zero = kZeroMatrix;
  close_to_zero(1, 2) = 1e-10;
  EXPECT_THAT(close_to_zero, IsApprox(kZeroMatrix, 1e-9));
}

TEST(IsApprox, SingleRowSingleColumn) {
  {
    Matrix<double, 1, 1> m1(0);
    Matrix<double, 1, 1> m2(0);
    EXPECT_THAT(m1, IsApprox(m2, 1e-9));
    EXPECT_THAT(m1, IsApprox(m2));
  }

  {
    Matrix<double, 1, 1> m1(0);
    Matrix<double, 1, 1> m2(1);
    EXPECT_THAT(m1, Not(IsApprox(m2, 1e-9)));
    EXPECT_THAT(m1, Not(IsApprox(m2)));
  }
}

TEST(IsApprox, ThreeByThree) {
  {
    const Matrix<double, 3, 3> m1 = Matrix<double, 3, 3>::Zero();
    const Matrix<double, 3, 3> m2 = Matrix<double, 3, 3>::Identity();
    EXPECT_THAT(m1, Not(IsApprox(m2, 1e-9)));
    EXPECT_THAT(m1, Not(IsApprox(m2)));
  }

  {
    const Matrix<double, 3, 3> m1 = Matrix<double, 3, 3>::Identity();
    const Matrix<double, 3, 3> m2 = Matrix<double, 3, 3>::Identity();
    EXPECT_THAT(m1, IsApprox(m2, 1e-9));
    EXPECT_THAT(m1, IsApprox(m2));
  }
}

TEST(IsApprox, EqualPoses) {
  Pose3d identity1;
  Pose3d identity2;
  EXPECT_THAT(identity1, IsApprox(identity2, 1e-9));
  EXPECT_THAT(identity1, IsApprox(identity2));
  EXPECT_THAT(identity1, IsApprox(identity2, 1e-5, 1e-5));
}

TEST(IsApprox, InequalPoses) {
  Pose3d identity1;
  Pose3d offset(Vector3d(1, 2, 3));
  EXPECT_THAT(identity1, Not(IsApprox(offset, 1e-9)));
  EXPECT_THAT(identity1, Not(IsApprox(offset)));
  EXPECT_THAT(identity1, Not(IsApprox(offset, 1e-5, 1e-5)));
}

TEST(IsApprox, ContainerMatchers) {
  Matrix<double, 1, 1> m1(0);
  Matrix<double, 1, 1> m2(1);
  Matrix<double, 1, 1> m3(2);
  Matrix<double, 1, 1> m4(3);

  std::vector<Matrix<double, 1, 1>> container{m1, m2, m3};

  EXPECT_THAT(container, Pointwise(IsApproxTuple(1e-9), {m1, m2, m3}));
  EXPECT_THAT(container, Pointwise(IsApproxTuple(), {m1, m2, m3}));

  EXPECT_THAT(container, Not(Pointwise(IsApproxTuple(1e-9), {m2, m3, m4})));
  EXPECT_THAT(container, Not(Pointwise(IsApproxTuple(), {m2, m3, m4})));
}

TEST(IsApprox, EqualQuaternions) {
  EXPECT_THAT(Quaternionf(1, 2, 3, 4), IsApprox(Quaternionf(1, 2, 3, 4)));
  EXPECT_THAT(Quaternionf(1, 2, 3, 4),
              IsApprox(Quaternionf(1 + 1e-4, 2, 3, 4), 1e-3f));

  // Tolerance is not of the quaternion's scalar type.
  EXPECT_THAT(Quaternionf(1, 2, 3, 4),
              IsApprox(Quaternionf(1 + 1e-4, 2, 3, 4), 1e-3));
}

TEST(IsApprox, NegatedQuaternions) {
  EXPECT_THAT(Quaternionf(1, 2, 3, 4),
              Not(IsApprox(Quaternionf(-1 + 1e-4, -2, -3, -4), 1e-3f)));
  // Tolerance is not of the quaternion's scalar type.
  EXPECT_THAT(Quaternionf(1, 2, 3, 4),
              Not(IsApprox(Quaternionf(-1 + 1e-4, -2, -3, -4), 1e-3)));
}

TEST(IsApprox, InEqualQuaternions) {
  EXPECT_THAT(Quaternionf(1, 2, 3, 4), Not(IsApprox(Quaternionf(4, 3, 2, 1))));
  EXPECT_THAT(Quaternionf(1, 2, 3, 4),
              Not(IsApprox(Quaternionf(1 + 1e-4, 2, 3, 4), 1e-6)));
}

TEST(IsApproxUndirected, SingleElement) {
  constexpr double kTolerance = 1e-5;
  const LineSegment2d original{{0.0, 1.0}, {2.0, 3.0}};
  const Vector2d disturbance = {1e-10, 2e-9};
  const LineSegment2d disturbed{original.from + disturbance,
                                original.to - disturbance};
  EXPECT_THAT(original, IsApproxUndirected(disturbed, kTolerance));
  const LineSegment2d flipped{disturbed.to, disturbed.from};
  EXPECT_THAT(original, IsApproxUndirected(flipped, kTolerance));
  const LineSegment2d negated = {-original.from, -original.to};
  EXPECT_THAT(original, Not(IsApproxUndirected(negated, kTolerance)));
}

TEST(IsApproxUndirected, Sequence) {
  constexpr double kTolerance = 1e-5;
  const LineSegment2d kValue = {{0.0, 1.0}, {2.0, 3.0}};
  const LineSegment2d original[] = {
      kValue, kValue, {2 * kValue.from, 100 * kValue.to}};
  EXPECT_THAT(original, Pointwise(IsApproxUndirected(kTolerance), original));
  const LineSegment2d same_length_list[3] = {};
  EXPECT_THAT(original,
              Not(Pointwise(IsApproxUndirected(kTolerance), same_length_list)));
}

TEST(IsApproxMeanAndCovariance, IsApproxPoseAndCovariance3d) {
  EXPECT_THAT(PoseAndCovariance3d(),
              IsApproxMeanAndCovariance(PoseAndCovariance3d()));
  const PoseAndCovariance3d a_pose_b{
      Pose3d{Quaterniond(Eigen::AngleAxisd(1.0, Vector3d(0.3, 0.5, 0.7))),
             Vector3d(0.1, 0.2, 0.4)},
      Matrix6d(Vector6d(1, 2, 3, 4, 5, 6).asDiagonal())};
  EXPECT_THAT(a_pose_b, IsApproxMeanAndCovariance(a_pose_b));
  const PoseAndCovariance3d aa_pose_bb{a_pose_b.mean.inverse().inverse(),
                                       a_pose_b.covariance.inverse().inverse()};
  EXPECT_THAT(a_pose_b, IsApproxMeanAndCovariance(aa_pose_bb));
  const PoseAndCovariance3d x_pose_y{a_pose_b.mean.inverse(),
                                     a_pose_b.covariance.inverse()};
  EXPECT_THAT(a_pose_b, Not(IsApproxMeanAndCovariance(x_pose_y)));
}

TEST(IsApproxMeanAndCovariance, IsApproxEuclideanMeanAndCovariance) {
  const EuclideanMeanAndCovariance<double, 3> a_translation_b(
      Vector3d(0.1, 0.2, 0.3),
      CreateCovarianceAngleAxis(1.0, Vector3d(2, 3, 4), Vector3d(5, 6, 7)));
  EXPECT_THAT(a_translation_b, IsApproxMeanAndCovariance(a_translation_b));
  const EuclideanMeanAndCovariance<double, 3> x_translation_y(
      Vector3d(1.1, 2.2, 3.3),
      CreateCovarianceAngleAxis(2.0, Vector3d(22, 33, 44),
                                Vector3d(55, 66, 77)));
  EXPECT_THAT(a_translation_b, Not(IsApproxMeanAndCovariance(x_translation_y)));
}

}  // namespace
}  // namespace testing
}  // namespace eigenmath
