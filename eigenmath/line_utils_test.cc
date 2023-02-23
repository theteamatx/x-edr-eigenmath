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

#include "line_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "types.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

TEST(TransformLineTest, IdentityTransform) {
  Line2f line(Vector2f(1.0, 0.0), 0.0);
  Pose2f transform(Eigen::Vector2f(0.0f, 0.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(0.0, 1.0, 0.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, PureRotation) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(0.0f, 0.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(0.0, 1.0, 1.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, PureTranslation) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(1.0f, 0.0f), 0.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(1.0, 0.0, 0.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, PureTranslation2) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(-1.0f, 0.0f), 0.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(1.0, 0.0, 2.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, PureTranslation3) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(0.0f, 1.0f), 0.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(1.0, 0.0, 1.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, MixedTransform) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(0.0f, 1.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(0.0, 1.0, 0.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, MixedTransform2) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(1.0f, 0.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(0.0, 1.0, 1.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, MixedTransform3) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(0.0f, 1.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(0.0, 1.0, 0.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, MixedTransform4) {
  Line2f line(Eigen::Vector2f(1.0, 0.0), 1.0);
  Pose2f transform(Eigen::Vector2f(1.0f, 1.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(0.0, 1.0, 0.0), 1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, MixedTransform5) {
  Line2f line(Eigen::Vector2f(std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0), 1.0);
  Pose2f transform(Eigen::Vector2f(1.0f, 1.0f), M_PI / 2.0f);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(),
              IsApprox(Eigen::Vector3f(-std::sqrt(2.0) / 2.0,
                                       std::sqrt(2.0) / 2.0, 1.0),
                       1e-3))
      << rotated_line.coeffs();
}

TEST(TransformLineTest, MixedTransform6) {
  Line2f line(Eigen::Vector2f(0.705542, -0.708668), -1.0);
  Pose2f transform(Eigen::Vector2f(1, 3), -0.783185);
  Line2f rotated_line = transform * line;
  EXPECT_THAT(rotated_line.coeffs(), IsApprox(Eigen::Vector3f(0, -1, 2), 1e-3))
      << rotated_line.coeffs();
}

TEST(IntersectLinesTest, SelfIntersection) {
  Line2f line({1.0, 0.0}, 1.0);
  Vector2f intersection = Vector2f::Zero();
  EXPECT_FALSE(IntersectLines(line, line, &intersection));
}

TEST(IntersectLinesTest, IntersectsOther) {
  Line2f a({1.0, 0.0}, 1.0);
  Line2f b({0.0, 1.0}, -1.0);

  Vector2f intersection = Vector2f::Zero();
  EXPECT_TRUE(IntersectLines(a, b, &intersection));
  EXPECT_THAT(intersection, IsApprox(Eigen::Vector2f(-1.0, 1.0), 1e-3))
      << intersection.transpose();
}

TEST(IntersectLinesTest, ROEThreshold) {
  Line2f a({1.0, 0.0}, 1.0);
  Line2f almost_a_invalid({1.0, 1e-7}, 1.0);

  Vector2f intersection = Vector2f::Zero();
  EXPECT_FALSE(IntersectLines(a, almost_a_invalid, &intersection));

  Line2f almost_a_valid({1.0, 1e-3}, 1.0);
  EXPECT_TRUE(IntersectLines(a, almost_a_valid, &intersection));
  EXPECT_THAT(intersection, IsApprox(Eigen::Vector2f(-1.0, 0.0), 1e-9))
      << intersection.transpose();
}

}  // namespace
}  // namespace eigenmath
