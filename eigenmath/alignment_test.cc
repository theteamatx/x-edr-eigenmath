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

#include <optional>
#include <type_traits>

#include "Eigen/Geometry"
#include "absl/types/optional.h"
#include "gtest/gtest.h"
#include "pose2.h"
#include "pose3.h"
#include "so2.h"
#include "so3.h"
#include "types.h"

namespace eigenmath {
namespace {

TEST(EigenAlignment, VerifyAlignment) {
  // Fixed size vectorizable eigen types
  EXPECT_EQ(16, alignof(Vector2d));
  EXPECT_EQ(16, alignof(Vector4f));
  EXPECT_EQ(16, alignof(Vector4d));
  EXPECT_EQ(16, alignof(Vector6d));
  EXPECT_EQ(16, alignof(Matrix2d));
  EXPECT_EQ(16, alignof(Matrix2f));
  EXPECT_EQ(16, alignof(Matrix4d));
  EXPECT_EQ(16, alignof(Matrix4f));
  EXPECT_EQ(16, alignof(Matrix6d));
  EXPECT_EQ(16, alignof(Matrix6f));
  EXPECT_EQ(16, alignof(Quaterniond));
  EXPECT_EQ(16, alignof(Quaternionf));

  // Example of fixed sized non-vectorizable eigen types
  EXPECT_EQ(sizeof(float), alignof(Vector2f));
  EXPECT_EQ(sizeof(float), alignof(Vector3f));
  EXPECT_EQ(sizeof(double), alignof(Vector3d));
  EXPECT_EQ(sizeof(float), alignof(Vector6f));
  EXPECT_EQ(sizeof(float), alignof(Vector3f));
  EXPECT_EQ(sizeof(double), alignof(Vector3d));
  EXPECT_EQ(sizeof(float), alignof(Matrix3f));
  EXPECT_EQ(sizeof(double), alignof(Matrix3d));

  // Verify alignment requirements of container of fixed size vectorizable
  // types.
  EXPECT_EQ(16, alignof(SO2d));
  EXPECT_EQ(16, alignof(SO3f));
  EXPECT_EQ(16, alignof(SO3d));
  EXPECT_EQ(16, alignof(Pose2d));
  EXPECT_EQ(16, alignof(Pose3d));
  EXPECT_EQ(16, alignof(Pose3f));

  EXPECT_EQ(sizeof(float), alignof(SO2f));
  EXPECT_EQ(sizeof(float), alignof(Pose2f));

  // optional types
  EXPECT_EQ(16, alignof(std::optional<Vector2d>));
  EXPECT_EQ(16, alignof(std::optional<Vector4f>));
  EXPECT_EQ(16, alignof(std::optional<Vector4d>));
  EXPECT_EQ(16, alignof(std::optional<Vector6d>));
  EXPECT_EQ(16, alignof(std::optional<Matrix2d>));
  EXPECT_EQ(16, alignof(std::optional<Matrix2f>));
  EXPECT_EQ(16, alignof(std::optional<Matrix4d>));
  EXPECT_EQ(16, alignof(std::optional<Matrix4f>));
  EXPECT_EQ(16, alignof(std::optional<Matrix6d>));
  EXPECT_EQ(16, alignof(std::optional<Matrix6f>));
  EXPECT_EQ(16, alignof(std::optional<Quaterniond>));
  EXPECT_EQ(16, alignof(std::optional<Quaternionf>));
  EXPECT_EQ(16, alignof(std::optional<SO2d>));
  EXPECT_EQ(16, alignof(std::optional<SO3f>));
  EXPECT_EQ(16, alignof(std::optional<SO3d>));
  EXPECT_EQ(16, alignof(std::optional<Pose2d>));
  EXPECT_EQ(16, alignof(std::optional<Pose3d>));
  EXPECT_EQ(16, alignof(std::optional<Pose3f>));
}

}  // namespace
}  // namespace eigenmath
