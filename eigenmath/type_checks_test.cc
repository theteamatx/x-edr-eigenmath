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

#include "type_checks.h"

#include "gtest/gtest.h"

namespace eigenmath {
namespace {

using Matrix23f = Eigen::Matrix<float, 2, 3>;
using ColumnVector3d = Eigen::Matrix<double, 3, 1>;
using RowVector3d = Eigen::Matrix<double, 1, 3>;

TEST(TestTypeChecks, IsPose) {
  EXPECT_FALSE(IsPose<Matrix23f>);
  EXPECT_FALSE(IsPose<ColumnVector3d>);
  EXPECT_FALSE(IsPose<RowVector3d>);
  EXPECT_FALSE(IsPose<SO2d>);
  EXPECT_TRUE(IsPose<Pose2f>);
  EXPECT_TRUE(IsPose<Pose3d>);
}

TEST(TestTypeChecks, IsPose2) {
  EXPECT_FALSE(IsPose2<Matrix23f>);
  EXPECT_FALSE(IsPose2<ColumnVector3d>);
  EXPECT_FALSE(IsPose2<RowVector3d>);
  EXPECT_FALSE(IsPose2<SO2d>);
  EXPECT_TRUE(IsPose2<Pose2f>);
  EXPECT_FALSE(IsPose2<Pose3d>);
}

TEST(TestTypeChecks, IsPose3) {
  EXPECT_FALSE(IsPose3<Matrix23f>);
  EXPECT_FALSE(IsPose3<ColumnVector3d>);
  EXPECT_FALSE(IsPose3<RowVector3d>);
  EXPECT_FALSE(IsPose3<SO2d>);
  EXPECT_FALSE(IsPose3<Pose2f>);
  EXPECT_TRUE(IsPose3<Pose3d>);
}

TEST(TestTypeChecks, ScalarTypeOf) {
  EXPECT_TRUE((std::is_same_v<ScalarTypeOf<Matrix23f>, float>));
  EXPECT_TRUE((std::is_same_v<ScalarTypeOf<ColumnVector3d>, double>));
  EXPECT_TRUE((std::is_same_v<ScalarTypeOf<RowVector3d>, double>));
  EXPECT_TRUE((std::is_same_v<ScalarTypeOf<SO2d>, double>));
  EXPECT_TRUE((std::is_same_v<ScalarTypeOf<Pose2f>, float>));
  EXPECT_TRUE((std::is_same_v<ScalarTypeOf<Pose3d>, double>));
}

}  // namespace
}  // namespace eigenmath
