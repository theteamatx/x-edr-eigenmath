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

#include "compare_vectors.h"

#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "types.h"

namespace eigenmath {
namespace {

TEST(CompareVectorsTest, VectorsAreEqual) {
  std::vector<Vector3d> points_a = {Vector3d(1, 2, 3), Vector3d(-2, -3, -4)};
  std::vector<Vector3d> points_b = {Vector3d(1, 2, 3), Vector3d(-2, -3, -4)};
  bool are_equal = VectorsAreEqual(absl::MakeConstSpan(points_a),
                                   absl::MakeConstSpan(points_b));
  EXPECT_TRUE(are_equal);

  points_a.push_back(Vector3d(2, 0, 0));
  are_equal = VectorsAreEqual(absl::MakeConstSpan(points_a),
                              absl::MakeConstSpan(points_b));
  EXPECT_FALSE(are_equal);

  points_b.push_back(Vector3d(2, 1, 0));
  are_equal = VectorsAreEqual(absl::MakeConstSpan(points_a),
                              absl::MakeConstSpan(points_b));
  EXPECT_FALSE(are_equal);
}

TEST(CompareVectorsTest, VectorsHaveExpectedSize) {
  std::vector<VectorNd> points;

  // Empty vectors always return true.
  bool is_expected = VectorsHaveExpectedSize(absl::MakeConstSpan(points), 10);
  EXPECT_TRUE(is_expected);

  points.push_back(VectorNd::Constant(2, 1.0));
  is_expected = VectorsHaveExpectedSize(absl::MakeConstSpan(points), 2);
  EXPECT_TRUE(is_expected);

  points.push_back(VectorNd::Constant(4, 1.0));
  is_expected = VectorsHaveExpectedSize(absl::MakeConstSpan(points), 2);
  EXPECT_FALSE(is_expected);
}

}  // namespace
}  // namespace eigenmath
