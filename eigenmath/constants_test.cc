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

#include "constants.h"

#include <cmath>

#include "gtest/gtest.h"

namespace eigenmath {
namespace {

TEST(TestConstants, RadianFromDegree) {
  EXPECT_DOUBLE_EQ(RadianFromDegree(45.0), 45.0 * M_PI / 180.0);
  EXPECT_DOUBLE_EQ(RadianFromDegree(132.0), 132.0 * M_PI / 180.0);
  EXPECT_FLOAT_EQ(RadianFromDegree(45.0f), 45.0f * M_PI / 180.0f);
  EXPECT_FLOAT_EQ(RadianFromDegree(132.0f), 132.0f * M_PI / 180.0f);
}

TEST(TestConstants, DegreeFromRadian) {
  EXPECT_DOUBLE_EQ(DegreeFromRadian(0.34), 0.34 * 180.0 / M_PI);
  EXPECT_DOUBLE_EQ(DegreeFromRadian(1.42), 1.42 * 180.0 / M_PI);
  EXPECT_FLOAT_EQ(DegreeFromRadian(0.34f), 0.34f * 180.0f / M_PI);
  EXPECT_FLOAT_EQ(DegreeFromRadian(1.42f), 1.42f * 180.0f / M_PI);
}

}  // namespace
}  // namespace eigenmath
