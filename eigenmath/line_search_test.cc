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

#include "line_search.h"

#include <limits>

#include "gtest/gtest.h"

namespace eigenmath {
namespace {

template <typename T>
class EigenmathGoldenSectionSearchMinimizeTest : public ::testing::Test {};
TYPED_TEST_SUITE_P(EigenmathGoldenSectionSearchMinimizeTest);

TYPED_TEST_P(EigenmathGoldenSectionSearchMinimizeTest, WorksAsExpected) {
  using Scalar = TypeParam;

  // As the test functions are quadratic, any x change smaller than sqrt(eps)
  // is lost, so choose the tolerance accordingly.
  const Scalar kTolerance =
      Scalar{1} * std::sqrt(std::numeric_limits<Scalar>::epsilon());
  Scalar best_x, best_f;

  auto quad_func = [](Scalar x) -> Scalar {
    return (x - Scalar{0.5}) * (x - Scalar{0.5}) + Scalar{0.75};
  };

  // Basic case:
  std::tie(best_x, best_f) = GoldenSectionSearchMinimize(
      Scalar{0.0}, Scalar{1.0}, quad_func, kTolerance);
  EXPECT_NEAR(best_x, Scalar{0.5}, kTolerance);
  EXPECT_NEAR(best_f, Scalar{0.75}, kTolerance);

  // Decreasing to upper-bound
  std::tie(best_x, best_f) = GoldenSectionSearchMinimize(
      Scalar{-1.0}, Scalar{0.0}, quad_func, kTolerance);
  EXPECT_NEAR(best_x, Scalar{0.0}, kTolerance);
  EXPECT_NEAR(best_f, Scalar{1.0}, kTolerance);

  // Flipped bounds:
  std::tie(best_x, best_f) = GoldenSectionSearchMinimize(
      Scalar{1.0}, Scalar{0.0}, quad_func, kTolerance);
  EXPECT_NEAR(best_x, Scalar{0.5}, kTolerance);
  EXPECT_NEAR(best_f, Scalar{0.75}, kTolerance);

  // Increasing from lower-bound
  std::tie(best_x, best_f) = GoldenSectionSearchMinimize(
      Scalar{1.0}, Scalar{2.0}, quad_func, kTolerance);
  EXPECT_NEAR(best_x, Scalar{1.0}, kTolerance);
  EXPECT_NEAR(best_f, Scalar{1.0}, kTolerance);

  auto dipping_func = [](double x) -> double {
    if (x < Scalar{0.24} || x > Scalar{0.26}) {
      return Scalar{0.001} * (x - Scalar{0.25}) * (x - Scalar{0.25}) +
             Scalar{1.0};
    } else {
      return Scalar{1.0000001} *
             std::cos((x - Scalar{0.24}) * Scalar{M_PI / 0.01});
    }
  };
  std::tie(best_x, best_f) = GoldenSectionSearchMinimize(
      Scalar{0.0}, Scalar{1.0}, dipping_func, kTolerance);
  EXPECT_NEAR(best_x, Scalar{0.25}, kTolerance);
  EXPECT_NEAR(best_f, Scalar{-1.0000001}, kTolerance);
}

REGISTER_TYPED_TEST_SUITE_P(EigenmathGoldenSectionSearchMinimizeTest,
                            WorksAsExpected);

typedef ::testing::Types<float, double> FPTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(EigenmathGoldenSectionSearchMinimizeTestSuite,
                               EigenmathGoldenSectionSearchMinimizeTest,
                               FPTypes);

TEST(TestEigenmathLineSearch, BisectionSearchZeroCross) {
  const double tolerance = 1e-5;
  double best_lower_x, best_upper_x;

  double x_limit = 0.2;
  auto step_func = [&](double x) -> double {
    return (x < x_limit ? -1.0 : 1.0);
  };
  // Basic case:
  EXPECT_EQ(BisectionSearchZeroCross(0.0, 1.0, step_func, tolerance,
                                     &best_lower_x, &best_upper_x),
            0);
  EXPECT_LE(best_lower_x, 0.2);
  EXPECT_GE(best_upper_x, 0.2);
  EXPECT_NEAR(best_lower_x, 0.2 - tolerance, tolerance);
  EXPECT_NEAR(best_upper_x, 0.2 + tolerance, tolerance);

  // Reversed bounds case:
  EXPECT_EQ(BisectionSearchZeroCross(1.0, 0.0, step_func, tolerance,
                                     &best_lower_x, &best_upper_x),
            0);
  EXPECT_LE(best_upper_x, 0.2);
  EXPECT_GE(best_lower_x, 0.2);
  EXPECT_NEAR(best_upper_x, 0.2 - tolerance, tolerance);
  EXPECT_NEAR(best_lower_x, 0.2 + tolerance, tolerance);

  // All negative case:
  x_limit = 1.2;
  EXPECT_EQ(BisectionSearchZeroCross(0.0, 1.0, step_func, tolerance,
                                     &best_lower_x, &best_upper_x),
            -1);

  // All positive case:
  x_limit = -0.2;
  EXPECT_EQ(BisectionSearchZeroCross(0.0, 1.0, step_func, tolerance,
                                     &best_lower_x, &best_upper_x),
            1);
}

}  // namespace
}  // namespace eigenmath
