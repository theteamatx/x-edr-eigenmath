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

#include "quadrature.h"

#include <cmath>

#include "gtest/gtest.h"

namespace eigenmath {
namespace {

// NOTE: Tolerances of the integrated values are not arbitrary,
// they are consistent with the level of approximation of the different
// orders of quadrature and the stiffness of the integrands.

TEST(GaussLegendreQuadrature, IntegralOfSin) {
  double omega = 1.0;
  auto f = [&omega](double x) -> double { return std::sin(omega * x); };

  double expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  double integ_2 = ComputeGaussLegendreIntegral(2, 1.0, 2.0, f);
  EXPECT_NEAR(integ_2, expected_integ, 0.0005);

  double integ_3 = ComputeGaussLegendreIntegral(3, 1.0, 2.0, f);
  EXPECT_NEAR(integ_3, expected_integ, 1.0e-6);

  double integ_4 = ComputeGaussLegendreIntegral(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 1.0e-9);

  double integ_5 = ComputeGaussLegendreIntegral(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 1.0e-12);

  double integ_6 = ComputeGaussLegendreIntegral(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 1.0e-15);

  double integ_7 = ComputeGaussLegendreIntegral(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 1.0e-15);

  double integ_8 = ComputeGaussLegendreIntegral(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 1.0e-15);

  double integ_9 = ComputeGaussLegendreIntegral(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 1.0e-15);

  double integ_10 = ComputeGaussLegendreIntegral(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 1.0e-15);

  double integ_12 = ComputeGaussLegendreIntegral(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 1.0e-15);

  omega = 2.0;
  expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  integ_2 = ComputeGaussLegendreIntegral(2, 1.0, 2.0, f);
  EXPECT_NEAR(integ_2, expected_integ, 0.001);

  integ_3 = ComputeGaussLegendreIntegral(3, 1.0, 2.0, f);
  EXPECT_NEAR(integ_3, expected_integ, 1.0e-5);

  integ_4 = ComputeGaussLegendreIntegral(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 5.0e-8);

  integ_5 = ComputeGaussLegendreIntegral(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 1.0e-10);

  integ_6 = ComputeGaussLegendreIntegral(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 5.0e-13);

  integ_7 = ComputeGaussLegendreIntegral(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 1.0e-15);

  integ_8 = ComputeGaussLegendreIntegral(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 1.0e-15);

  integ_9 = ComputeGaussLegendreIntegral(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 1.0e-15);

  integ_10 = ComputeGaussLegendreIntegral(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 1.0e-15);

  integ_12 = ComputeGaussLegendreIntegral(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 1.0e-15);

  omega = 5.0;
  expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  integ_3 = ComputeGaussLegendreIntegral(3, 1.0, 2.0, f);
  EXPECT_NEAR(integ_3, expected_integ, 0.01);

  integ_4 = ComputeGaussLegendreIntegral(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 0.0005);

  integ_5 = ComputeGaussLegendreIntegral(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 5.0e-6);

  integ_6 = ComputeGaussLegendreIntegral(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 5.0e-8);

  integ_7 = ComputeGaussLegendreIntegral(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 5.0e-10);

  integ_8 = ComputeGaussLegendreIntegral(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 5.0e-12);

  integ_9 = ComputeGaussLegendreIntegral(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 5.0e-14);

  integ_10 = ComputeGaussLegendreIntegral(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 1.0e-15);

  integ_12 = ComputeGaussLegendreIntegral(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 1.0e-15);

  omega = 10.0;
  expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  integ_4 = ComputeGaussLegendreIntegral(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 0.05);

  integ_5 = ComputeGaussLegendreIntegral(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 2.5e-3);

  integ_6 = ComputeGaussLegendreIntegral(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 1.0e-4);

  integ_7 = ComputeGaussLegendreIntegral(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 5.0e-6);

  integ_8 = ComputeGaussLegendreIntegral(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 1.0e-7);

  integ_9 = ComputeGaussLegendreIntegral(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 5.0e-9);

  integ_10 = ComputeGaussLegendreIntegral(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 5.0e-11);

  integ_12 = ComputeGaussLegendreIntegral(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 5.0e-15);
}

TEST(GaussLegendreQuadrature, IncrementalIntegralOfSin) {
  double omega = 1.0;
  double current_x = 1.0;
  auto f = [&omega, &current_x](double dx) -> double {
    current_x += dx;
    return std::sin(omega * current_x);
  };

  double expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  current_x = 1.0;
  double integ_2 = ComputeGaussLegendreIntegralIncrementally(2, 1.0, 2.0, f);
  EXPECT_NEAR(integ_2, expected_integ, 0.0005);

  current_x = 1.0;
  double integ_3 = ComputeGaussLegendreIntegralIncrementally(3, 1.0, 2.0, f);
  EXPECT_NEAR(integ_3, expected_integ, 1.0e-6);

  current_x = 1.0;
  double integ_4 = ComputeGaussLegendreIntegralIncrementally(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 1.0e-9);

  current_x = 1.0;
  double integ_5 = ComputeGaussLegendreIntegralIncrementally(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 1.0e-12);

  current_x = 1.0;
  double integ_6 = ComputeGaussLegendreIntegralIncrementally(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 1.0e-15);

  current_x = 1.0;
  double integ_7 = ComputeGaussLegendreIntegralIncrementally(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 1.0e-15);

  current_x = 1.0;
  double integ_8 = ComputeGaussLegendreIntegralIncrementally(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 1.0e-15);

  current_x = 1.0;
  double integ_9 = ComputeGaussLegendreIntegralIncrementally(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 1.0e-15);

  current_x = 1.0;
  double integ_10 = ComputeGaussLegendreIntegralIncrementally(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 1.0e-15);

  current_x = 1.0;
  double integ_12 = ComputeGaussLegendreIntegralIncrementally(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 1.0e-15);

  omega = 2.0;
  expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  current_x = 1.0;
  integ_2 = ComputeGaussLegendreIntegralIncrementally(2, 1.0, 2.0, f);
  EXPECT_NEAR(integ_2, expected_integ, 0.001);

  current_x = 1.0;
  integ_3 = ComputeGaussLegendreIntegralIncrementally(3, 1.0, 2.0, f);
  EXPECT_NEAR(integ_3, expected_integ, 1.0e-5);

  current_x = 1.0;
  integ_4 = ComputeGaussLegendreIntegralIncrementally(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 5.0e-8);

  current_x = 1.0;
  integ_5 = ComputeGaussLegendreIntegralIncrementally(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 1.0e-10);

  current_x = 1.0;
  integ_6 = ComputeGaussLegendreIntegralIncrementally(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 5.0e-13);

  current_x = 1.0;
  integ_7 = ComputeGaussLegendreIntegralIncrementally(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 1.0e-15);

  current_x = 1.0;
  integ_8 = ComputeGaussLegendreIntegralIncrementally(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 1.0e-15);

  current_x = 1.0;
  integ_9 = ComputeGaussLegendreIntegralIncrementally(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 1.0e-15);

  current_x = 1.0;
  integ_10 = ComputeGaussLegendreIntegralIncrementally(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 1.0e-15);

  current_x = 1.0;
  integ_12 = ComputeGaussLegendreIntegralIncrementally(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 1.0e-15);

  omega = 5.0;
  expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  current_x = 1.0;
  integ_3 = ComputeGaussLegendreIntegralIncrementally(3, 1.0, 2.0, f);
  EXPECT_NEAR(integ_3, expected_integ, 0.01);

  current_x = 1.0;
  integ_4 = ComputeGaussLegendreIntegralIncrementally(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 0.0005);

  current_x = 1.0;
  integ_5 = ComputeGaussLegendreIntegralIncrementally(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 5.0e-6);

  current_x = 1.0;
  integ_6 = ComputeGaussLegendreIntegralIncrementally(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 5.0e-8);

  current_x = 1.0;
  integ_7 = ComputeGaussLegendreIntegralIncrementally(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 5.0e-10);

  current_x = 1.0;
  integ_8 = ComputeGaussLegendreIntegralIncrementally(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 5.0e-12);

  current_x = 1.0;
  integ_9 = ComputeGaussLegendreIntegralIncrementally(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 5.0e-14);

  current_x = 1.0;
  integ_10 = ComputeGaussLegendreIntegralIncrementally(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 1.0e-15);

  current_x = 1.0;
  integ_12 = ComputeGaussLegendreIntegralIncrementally(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 1.0e-15);

  omega = 10.0;
  expected_integ = (std::cos(omega) - std::cos(2.0 * omega)) / omega;

  current_x = 1.0;
  integ_4 = ComputeGaussLegendreIntegralIncrementally(4, 1.0, 2.0, f);
  EXPECT_NEAR(integ_4, expected_integ, 0.05);

  current_x = 1.0;
  integ_5 = ComputeGaussLegendreIntegralIncrementally(5, 1.0, 2.0, f);
  EXPECT_NEAR(integ_5, expected_integ, 2.5e-3);

  current_x = 1.0;
  integ_6 = ComputeGaussLegendreIntegralIncrementally(6, 1.0, 2.0, f);
  EXPECT_NEAR(integ_6, expected_integ, 1.0e-4);

  current_x = 1.0;
  integ_7 = ComputeGaussLegendreIntegralIncrementally(7, 1.0, 2.0, f);
  EXPECT_NEAR(integ_7, expected_integ, 5.0e-6);

  current_x = 1.0;
  integ_8 = ComputeGaussLegendreIntegralIncrementally(8, 1.0, 2.0, f);
  EXPECT_NEAR(integ_8, expected_integ, 1.0e-7);

  current_x = 1.0;
  integ_9 = ComputeGaussLegendreIntegralIncrementally(9, 1.0, 2.0, f);
  EXPECT_NEAR(integ_9, expected_integ, 5.0e-9);

  current_x = 1.0;
  integ_10 = ComputeGaussLegendreIntegralIncrementally(10, 1.0, 2.0, f);
  EXPECT_NEAR(integ_10, expected_integ, 5.0e-11);

  current_x = 1.0;
  integ_12 = ComputeGaussLegendreIntegralIncrementally(12, 1.0, 2.0, f);
  EXPECT_NEAR(integ_12, expected_integ, 5.0e-15);
}

}  // namespace
}  // namespace eigenmath
