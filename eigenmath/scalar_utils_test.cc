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

#include "scalar_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "sampling.h"

namespace eigenmath {
namespace {

using ::testing::DoubleNear;

constexpr int kBatchSize = 1000;

// Returns a list of values <= 0.
std::vector<double> NonPositiveSamples(const int batch_size) {
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist;
  std::vector<double> test_values(batch_size);
  std::generate_n(test_values.begin(), batch_size,
                  [&]() { return -std::abs(dist(generator)); });
  return test_values;
}

TEST(UtilityFunctions, Saturate) {
  EXPECT_DOUBLE_EQ(Saturate(7, 9), 7);
  EXPECT_DOUBLE_EQ(Saturate(-17, 9), -9);
  EXPECT_DOUBLE_EQ(Saturate(32, 9), 9);
}

TEST(UtilityFunctions, Radians) {
  EXPECT_DOUBLE_EQ(Radians(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Radians(90.0), M_PI / 2.0);
  EXPECT_DOUBLE_EQ(Radians(180.0), M_PI);
  EXPECT_DOUBLE_EQ(Radians(360.0), 2.0 * M_PI);

  // Prevent rounding for integer types.
  EXPECT_FLOAT_EQ(Radians<int>(0), 0.0);
  EXPECT_FLOAT_EQ(Radians<int>(90), M_PI / 2.0);
  EXPECT_FLOAT_EQ(Radians<int>(180), M_PI);
  EXPECT_FLOAT_EQ(Radians<int>(360), 2.0 * M_PI);

  EXPECT_FLOAT_EQ(Radians(0), 0.0);
  EXPECT_FLOAT_EQ(Radians(90), M_PI / 2.0);
  EXPECT_FLOAT_EQ(Radians(180), M_PI);
  EXPECT_FLOAT_EQ(Radians(360), 2.0 * M_PI);
}

TEST(UtilityFunctions, Degrees) {
  EXPECT_DOUBLE_EQ(Degrees(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Degrees(M_PI / 2.0), 90.0);
  EXPECT_DOUBLE_EQ(Degrees(M_PI), 180.0);
  EXPECT_DOUBLE_EQ(Degrees(2.0 * M_PI), 360.0);

  EXPECT_FLOAT_EQ(Degrees<int>(3), 171.88733853924697);
  EXPECT_FLOAT_EQ(Degrees<int>(4), 229.1831180523293);
  EXPECT_FLOAT_EQ(Degrees<int>(5), 286.4788975654116);
  EXPECT_FLOAT_EQ(Degrees<int>(-17), -974.0282517223995);

  EXPECT_FLOAT_EQ(Degrees(3), 171.88733853924697);
  EXPECT_FLOAT_EQ(Degrees(4), 229.1831180523293);
  EXPECT_FLOAT_EQ(Degrees(5), 286.4788975654116);
  EXPECT_FLOAT_EQ(Degrees(-17), -974.0282517223995);
}

TEST(UtilityFunctions, Sinc) {
  constexpr double kEpsAngle = std::numeric_limits<double>::epsilon();
  EXPECT_NEAR(Sinc(0.0), 1.0, 1e-15);

  EXPECT_NEAR(Sinc(0.01), std::sin(0.01) / 0.01, 1e-15);
  EXPECT_NEAR(Sinc(kEpsAngle), std::sin(kEpsAngle) / kEpsAngle, 1e-15);
  EXPECT_NEAR(Sinc(kEpsAngle * 0.5), 1.0, 1e-15);
  EXPECT_NEAR(Sinc(kEpsAngle * 1.1),
              std::sin(kEpsAngle * 1.1) / kEpsAngle / 1.1, 1e-15);

  EXPECT_NEAR(Sinc(-0.01), std::sin(0.01) / 0.01, 1e-15);
  EXPECT_NEAR(Sinc(-kEpsAngle), std::sin(kEpsAngle) / kEpsAngle, 1e-15);
  EXPECT_NEAR(Sinc(-kEpsAngle * 0.5), 1.0, 1e-15);
  EXPECT_NEAR(Sinc(-kEpsAngle * 1.1),
              std::sin(kEpsAngle * 1.1) / kEpsAngle / 1.1, 1e-15);
}

TEST(UtilityFunctions, OneMinusCosOverX) {
  constexpr double kEpsAngle = std::numeric_limits<double>::epsilon();
  EXPECT_NEAR(OneMinusCosOverX(0.0), 0.0, 1e-15);

  EXPECT_NEAR(OneMinusCosOverX(0.01), (1.0 - std::cos(0.01)) / 0.01, 1e-15);
  EXPECT_NEAR(OneMinusCosOverX(kEpsAngle),
              (1.0 - std::cos(kEpsAngle)) / kEpsAngle, 1e-15);
  EXPECT_NEAR(OneMinusCosOverX(kEpsAngle * 0.5), 0.0, 1e-15);
  EXPECT_NEAR(OneMinusCosOverX(kEpsAngle * 1.1),
              (1.0 - std::cos(kEpsAngle * 1.1)) / kEpsAngle / 1.1, 1e-15);

  EXPECT_NEAR(OneMinusCosOverX(-0.01), (1.0 - std::cos(-0.01)) / -0.01, 1e-15);
  EXPECT_NEAR(OneMinusCosOverX(-kEpsAngle),
              (1.0 - std::cos(kEpsAngle)) / kEpsAngle, 1e-15);
  EXPECT_NEAR(OneMinusCosOverX(-kEpsAngle * 0.5), 0.0, 1e-15);
  EXPECT_NEAR(OneMinusCosOverX(-kEpsAngle * 1.1),
              (1.0 - std::cos(kEpsAngle * 1.1)) / kEpsAngle / 1.1, 1e-15);
}

TEST(ApproximateExp, HandpickedValues) {
  constexpr double kEpsilon = 1e-3;
  const double kTestValues[] = {0.0,     -0.1,    -1.0,    -10.0,
                                -1024.0, -1024.1, -10000.0};

  for (const double value : kTestValues) {
    EXPECT_THAT(ApproximateExp(value), DoubleNear(std::exp(value), kEpsilon))
        << "for input " << value;
  }
}

TEST(ApproximateExp, RandomValues) {
  constexpr double kEpsilon = 1e-3;
  constexpr int kNumberOfValues = 1000;
  const auto test_values = NonPositiveSamples(kNumberOfValues);

  for (const double value : test_values) {
    const double expected = std::exp(value);
    EXPECT_THAT(ApproximateExp(value), DoubleNear(expected, kEpsilon))
        << "for input " << value;
  }
}

void BM_ApproximateExp(benchmark::State& state) {
  const auto test_values = NonPositiveSamples(kBatchSize);
  int i = 0;
  while (state.KeepRunningBatch(kBatchSize)) {
    benchmark::DoNotOptimize(ApproximateExp(test_values[i % kBatchSize]));
  }
}
BENCHMARK(BM_ApproximateExp);

void BM_StdExp(benchmark::State& state) {
  const auto test_values = NonPositiveSamples(kBatchSize);
  int i = 0;
  while (state.KeepRunningBatch(kBatchSize)) {
    benchmark::DoNotOptimize(std::exp(test_values[i % kBatchSize]));
  }
}
BENCHMARK(BM_StdExp);

//  INPUT X: -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10
//  OUTPUT:  5  2  3 4 5 2 3 4 5 2 3 4 5  2
TEST(UtilityFunctions, Wrap) {
  EXPECT_DOUBLE_EQ(Wrap(-3.0, 2.0, 6.0), 5.0);
  EXPECT_DOUBLE_EQ(Wrap(-2.0, 2.0, 6.0), 2.0);
  EXPECT_DOUBLE_EQ(Wrap(-1.0, 2.0, 6.0), 3.0);
  EXPECT_DOUBLE_EQ(Wrap(0.0, 2.0, 6.0), 4.0);
  EXPECT_DOUBLE_EQ(Wrap(1.0, 2.0, 6.0), 5.0);
  EXPECT_DOUBLE_EQ(Wrap(2.0, 2.0, 6.0), 2.0);
  EXPECT_DOUBLE_EQ(Wrap(3.0, 2.0, 6.0), 3.0);
  EXPECT_DOUBLE_EQ(Wrap(4.0, 2.0, 6.0), 4.0);
  EXPECT_DOUBLE_EQ(Wrap(5.0, 2.0, 6.0), 5.0);
  EXPECT_DOUBLE_EQ(Wrap(6.0, 2.0, 6.0), 2.0);
  EXPECT_DOUBLE_EQ(Wrap(7.0, 2.0, 6.0), 3.0);
  EXPECT_DOUBLE_EQ(Wrap(8.0, 2.0, 6.0), 4.0);
  EXPECT_DOUBLE_EQ(Wrap(9.0, 2.0, 6.0), 5.0);
  EXPECT_DOUBLE_EQ(Wrap(10.0, 2.0, 6.0), 2.0);
}

TEST(UtilityFunctions, Square) {
  EXPECT_DOUBLE_EQ(Square(-3.0), 9.0);
  EXPECT_DOUBLE_EQ(Square(4.0), 16.0);
}

TEST(UtilityFunctions, CombineIndependentProbabilities) {
  EXPECT_DOUBLE_EQ(CombineIndependentProbabilities(0.5, 0.5), 0.75);
  EXPECT_DOUBLE_EQ(CombineIndependentProbabilities(0.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(CombineIndependentProbabilities(1.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(CombineIndependentProbabilities(0.23423, 0.65534),
                   CombineIndependentProbabilities(0.65534, 0.23423));
}

TEST(UtilityFunctions, CombineDependentProbabilities) {
  EXPECT_DOUBLE_EQ(CombineDependentProbabilities(0.25, 0.5), 0.5);
  EXPECT_DOUBLE_EQ(CombineDependentProbabilities(0.5, 0.5), 0.5);
  EXPECT_DOUBLE_EQ(CombineDependentProbabilities(0.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(CombineDependentProbabilities(1.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(CombineDependentProbabilities(0.23423, 0.65534),
                   CombineDependentProbabilities(0.65534, 0.23423));
}

TEST(ComputeQuadraticRoots, TwoRealRoots1) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(2.0, 5.0, -3.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 2);
  EXPECT_DOUBLE_EQ(root1_re, -3.0);
  EXPECT_DOUBLE_EQ(root2_re, 0.5);
}

TEST(ComputeQuadraticRoots, TwoRealRoots2) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(2.0, 5.0, 3.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 2);
  EXPECT_DOUBLE_EQ(root1_re, -1.5);
  EXPECT_DOUBLE_EQ(root2_re, -1.0);
}

TEST(ComputeQuadraticRoots, ComplexRoots) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(2.0, 0.0, 3.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 0);
  EXPECT_DOUBLE_EQ(root1_re, 0.0);
  EXPECT_DOUBLE_EQ(root2_re, 0.0);
  EXPECT_DOUBLE_EQ(root_im, 1.2247448713915889);
}

TEST(ComputeQuadraticRoots, SymmetricRealRoots) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(2.0, 0.0, -3.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 2);
  EXPECT_DOUBLE_EQ(root1_re, -1.2247448713915889);
  EXPECT_DOUBLE_EQ(root2_re, 1.2247448713915889);
}

TEST(ComputeQuadraticRoots, DegenerateCase) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(0.0, 0.0, -3.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, -1);
}

TEST(ComputeQuadraticRoots, LinearEquation) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(0.0, 2.0, -3.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 1);
  EXPECT_DOUBLE_EQ(root1_re, 1.5);
  EXPECT_DOUBLE_EQ(root2_re, 1.5);
}

TEST(ComputeQuadraticRoots, NearLinearEquation) {
  const double coeff_a = -1.4210854715202e-14;
  const double coeff_b = -3.81040429245075;
  const double coeff_c = 0.9074488044955483;
  double root1_re, root2_re, root_im;
  int num_real_roots = ComputeQuadraticRoots(coeff_a, coeff_b, coeff_c,
                                             &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 2);
  EXPECT_DOUBLE_EQ(root1_re, -268133364868939.8);
  EXPECT_DOUBLE_EQ(root2_re, 0.2381502682781991);
  EXPECT_DOUBLE_EQ(coeff_a * root2_re * root2_re + coeff_b * root2_re + coeff_c,
                   0.0);
}

TEST(ComputeQuadraticRoots, RealRepeatedRoots) {
  double root1_re, root2_re, root_im;
  int num_real_roots =
      ComputeQuadraticRoots(4.0, 4.0, 1.0, &root1_re, &root2_re, &root_im);
  EXPECT_EQ(num_real_roots, 1);
  EXPECT_DOUBLE_EQ(root1_re, -0.5);
  EXPECT_DOUBLE_EQ(root2_re, -0.5);
}

}  // namespace
}  // namespace eigenmath
