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

#include "so2.h"

#include <cmath>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/random/distributions.h"
#include "benchmark/benchmark.h"
#include "distribution.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "pose3_utils.h"
#include "sampling.h"
#include "utils.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

Matrix2d CreateRotationMatrix2(double angle) {
  double cos_angle = std::cos(angle);
  double sin_angle = std::sin(angle);
  return MakeMatrix<double, 2, 2>(
      {{cos_angle, -sin_angle}, {sin_angle, cos_angle}});
}

TEST(TestSO2, ConstructorIdentity) {
  {
    SO2d p;
    EXPECT_DOUBLE_EQ(1, p.cos_angle());
    EXPECT_DOUBLE_EQ(0, p.sin_angle());
  }
  {
    SO2d p{-0};
    EXPECT_DOUBLE_EQ(1, p.cos_angle());
    EXPECT_DOUBLE_EQ(0, p.sin_angle());
  }
}

TEST(SO2DeathTest, InvalidInitializationPanics) {
  EXPECT_DEATH(SO2d(0.0, 0.0), "");
}

TEST(TestSO2, MultiplyIsNormalized) {
  SO2d a{0.999999, -0.00158382};
  SO2d b{0.999997, 0.00247471};
  SO2d c = a * b;
  EXPECT_DOUBLE_EQ(1, c.coeffs().norm());
}

TEST(TestSO2, MultiplyImprovesNormalization) {
  const SO2d a;
  const_cast<Vector2<double>&>(a.coeffs())(0) = 0.99;
  const_cast<Vector2<double>&>(a.coeffs())(1) = -0.002;
  EXPECT_DOUBLE_EQ(0.99, a.coeffs()(0));
  EXPECT_DOUBLE_EQ(-0.002, a.coeffs()(1));
  EXPECT_GT(std::abs(1.0 - a.coeffs().norm()), 0.0);
  const SO2d b(0.98, 0.001);
  const SO2d c = a * b;
  const Vector2d d{
      a.cos_angle() * b.cos_angle() - a.sin_angle() * b.sin_angle(),
      b.cos_angle() * a.sin_angle() + a.cos_angle() * b.sin_angle()};
  EXPECT_LT(std::abs(1.0 - c.coeffs().norm()), std::abs(1.0 - d.norm()));
}

TEST(SO2DeathTest, InvalidScalarInitialization) {
  const double an_inf = 1.0 / 0.0;
  const double a_nan = 0.0 / 0.0;

  SO2d tmp;
  EXPECT_DEATH(tmp = SO2d(an_inf), "");
  EXPECT_DEATH(tmp = SO2d(a_nan), "");
}

TEST(TestSO2, ScalarInitialization) {
  const double a_subnormal = DBL_MIN / 2.0;
  const double a_zero = -0.0;
  const double a_normal = 1.0;

  SO2d tmp;
  tmp = SO2d(a_subnormal);
  tmp = SO2d(a_zero);
  tmp = SO2d(a_normal);
}

TEST(SO2DeathTest, MatrixInitialization) {
  const double an_inf = 1.0 / 0.0;
  const double a_nan = 0.0 / 0.0;

  SO2d tmp;
  Matrix2d mx = Matrix2d::Zero();
  mx << an_inf, 1.0, 2.0, 3.0;
  EXPECT_DEATH(tmp = SO2d(mx), "");
  mx << a_nan, 1.0, 2.0, 3.0;
  EXPECT_DEATH(tmp = SO2d(mx), "");
}

TEST(TestSO2, MatrixInitialization) {
  const double a_subnormal = DBL_MIN / 2.0;
  const double a_zero = -0.0;
  const double a_normal = 1.0;

  SO2d tmp;
  Matrix2d mx = Matrix2d::Zero();
  mx << a_subnormal, 1.0, 2.0, 3.0;
  { tmp = SO2d(mx); }
  mx << a_zero, 1.0, 2.0, 3.0;
  { tmp = SO2d(mx); }
  mx << a_normal, 1.0, 2.0, 3.0;
  { tmp = SO2d(mx); }
}

TEST(SO2DeathTest, PairInitialization) {
  const double an_inf = 1.0 / 0.0;
  const double a_nan = 0.0 / 0.0;

  SO2d tmp;
  for (bool do_normalize = false; !do_normalize; do_normalize = true) {
    // check cos_angle
    EXPECT_DEATH(tmp = SO2d(an_inf, 0.0, do_normalize), "");
    EXPECT_DEATH(tmp = SO2d(a_nan, 0.0, do_normalize), "");
    // check sin_angle
    EXPECT_DEATH(tmp = SO2d(0.0, an_inf, do_normalize), "");
    EXPECT_DEATH(tmp = SO2d(0.0, a_nan, do_normalize), "");
  }
}

TEST(TestSO2, PairInitialization) {
  const double a_subnormal = DBL_MIN / 2.0;
  const double a_zero = -0.0;
  const double a_normal = 1.0;

  SO2d tmp;
  for (bool do_normalize = false; !do_normalize; do_normalize = true) {
    // check cos_angle
    { tmp = SO2d(a_subnormal, 1.0, do_normalize); }
    { tmp = SO2d(a_zero, 1.0, do_normalize); }
    { tmp = SO2d(a_normal, 0.0, do_normalize); }
    // check sin_angle
    { tmp = SO2d(1.0, a_subnormal, do_normalize); }
    { tmp = SO2d(1.0, a_zero, do_normalize); }
    { tmp = SO2d(0.0, a_normal, do_normalize); }
  }
}

TEST(TestSO2, ConstructorAngle) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double a = dist(rnd_engine);
    SO2d q{a};
    EXPECT_DOUBLE_EQ(std::cos(a), q.cos_angle());
    EXPECT_DOUBLE_EQ(std::sin(a), q.sin_angle());
  }
}

TEST(TestSO2, ConstructorMatrix) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    Matrix2d m = CreateRotationMatrix2(dist(rnd_engine));
    SO2d q{m};
    EXPECT_THAT(m, IsApprox(q.matrix(), 1e-12));
  }
}

TEST(TestPose2, Cast) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double a = dist(rnd_engine);
    SO2d p{a};
    SO2f pf{static_cast<float>(a)};
    SO2f casted_pf = p.cast<float>();
    SO2d casted_p = casted_pf.cast<double>();

    EXPECT_NEAR(pf.angle(), casted_pf.angle(), 1e-5);
    EXPECT_NEAR(p.angle(), casted_p.angle(), 1e-5);
  }
}

TEST(TestSO2, ConstructorMatrixNonUnit) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    Matrix2d m = CreateRotationMatrix2(dist(rnd_engine));
    double r = std::abs(dist(rnd_engine));
    Matrix2d ms = m * MakeMatrix<double, 2, 2>({{r, 0}, {0, r}});
    SO2d q{ms};
    EXPECT_THAT(m, IsApprox(q.matrix(), 1e-12));
  }
}

TEST(TestSO2, ConstructorCosSin) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double a = dist(rnd_engine);
    SO2d q{std::cos(a), std::sin(a)};
    EXPECT_DOUBLE_EQ(std::cos(a), q.cos_angle());
    EXPECT_DOUBLE_EQ(std::sin(a), q.sin_angle());
  }
}

TEST(TestSO2, ConstructorCosSinNonUnit) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double a = dist(rnd_engine);
    double r = std::abs(dist(rnd_engine));
    SO2d q{r * std::cos(a), r * std::sin(a)};
    EXPECT_DOUBLE_EQ(std::cos(a), q.cos_angle());
    EXPECT_DOUBLE_EQ(std::sin(a), q.sin_angle());
  }
}

TEST(TestSO2, Angle) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double a = dist(rnd_engine);
    double expected_angle = WrapAngle(a);
    SO2d q{a};
    EXPECT_DOUBLE_EQ(expected_angle, q.angle());
  }
}

TEST(TestSO2, AngleNorm) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double a = dist(rnd_engine);
    double expected_angle = WrapAngle(a);
    SO2d q{a};
    EXPECT_DOUBLE_EQ(std::abs(expected_angle), q.norm());
  }
}

TEST(TestSO2, AngleSpecial) {
  constexpr double pi = M_PI;
  EXPECT_DOUBLE_EQ(-pi, SO2d{-pi}.angle());
  EXPECT_DOUBLE_EQ(pi, SO2d{pi}.angle());
  EXPECT_DOUBLE_EQ(0, SO2d{0.0}.angle());
  EXPECT_NEAR(0, SO2d{2.0 * pi}.angle(), 1e-15);
  EXPECT_NEAR(0, SO2d{-2.0 * pi}.angle(), 1e-15);
  EXPECT_DOUBLE_EQ(pi, SO2d{3.0 * pi}.angle());
  EXPECT_DOUBLE_EQ(-pi, SO2d{-3.0 * pi}.angle());
}

TEST(TestSO2, DeltaAngle) {
  constexpr double pi = M_PI;
  EXPECT_NEAR(DeltaAngle(SO2d{-pi}, SO2d{-pi}), 0.0, 1e-15);
  EXPECT_NEAR(DeltaAngle(SO2d{pi}, SO2d{pi}), 0.0, 1e-15);
  EXPECT_NEAR(DeltaAngle(SO2d{-pi}, SO2d{pi}), 0.0, 1e-15);
  EXPECT_NEAR(DeltaAngle(SO2d{0.0}, SO2d{pi}), pi, 1e-15);
  EXPECT_NEAR(DeltaAngle(SO2d{pi}, SO2d{0.0}), -pi, 1e-15);
  EXPECT_NEAR(DeltaAngle(SO2d{0.5 * pi}, SO2d{0.0}), -0.5 * pi, 1e-15);
  EXPECT_NEAR(DeltaAngle(SO2d{0.0}, SO2d{0.5 * pi}), 0.5 * pi, 1e-15);
}

TEST(TestSO2, IsInInterval) {
  constexpr double pi = M_PI;
  EXPECT_TRUE(IsInInterval(SO2d(-3.0 * pi / 4.0), 0.0, 3.0 * pi / 2.0));
  EXPECT_FALSE(IsInInterval(SO2d(-pi / 4.0), 0.0, 3.0 * pi / 2.0));
  EXPECT_TRUE(IsInInterval(SO2d(3.0 * pi / 4.0), -3.0 * pi / 2.0, 0.0));
  EXPECT_FALSE(IsInInterval(SO2d(pi / 4.0), -3.0 * pi / 2.0, 0.0));
  EXPECT_TRUE(IsInInterval(SO2d(pi / 2.0), pi / 2.0 - 1e-12, pi / 2.0 + 1e-12));
  EXPECT_TRUE(
      IsInInterval(SO2d(-3.0 * pi / 2.0), pi / 2.0 - 1e-12, pi / 2.0 + 1e-12));
}

TEST(TestSO2, Inverse) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    SO2d q{dist(rnd_engine)};
    SO2d q_inv = q.inverse();
    EXPECT_DOUBLE_EQ(+q.cos_angle(), q_inv.cos_angle());
    EXPECT_DOUBLE_EQ(-q.sin_angle(), q_inv.sin_angle());
  }
}

TEST(TestSO2, MultiplyPoses) {
  double data[20][3] = {
      {2.13539, -0.428563, 1.70683},   {3.04255, 4.13028, 0.889645},
      {1.30826, 0.687653, 1.99591},    {1.20441, -2.53421, -1.3298},
      {-2.28639, -6.93576, -2.93896},  {2.18219, -1.1469, 1.03529},
      {2.84702, 1.07081, -2.36535},    {-2.63269, -3.9756, -0.325105},
      {-4.24956, 0.83103, 2.86465},    {0.185368, -3.22869, -3.04332},
      {-4.46822, 4.089, -0.379217},    {0.0201857, 1.01151, 1.0317},
      {-1.25291, -2.62507, 2.4052},    {0.0527632, -1.90983, -1.85707},
      {2.33812, 0.446237, 2.78436},    {2.49567, -0.438392, 2.05728},
      {-3.64461, 1.9442, -1.70041},    {-0.959654, -3.77532, 1.54821},
      {-0.241879, -2.04588, -2.28776}, {-1.33165, -1.29835, -2.62999}};
  for (int i = 0; i < 20; i++) {
    SO2d a{data[i][0]};
    SO2d b{data[i][1]};
    SO2d expected_pose{data[i][2]};
    SO2d actual_pose = a * b;
    EXPECT_NEAR(expected_pose.angle(), actual_pose.angle(), 3e-5);
  }
}

TEST(TestSO2, MultiplyPoints) {
  double data[20][5] = {{1.22594, -2.87572, -2.59696, 1.47191, -3.58433},
                        {-0.745039, 0.0362267, -3.37071, -2.25872, -2.50224},
                        {0.160232, 3.45996, -7.75454, 4.65285, -7.10318},
                        {-1.3295, -2.2177, 3.51488, 2.88309, 2.99339},
                        {-3.16244, 0.984771, -2.39468, -0.934628, 2.41469},
                        {1.38914, -0.426994, 6.10228, -6.07901, 0.682473},
                        {5.00931, -0.742274, 0.14119, -0.082159, 0.751103},
                        {1.62072, 4.66891, 1.46557, -1.69673, 4.58995},
                        {-3.01335, 2.04642, -0.440395, -2.08594, 0.175063},
                        {-4.05833, -1.511, -1.58394, 2.17636, -0.235467},
                        {-1.83192, 3.86255, -1.0506, -2.01217, -3.46038},
                        {0.319381, 3.36367, -0.822758, 3.4519, 0.274972},
                        {-0.576721, 1.53967, -4.83553, -1.34607, -4.89295},
                        {0.686511, -3.26331, 1.24835, -3.31531, -1.10287},
                        {4.37156, 3.47344, -0.291395, -1.43568, -3.17624},
                        {5.18724, -2.00771, 2.05536, 0.910023, 2.7253},
                        {3.89153, -0.0666037, -5.25921, -3.53588, 3.89374},
                        {-2.01483, 2.28455, -0.838613, -1.7387, -1.70275},
                        {-5.34752, -4.47043, 0.536319, -3.08398, -3.28047},
                        {5.69115, -0.0547007, -6.17186, -3.4896, -5.09093}};
  for (int i = 0; i < 20; i++) {
    SO2d pose{data[i][0]};
    Vector2d point{data[i][1], data[i][2]};
    Vector2d expected_point{data[i][3], data[i][4]};
    Vector2d actual_point = pose * point;
    EXPECT_THAT(expected_point, IsApprox(actual_point, 3e-5));
  }
}

TEST(QuaternionZ, CompareWithAngleAxis) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  for (int i = 0; i < 50; i++) {
    const double sampled_angle = absl::Uniform(rnd_engine, -M_PI, M_PI);
    SO2d so2{sampled_angle};
    Quaterniond expected(Eigen::AngleAxisd(sampled_angle, Vector3d{0, 0, 1}));
    EXPECT_THAT(QuaternionZ(so2), IsApprox(expected, 1e-12));
  }
}

TEST(QuaternionZ, CompareWithRotationZ) {
  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  for (int i = 0; i < 50; i++) {
    const double sampled_angle = absl::Uniform(rnd_engine, -M_PI, M_PI);
    SO2d so2{sampled_angle};
    Quaterniond expected = RotationZ(sampled_angle).quaternion();
    EXPECT_THAT(QuaternionZ(so2), IsApprox(expected, 1e-12));
  }
}

void BM_ConstructionFromAngle(benchmark::State& state) {
  constexpr int kSampleCount = 1000;
  std::vector<double> angles(kSampleCount);
  TestGenerator generator(kGeneratorTestSeed);
  for (double& angle : angles) {
    angle = absl::Uniform(generator, -M_PI, M_PI);
  }
  int count = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(SO2d(angles[count]));
    if (++count == kSampleCount) {
      count = 0;
    }
  }
}
BENCHMARK(BM_ConstructionFromAngle);

void BM_CalculateAngle(benchmark::State& state) {
  constexpr int kSampleCount = 1000;
  std::vector<eigenmath::SO2d> rotations;
  rotations.reserve(kSampleCount);
  TestGenerator generator(kGeneratorTestSeed);
  eigenmath::UniformDistributionSO2d rotation_dist;
  for (int i = 0; i < kSampleCount; ++i) {
    rotations.emplace_back(rotation_dist(generator));
  }
  int count = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize((rotations[count].angle()));
    if (++count == kSampleCount) {
      count = 0;
    }
  }
}
BENCHMARK(BM_CalculateAngle);

void BM_QuaternionZ(benchmark::State& state) {
  constexpr int kSampleCount = 1000;
  std::vector<eigenmath::SO2d> rotations;
  rotations.reserve(kSampleCount);
  TestGenerator generator(kGeneratorTestSeed);
  eigenmath::UniformDistributionSO2d rotation_dist;
  for (int i = 0; i < kSampleCount; ++i) {
    rotations.emplace_back(rotation_dist(generator));
  }
  int count = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(QuaternionZ(rotations[count]));
    if (++count == kSampleCount) {
      count = 0;
    }
  }
}
BENCHMARK(BM_QuaternionZ);

void BM_SO2DerivativeTheta(benchmark::State& state) {
  constexpr int kSampleCount = 1000;
  std::vector<double> angles(kSampleCount);
  TestGenerator generator(kGeneratorTestSeed);
  for (double& angle : angles) {
    angle = absl::Uniform(generator, -M_PI, M_PI);
  }
  int count = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(SO2DerivativeTheta(angles[count]));
    if (++count == kSampleCount) {
      count = 0;
    }
  }
}
BENCHMARK(BM_SO2DerivativeTheta);

}  // namespace
}  // namespace eigenmath
