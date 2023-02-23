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

#include "pose2.h"

#include <cmath>
#include <random>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "distribution.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "sampling.h"
#include "scalar_utils.h"
#include "types.h"
#include "utils.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

constexpr float kToleranceFlt = 1e-05;
constexpr double kToleranceDbl = 1e-14;

Matrix2d CreateRotationMatrix2(double angle) {
  double cos_angle = std::cos(angle);
  double sin_angle = std::sin(angle);
  return MakeMatrix<double, 2, 2>(
      {{cos_angle, -sin_angle}, {sin_angle, cos_angle}});
}

TEST(TestPose2, Identity) {
  Pose2d p;
  EXPECT_DOUBLE_EQ(0, p.translation().x());
  EXPECT_DOUBLE_EQ(0, p.translation().y());
  EXPECT_DOUBLE_EQ(0, p.angle());
}

TEST(TestPose2, Constructor1) {
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 2);
  std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
  for (int i = 0; i < 50; i++) {
    double x = dist(generator);
    double y = dist(generator);
    double a = angle_dist(generator);
    Pose2d p{{x, y}, a};
    EXPECT_DOUBLE_EQ(x, p.translation().x());
    EXPECT_DOUBLE_EQ(y, p.translation().y());
    EXPECT_DOUBLE_EQ(a, p.angle());
  }
}

TEST(TestPose2, Constructor2) {
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 2);
  std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
  for (int i = 0; i < 50; i++) {
    Vector2d t(dist(generator), dist(generator));
    double a = angle_dist(generator);
    Pose2d p{t, a};
    EXPECT_THAT(t, IsApprox(p.translation()));
    EXPECT_DOUBLE_EQ(a, p.angle());
  }
}

TEST(TestPose2, Cast) {
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 2);
  std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
  for (int i = 0; i < 50; i++) {
    Vector2d t(dist(generator), dist(generator));
    Vector2f t_f = t.cast<float>();
    double a = angle_dist(generator);
    Pose2d p{t, a};
    Pose2f pf{t_f, static_cast<float>(a)};
    Pose2f casted_pf = p.cast<float>();
    Pose2d casted_p = casted_pf.cast<double>();

    EXPECT_THAT(pf, IsApprox(casted_pf, kToleranceFlt));
    EXPECT_THAT(p, IsApprox(casted_p, kToleranceFlt));
  }
}

TEST(TestPose2, Translation) {
  Pose2d pose;
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    Vector2d t(dist(generator), dist(generator));
    pose.translation() = t;
    EXPECT_THAT(t, IsApprox(pose.translation()));
  }
}

TEST(TestPose2, Rotation) {
  Pose2d pose;
  TestGenerator generator(kGeneratorTestSeed);
  std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double angle = angle_dist(generator);
    pose.so2() = SO2d{angle};
    EXPECT_DOUBLE_EQ(angle, pose.angle());
  }
}

TEST(TestPose2, Angle) {
  Pose2d pose;
  TestGenerator generator(kGeneratorTestSeed);
  std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double angle = angle_dist(generator);
    pose.setAngle(angle);
    EXPECT_DOUBLE_EQ(angle, pose.angle());
  }
}

TEST(TestPose2, RotationMatrix) {
  Pose2d pose;
  TestGenerator generator(kGeneratorTestSeed);
  std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    double angle = angle_dist(generator);
    Matrix2d mat = CreateRotationMatrix2(angle);
    pose.setRotationMatrix(mat);
    EXPECT_THAT(mat, IsApprox(pose.rotationMatrix(), 1e-12));
    EXPECT_NEAR(angle, pose.angle(), 1e-12);
  }
}

TEST(TestPose2, Inverse) {
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    Pose2d p{{dist(generator), dist(generator)}, dist(generator)};
    Pose2d p_inv = p.inverse();
    Pose2d actual_left = p_inv * p;
    EXPECT_NEAR(0, actual_left.translation().x(), kToleranceDbl);
    EXPECT_NEAR(0, actual_left.translation().y(), kToleranceDbl);
    EXPECT_NEAR(0, actual_left.angle(), kToleranceDbl);
    Pose2d actual_right = p * p_inv;
    EXPECT_NEAR(0, actual_right.translation().x(), kToleranceDbl);
    EXPECT_NEAR(0, actual_right.translation().y(), kToleranceDbl);
    EXPECT_NEAR(0, actual_right.angle(), kToleranceDbl);
  }
}

TEST(TestPose2, Matrix) {
  TestGenerator generator(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 2);
  for (int i = 0; i < 50; i++) {
    Pose2d pose{{dist(generator), dist(generator)}, dist(generator)};
    Vector2d point{dist(generator), dist(generator)};
    Vector2d expected_point = pose.translation() + pose.so2() * point;
    Vector3d point3 = pose.matrix() * Vector3d{point.x(), point.y(), 1};
    Vector2d actual_point{point3.x(), point3.y()};
    EXPECT_THAT(expected_point, IsApprox(actual_point, kToleranceDbl));
  }
}

TEST(TestPose2, MultiplyPose) {
  double data[20][9] = {{-2.2855, -3.14945, 1.51059, 1.47608, -1.46696, 2.54123,
                         -0.732387, -1.7643, -2.23136},
                        {-0.120255, -4.67744, -0.599885, 0.37778, 1.52151,
                         1.97756, 1.05053, -3.63485, 1.37767},
                        {-0.00196565, -3.48862, -4.31458, 3.54334, 1.60869,
                         0.509264, -2.85772, -0.845172, 2.47787},
                        {0.901791, 0.943961, -5.46489, 5.84951, 2.91729,
                         -1.34155, 2.77016, 7.20787, -0.523255},
                        {0.64189, 2.73509, 0.671552, 5.43937, 4.87912, 4.34307,
                         1.86434, 9.93912, -1.26856},
                        {-2.08011, -1.40463, -0.208562, -3.84461, 2.68256,
                         -2.70254, -5.28598, 2.01583, -2.9111},
                        {-0.211354, -1.12717, 2.93002, 0.0963011, 6.2845,
                         0.469171, -1.62523, -7.25132, -2.88399},
                        {3.7336, -2.84521, -6.21748, 3.20302, -0.73306,
                         -2.37077, 6.97785, -3.36638, -2.30507},
                        {2.2359, 1.25588, 0.144626, 5.88102, 1.94026, -1.05329,
                         7.77589, 4.02347, -0.908668},
                        {-2.25556, 2.91937, 0.542332, -0.169792, 1.15592,
                         6.4158, -2.9976, 3.82178, 0.674948},
                        {-4.37265, 3.02192, -0.337689, -3.56523, 6.12383,
                         7.60334, -5.70765, 9.98109, 0.98247},
                        {3.11297, -1.76622, -1.3109, 4.97816, -8.02184, 3.49841,
                         -3.36019, -8.63863, 2.1875},
                        {-4.30496, -0.59307, 0.941545, -0.0176346, -5.03622,
                         1.66765, -0.243719, -3.57134, 2.60919},
                        {-2.96792, -1.50972, 2.6497, 5.23405, 7.07505, -2.41477,
                         -10.9229, -5.27391, 0.234924},
                        {-1.60613, -1.10775, -3.81966, 3.28953, -2.43657,
                         -1.4705, -2.63952, 2.85331, 0.993024},
                        {1.14728, -3.04044, 1.18721, -1.90162, 1.85385,
                         -0.95853, -1.28353, -4.11007, 0.228682},
                        {-1.83665, -2.58921, -1.76016, -2.73072, -2.75763,
                         -2.86235, -4.03098, 0.611769, 1.66068},
                        {-0.346852, 0.534982, -0.898792, 1.63275, 1.84978,
                         -3.82494, 2.11722, 0.40883, 1.55945},
                        {-2.32097, -5.3066, 4.37825, -3.96174, 4.04929, 2.63828,
                         2.80365, -2.89199, 0.733336},
                        {1.72049, -1.53052, 3.33529, -0.983484, -2.33268,
                         0.638919, 2.23656, 0.947844, -2.30897}};
  for (const auto& sample : data) {
    Pose2d a{{sample[0], sample[1]}, sample[2]};
    Pose2d b{{sample[3], sample[4]}, sample[5]};
    Pose2d expected_pose{{sample[6], sample[7]}, sample[8]};
    Pose2d actual_pose = a * b;
    EXPECT_THAT(expected_pose.translation(),
                IsApprox(actual_pose.translation(), 5e-5));
    EXPECT_NEAR(expected_pose.angle(), actual_pose.angle(), 5e-5);
    Pose2d actual_ps_pose = a * b.so2();
    EXPECT_THAT(a.translation(), IsApprox(actual_ps_pose.translation(), 5e-5));
    EXPECT_NEAR(expected_pose.angle(), actual_ps_pose.angle(), 5e-5);
    Pose2d actual_sp_pose = a.so2() * b;
    EXPECT_THAT(a.so2() * b.translation(),
                IsApprox(actual_sp_pose.translation(), 5e-5));
    EXPECT_NEAR(expected_pose.angle(), actual_sp_pose.angle(), 5e-5);
  }
}

TEST(TestPose2, MultiplyPoseZero) {
  Pose2d a{
      {-3.50421039696121030542646547500940101826927275396883487701416015625e-06,
       0.0016668694441564417367540595904529254767112433910369873046875},
      SO2d{0.9999999999984658938245729586924426257610321044921875,
           1.75173794216727999445946950540786701822071336209774017333984375e-06,
           false}};
  Pose2d b{{0, 0}, SO2d{1, -0, false}};
  Pose2d c = a * b;
  EXPECT_THAT(c, IsApprox(a));
}

TEST(TestPose2, MultiplyVector) {
  double data[20][9] = {
      {7.98879, 0.164215, -6.26819, 3.57173, -1.10212, 11.5766, -0.884238},
      {1.63254, 0.976215, 2.51322, -1.31038, 2.40175, 1.2808, -1.73704},
      {1.94556, -2.95001, -3.30902, -1.04615, 2.32077, 2.59033, -5.41267},
      {2.12215, -3.42273, 0.582108, 0.194614, 2.18859, 1.08146, -1.4876},
      {-0.396187, 2.44764, 0.883045, -1.60686, -1.43863, -0.304628, 0.292813},
      {-0.0730169, -3.73681, 2.36387, -2.1631, 3.88262, -1.25603, -8.02099},
      {-2.93613, -1.22675, -1.69676, 1.56388, 2.02361, -1.12502, -3.03247},
      {3.62564, -1.51122, -1.22899, 2.44517, -0.191787, 4.26454, -3.87923},
      {1.02649, -3.40588, 8.13483, 6.85545, 1.4911, -2.3063, 2.76771},
      {0.313916, -1.68565, -2.66805, -1.16617, -3.29884, -0.152634, 1.782},
      {0.191629, -2.79375, 2.79636, -3.03855, 1.66358, 2.48792, -5.38746},
      {-1.06814, 3.49853, 3.34436, 2.34521, 5.33393, -2.29117, -2.1984},
      {0.521832, -2.5177, -1.70839, -2.1986, 4.31494, 5.09755, -0.931736},
      {-1.74615, -1.37773, -0.705774, 2.38249, -0.555912, -0.293388, -3.34618},
      {-0.443955, 0.200037, 0.00976551, 3.29472, 5.06621, 2.80114, 5.29818},
      {2.89938, -1.56476, -1.52032, 0.28903, 3.21549, 6.12536, -1.69117},
      {-2.70692, -0.783412, 6.42111, -1.85979, -6.22783, -3.69279, -7.20779},
      {2.70351, -1.68712, 1.16892, 1.16284, 0.21916, 2.95665, -0.531198},
      {-0.572838, -2.53295, -1.32621, -1.32598, -0.280838, -1.16641, -1.31444},
      {-0.0774288, 2.14968, -4.47031, 0.728472, 4.29035, -4.41731, 1.82842}};
  for (const auto& sample : data) {
    Pose2d pose{{sample[0], sample[1]}, sample[2]};
    Vector2d point{sample[3], sample[4]};
    Vector2d expected_point{sample[5], sample[6]};
    Vector2d actual_point = pose * point;
    EXPECT_THAT(expected_point, IsApprox(actual_point, 5e-5));
  }
}

}  // namespace
}  // namespace eigenmath
