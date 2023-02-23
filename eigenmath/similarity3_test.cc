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

#include "similarity3.h"

#include <cmath>
#include <functional>
#include <random>

#include "gtest/gtest.h"
#include "matchers.h"
#include "pose3_utils.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

template <typename T>
class Similarity3Test : public ::testing::Test {};
using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Similarity3Test, FloatTypes);

TYPED_TEST(Similarity3Test, ConstructorsAndAccessors) {
  using T = TypeParam;

  T eps = 1e-5;

  Similarity3<T> I;
  Vector3<T> o = Vector3<T>::Zero();
  o.setZero();

  const Pose3<T> IdentityPose;
  EXPECT_THAT(I.pose(), IsApprox(IdentityPose, eps));
  EXPECT_NEAR(I.scale(), T{1}, eps);

  T quarter_pi = M_PI_4;
  // quaternion representing 45 degree rotation around z-axis
  Quaternion<T, kDefaultOptions> q45(
      Eigen::AngleAxis<T>(quarter_pi, Vector3<T>::UnitZ()));
  // similarity representing 45 rotation around z-axis, and scale change of 2
  Similarity3<T> a_S_b(T(2), RotationZ(quarter_pi));
  EXPECT_THAT(q45.matrix(), IsApprox(a_S_b.pose().so3().matrix(), eps));
  EXPECT_NEAR(a_S_b.scale(), T(2), eps);

  Vector3<T> offset(1, 2, 3);
  // pose representing translation by offset and scale change of 0.5
  Similarity3<T> b_S_c(T(0.5), Pose3<T>(offset));

  EXPECT_THAT(offset, IsApprox(b_S_c.pose().translation(), eps));
  EXPECT_NEAR(b_S_c.scale(), T(0.5), eps);
}

TYPED_TEST(Similarity3Test, MatrixTransformTest) {
  using T = TypeParam;

  T eps = 1e-3;

  // a number of angles wrt. axis x,y,z
  T theta_x = 3;
  T theta_y = 2;
  T theta_z = 1;

  Vector3<T> translation(20, -15, 2);

  // a transform representing rotation, translation, and scale change by 1.3
  Similarity3<T> a_S_c(T(1.3), Pose3<T>(translation) * RotationZ(theta_z) *
                                   RotationY(theta_y) * RotationX(theta_x));

  Eigen::Matrix<T, 4, 4> a_M_c = a_S_c.matrix();

  // Transform a bunch of points by matrix 4x4 and Similarity3 to make sure
  // they're close to equal.
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(-100.0, 100.0);
  auto rand_gen = std::bind(distribution, generator);
  for (size_t i = 0; i < 100; ++i) {
    Vector3<T> point_c(rand_gen(), rand_gen(), rand_gen());

    // Transform point_c by matrix M_a_c.
    Vector4<T> homogeneous_point1_a = a_M_c * Unproject(point_c);
    Vector3<T> point1_a = Project(homogeneous_point1_a);

    // Transform point_c manually
    Vector3<T> point2_a =
        a_S_c.scale() * (a_S_c.pose().so3().matrix() * point_c) +
        a_S_c.pose().translation();

    // Transform point_c directly
    Vector3<T> point3_a = a_S_c * point_c;

    EXPECT_THAT(point1_a, IsApprox(point3_a, eps));
    EXPECT_THAT(point2_a, IsApprox(point3_a, eps));
  }
}

}  // namespace
}  // namespace eigenmath
