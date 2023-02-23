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

#include "pose3.h"

#include <cmath>
#include <functional>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "pose3_utils.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

template <class Scalar>
class Pose3Test : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Pose3Test, ScalarTypes);

TYPED_TEST(Pose3Test, ConstructIdentity) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  Pose3<Scalar> identity;

  EXPECT_THAT(
      identity.quaternion().matrix().eval(),
      IsApprox(Quaternion<Scalar>(1, 0, 0, 0).matrix().eval(), kTolerance));
  EXPECT_THAT(identity.translation(),
              IsApprox(Vector3<Scalar>::Zero(), kTolerance));
}

TYPED_TEST(Pose3Test, ConstructPureRotation) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  const Scalar quarter_pi = M_PI_4;
  const Quaternion<Scalar> quaternion(
      Eigen::AngleAxis<Scalar>(quarter_pi, Vector3<Scalar>::UnitZ()));
  const Pose3<Scalar> pose = RotationZ(quarter_pi);
  EXPECT_THAT(pose.quaternion().matrix().eval(),
              IsApprox(quaternion.matrix().eval(), kTolerance));
  EXPECT_THAT(pose.translation(),
              IsApprox(Vector3<Scalar>::Zero(), kTolerance));
}

TYPED_TEST(Pose3Test, ConstructPureTranslation) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  const Vector3<Scalar> offset(1, 2, 3);
  const Pose3<Scalar> pose(offset);
  EXPECT_THAT(
      pose.quaternion().matrix(),
      IsApprox(Quaternion<Scalar>::Identity().matrix().eval(), kTolerance));
  EXPECT_THAT(pose.translation(), IsApprox(offset, kTolerance));
}

TYPED_TEST(Pose3Test, ConstructNonTrivialPose) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  const Scalar quarter_pi = M_PI_4;
  const Quaternion<Scalar> quaternion(
      Eigen::AngleAxis<Scalar>(quarter_pi, Vector3<Scalar>::UnitZ()));
  const Vector3<Scalar> offset(1, 2, 3);

  const Pose3<Scalar> pose(quaternion, offset);
  EXPECT_THAT(
      Matrix3<Scalar>(quaternion.matrix()),
      IsApprox(Matrix3<Scalar>(pose.quaternion().matrix()), kTolerance));
  EXPECT_THAT(offset, IsApprox(pose.translation(), kTolerance));
}

TYPED_TEST(Pose3Test, ConstructNonTrivialPoseFromMatrix) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  const Scalar quarter_pi = M_PI_4;
  const Quaternion<Scalar> quaternion(
      Eigen::AngleAxis<Scalar>(quarter_pi, Vector3<Scalar>::UnitZ()));
  const Vector3<Scalar> offset(1, 2, 3);
  Matrix<Scalar, 4, 4> matrix = Matrix<Scalar, 4, 4>::Zero();
  matrix.setIdentity();
  matrix.col(0).template head<2>() = Vector2<Scalar>(M_SQRT1_2, M_SQRT1_2);
  matrix.col(1).template head<2>() = Vector2<Scalar>(-M_SQRT1_2, M_SQRT1_2);
  matrix.col(3).template head<3>() = offset;
  const Pose3<Scalar> pose_from_matrix(matrix);
  EXPECT_THAT(pose_from_matrix.quaternion().matrix().eval(),
              IsApprox(quaternion.matrix().eval(), kTolerance));
  EXPECT_THAT(pose_from_matrix.translation(), IsApprox(offset, kTolerance));
  EXPECT_THAT(pose_from_matrix.matrix(), IsApprox(matrix, kTolerance));
}

TYPED_TEST(Pose3Test, Composition) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  // a point in frame a
  Vector3<Scalar> point_a(1, 2, 3);
  // this pose represents only a translation
  Pose3<Scalar> a_pose_b(point_a);
  // this pose represents only a rotation
  Scalar quarter_pi = M_PI_4;
  Pose3<Scalar> b_pose_c = RotationZ(quarter_pi);
  // this pose represents a rotation and translation
  Pose3<Scalar> a_pose_c = a_pose_b * b_pose_c;
  EXPECT_THAT(
      Matrix3<Scalar>(b_pose_c.quaternion().matrix()),
      IsApprox(Matrix3<Scalar>(a_pose_c.quaternion().matrix()), kTolerance));
  EXPECT_THAT(point_a, IsApprox(a_pose_c.translation(), kTolerance));

  Pose3<Scalar> inplacea_pose_c = (a_pose_b *= b_pose_c);
  EXPECT_THAT(inplacea_pose_c, IsApprox(a_pose_c));

  Pose3<Scalar> c_pose_a = a_pose_c.inverse();
  Pose3<Scalar> c_pose_c = c_pose_a * a_pose_c;
  Pose3<Scalar> a_pose_a = a_pose_c * c_pose_a;

  EXPECT_THAT(a_pose_a, IsApprox(c_pose_c, kTolerance));
  Pose3<Scalar> identity_pose;
  EXPECT_THAT(a_pose_a, IsApprox(identity_pose, kTolerance));
}

TYPED_TEST(Pose3Test, MatrixTransform) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  // a number of angles wrt. axis x,y,z
  Scalar theta_x = 3;
  Scalar theta_y = 2;
  Scalar theta_z = 1;

  // a point in frame a
  Vector3<Scalar> point_a(20, -15, 2);

  // a pose only representing translation
  Pose3<Scalar> a_pose_b(point_a);
  // a pose only representing rotation
  Pose3<Scalar> b_pose_c =
      RotationZ(theta_z) * RotationY(theta_y) * RotationX(theta_x);
  // a pose representing rotation & translation
  Pose3<Scalar> a_pose_c = a_pose_b * b_pose_c;
  Matrix<Scalar, 4, 4> M_a_c = a_pose_c.matrix();
  Matrix<Scalar, 4, 4> M_b_c = b_pose_c.matrix();

  // Transform a bunch of points by matrix 4x4 and Pose3 to make sure they're
  // close to equal.
  std::default_random_engine generator(54321);
  std::uniform_real_distribution<Scalar> distribution(-100.0, 100.0);
  auto rand_gen = std::bind(distribution, generator);
  for (size_t i = 0; i < 100; ++i) {
    Vector3<Scalar> point_c(rand_gen(), rand_gen(), rand_gen());

    // Transform point_c by matrix M_a_c.
    Vector4<Scalar> homogeneous_point1_a = M_a_c * Unproject(point_c);
    Vector3<Scalar> point1_a = Project(homogeneous_point1_a);

    // Transform point_c by matrix M_a_b.
    Vector4<Scalar> homogeneous_point1_b = M_b_c * Unproject(point_c);
    Vector3<Scalar> point1_b = Project(homogeneous_point1_b);

    // Transform point_c just by the rotation of a_pose_c
    Vector3<Scalar> point2_b = a_pose_c.quaternion() * point_c;
    Vector3<Scalar> point3_b = a_pose_c.rotationMatrix() * point_c;

    EXPECT_THAT(point1_b, IsApprox(point2_b, kTolerance));
    EXPECT_THAT(point2_b, IsApprox(point3_b, kTolerance));

    // Transform point_c by pose a_pose_c
    Vector3<Scalar> point2_a = a_pose_c * point_c;

    EXPECT_THAT(point1_a, IsApprox(point2_a, kTolerance));

    // Transform by a_pose_b, which is just a translation
    Vector3<Scalar> point3_a = a_pose_b * point2_b;
    Vector3<Scalar> point4_a = point_a + point2_b;

    EXPECT_THAT(point2_a, IsApprox(point3_a, kTolerance));
    EXPECT_THAT(point3_a, IsApprox(point4_a, kTolerance));
  }
}

TYPED_TEST(Pose3Test, CanonicalRotations) {
  using T = TypeParam;
  using ToScalarFnPtr = T (*)(const Pose3<T>&);

  struct Fn {
    std::function<Pose3<T>(T)> rotation_fn;
    std::function<T(Pose3<T>)> rotation_angle_fn;
  };

  std::vector<Fn> fns = {{RotationX<T>, ToScalarFnPtr(ToRotationAngleX<T>)},
                         {RotationY<T>, ToScalarFnPtr(ToRotationAngleY<T>)},
                         {RotationZ<T>, ToScalarFnPtr(ToRotationAngleZ<T>)}};
  std::default_random_engine generator(54321);
  std::uniform_real_distribution<double> uniform(0., 1.);
  auto rand_uniform = std::bind(uniform, generator);

  for (const auto& fn : fns) {
    for (int i = 0; i < 100; ++i) {
      T w_angle = M_PI * rand_uniform() - 0.5 * M_PI;
      Pose3<T> Rw = fn.rotation_fn(w_angle);
      T w = fn.rotation_angle_fn(Rw);
      Pose3<T> Rw2 = fn.rotation_fn(w);
      ASSERT_TRUE(Rw.isApprox(Rw2));
    }
  }
}

TYPED_TEST(Pose3Test, ToFromPose2) {
  using T = TypeParam;

  struct Fn {
    std::function<Pose3<T>(const Pose2<T>&)> from_pose2;
    std::function<Pose2<T>(const Pose3<T>&)> to_pose2;
    int idx0;
    int idx1;
    int other_idx;
  };

  std::vector<Fn> fns = {{FromPose2XY<T>, ToPose2XY<T>, 0, 1, 2},
                         {FromPose2ZX<T>, ToPose2ZX<T>, 2, 0, 1},
                         {FromPose2YZ<T>, ToPose2YZ<T>, 1, 2, 0}};

  std::default_random_engine rnd_engine(10142);  // Fixed seed.
  std::normal_distribution<T> dist(T(0), T(2));
  const T eps = 1e-5;
  for (const auto& fn : fns) {
    for (int i = 0; i < 50; i++) {
      T x = dist(rnd_engine);
      T y = dist(rnd_engine);
      T a = dist(rnd_engine);
      Pose2<T> a_pose_b{{x, y}, a};
      Pose3<T> a_uv_pose_b_uv = fn.from_pose2(a_pose_b);
      EXPECT_EQ(a_pose_b.translation().x(),
                a_uv_pose_b_uv.translation()[fn.idx0]);
      EXPECT_EQ(a_pose_b.translation().y(),
                a_uv_pose_b_uv.translation()[fn.idx1]);
      EXPECT_EQ(T(0), a_uv_pose_b_uv.translation()[fn.other_idx]);

      Matrix2<T> a_R_b = a_pose_b.rotationMatrix();
      Matrix3<T> a_uv_R_b_uv = a_uv_pose_b_uv.rotationMatrix();
      EXPECT_NEAR(a_R_b(0, 0), a_uv_R_b_uv(fn.idx0, fn.idx0), eps);
      EXPECT_NEAR(a_R_b(0, 1), a_uv_R_b_uv(fn.idx0, fn.idx1), eps);
      EXPECT_NEAR(a_R_b(1, 0), a_uv_R_b_uv(fn.idx1, fn.idx0), eps);
      EXPECT_NEAR(a_R_b(1, 1), a_uv_R_b_uv(fn.idx1, fn.idx1), eps);
      EXPECT_NEAR(T(0), a_uv_R_b_uv(fn.other_idx, fn.idx0), eps);
      EXPECT_NEAR(T(0), a_uv_R_b_uv(fn.other_idx, fn.idx1), eps);
      EXPECT_NEAR(T(0), a_uv_R_b_uv(fn.idx0, fn.other_idx), eps);
      EXPECT_NEAR(T(0), a_uv_R_b_uv(fn.idx1, fn.other_idx), eps);
      EXPECT_NEAR(T(1), a_uv_R_b_uv(fn.other_idx, fn.other_idx), eps);

      Pose2<T> a_pose_b2 = fn.to_pose2(a_uv_pose_b_uv);
      EXPECT_THAT(a_pose_b, IsApprox(a_pose_b2, eps));
    }
  }
}

TYPED_TEST(Pose3Test, IsApprox) {
  using T = TypeParam;

  std::default_random_engine generator(54321);
  std::uniform_real_distribution<double> uniform(-1., 1.);
  auto rand_uniform = std::bind(uniform, generator);

  T tolerance = 1e-3;

  for (int i = 0; i < 100; ++i) {
    // Create random pose - starting with identity transform for i == 0
    T x_a = T(i) * rand_uniform();
    T y_a = T(i) * rand_uniform();
    T z_a = T(i) * rand_uniform();
    T Rr_a = T(i) * rand_uniform();
    T Rp_a = T(i) * rand_uniform();
    T Ry_a = T(i) * rand_uniform();
    Pose3<T> pose_a(SO3<T>(Rr_a, Rp_a, Ry_a), Vector3<T>(x_a, y_a, z_a));

    EXPECT_TRUE(pose_a.isApprox(pose_a, tolerance));

    // Add small perturbations
    T max_b_perturbation = 1e-6;
    T x_b = x_a + rand_uniform() * max_b_perturbation;
    T y_b = y_a + rand_uniform() * max_b_perturbation;
    T z_b = z_a + rand_uniform() * max_b_perturbation;
    T Rr_b = Rr_a + rand_uniform() * max_b_perturbation;
    T Rp_b = Rp_a + rand_uniform() * max_b_perturbation;
    T Ry_b = Ry_a + rand_uniform() * max_b_perturbation;
    Pose3<T> pose_b(SO3<T>(Rr_b, Rp_b, Ry_b), Vector3<T>(x_b, y_b, z_b));

    EXPECT_TRUE(pose_a.isApprox(pose_b, tolerance));

    // Add large perturbations
    T max_c_perturbation = 1e-2;
    T x_c = x_a + rand_uniform() * max_c_perturbation;
    T y_c = y_a + rand_uniform() * max_c_perturbation;
    T z_c = z_a + rand_uniform() * max_c_perturbation;
    T Rr_c = Rr_a + rand_uniform() * max_c_perturbation;
    T Rp_c = Rp_a + rand_uniform() * max_c_perturbation;
    T Ry_c = Ry_a + rand_uniform() * max_c_perturbation;
    Pose3<T> pose_c(SO3<T>(Rr_c, Rp_c, Ry_c), Vector3<T>(x_c, y_c, z_c));

    EXPECT_FALSE(pose_a.isApprox(pose_c, tolerance));
  }
}

}  // namespace
}  // namespace eigenmath
