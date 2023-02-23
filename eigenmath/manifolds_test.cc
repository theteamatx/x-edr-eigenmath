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

#include "manifolds.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <random>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "distribution.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "numerical_derivatives.h"
#include "pose3.h"
#include "pose3_utils.h"
#include "sampling.h"
#include "so3.h"
#include "types.h"
#include "utils.h"

namespace eigenmath {
namespace {

using ::testing::DoubleNear;
using testing::IsApprox;

template <typename Scalar>
class ManifoldsTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ManifoldsTest, ScalarTypes);

TYPED_TEST(ManifoldsTest, NumericMatrixExpAtZero) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  Matrix<Scalar, 3, 3> identity = Matrix<Scalar, 3, 3>::Identity();
  Matrix<Scalar, 3, 3> zero = Matrix<Scalar, 3, 3>::Zero();

  // test: exp(O) = I
  const Matrix<Scalar, 3, 3> expX = MatrixExp(zero);
  const Matrix<Scalar, 3, 3> expXTaylor =
      manifolds_internal::MatrixExpTaylorImpl(zero, 100);
  EXPECT_THAT(expX, IsApprox(expXTaylor, kTolerance));
  EXPECT_THAT(identity, IsApprox(expXTaylor, kTolerance));
}

TYPED_TEST(ManifoldsTest, NumericMatrixExpPerturbationAtZero) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  // test: both method produce same result for small delta
  Matrix<Scalar, 3, 3> perturbation = Matrix<Scalar, 3, 3>::Zero();
  perturbation(0, 1) = 1e-5;
  const Matrix<Scalar, 3, 3> expX = MatrixExp(perturbation);
  const Matrix<Scalar, 3, 3> expXTaylor =
      manifolds_internal::MatrixExpTaylorImpl(perturbation, 100);
  EXPECT_THAT(expX, IsApprox(expXTaylor, kTolerance));
}

TYPED_TEST(ManifoldsTest, NumericMatrixExpCompareWithAnalytical) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  /*          /O|x\   /I|x\
     test: exp|---| = |---|
              \0|1/   \0|1/ */
  Matrix<Scalar, 3, 3> matrix = Matrix<Scalar, 3, 3>::Zero();
  matrix(0, 1) = 0.;
  matrix(0, 2) = 4.;
  matrix(1, 2) = 5.;
  Matrix<Scalar, 3, 3> expected = Matrix<Scalar, 3, 3>::Identity();
  expected(0, 2) = 4.;
  expected(1, 2) = 5.;
  const Matrix<Scalar, 3, 3> expX = MatrixExp(matrix);
  const Matrix<Scalar, 3, 3> expXTaylor =
      manifolds_internal::MatrixExpTaylorImpl(matrix, 100);
  EXPECT_THAT(expX, IsApprox(expXTaylor, kTolerance));
  EXPECT_THAT(expected, IsApprox(expXTaylor, kTolerance));
}

TYPED_TEST(ManifoldsTest, NumericMatrixExp2dRotations) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();
  // test exp of 2d rotation
  std::vector<Scalar> angles = {0.,   0.00001,   M_PI_4, M_PI_2,
                                M_PI, 2. * M_PI, 9,      10. * M_PI};
  if constexpr (std::is_same_v<Scalar, double>) {
    angles.push_back(1000 * M_PI);
  }
  for (const Scalar angle : angles) {
    Matrix<Scalar, 2, 2> Omega = Matrix<Scalar, 2, 2>::Zero();
    Omega.setZero();
    Omega(0, 1) = -angle;
    Omega(1, 0) = angle;
    Matrix<Scalar, 2, 2> expOmega = MatrixExp(Omega);
    Matrix<Scalar, 2, 2> R = Eigen::Rotation2D<Scalar>(angle).matrix();
    ASSERT_THAT(expOmega, IsApprox(R, kTolerance)) << "angle: " << angle;
  }
}

TYPED_TEST(ManifoldsTest, ExpLogSO3) {
  using Scalar = TypeParam;
  const Scalar kTolerance = Eigen::NumTraits<Scalar>::dummy_precision();

  const Vector3<Scalar> rotation_vectors[] = {
      Vector3<Scalar>(1., 1., 0.),
      Vector3<Scalar>(0., 0., 0.),
      Vector3<Scalar>(0., 0., 1e-5),
      Vector3<Scalar>(0., 0., 1e-9),
      Vector3<Scalar>(0., 0., 0.),
      Vector3<Scalar>(0., 0., 1.),
      Vector3<Scalar>(0., 1., 0.),
      Vector3<Scalar>(0., 1., 1.),
      Vector3<Scalar>(1., 0., 0.),
      Vector3<Scalar>(1., 0., 1.),
      Vector3<Scalar>(1., 1., 0.),
      Vector3<Scalar>(1., 1., 1.),
      Vector3<Scalar>(M_PI, 0., 0.),
      Vector3<Scalar>(M_PI, 0., 10.),
      Vector3<Scalar>(M_PI - 1e-5, 0., 0.),
      Vector3<Scalar>(1., sqrt(M_PI * M_PI - 1.), 0.),
      Vector3<Scalar>(1. + 1e-5, sqrt(M_PI * M_PI - 1.), 0.),
      Vector3<Scalar>(1. + 1e-9, sqrt(M_PI * M_PI - 1.), 0.)};

  for (const Vector3<Scalar>& r : rotation_vectors) {
    for (float sign : {1., -1.}) {
      Vector3<Scalar> rot = r * sign;
      SO3<Scalar> q = ExpSO3(rot);
      ASSERT_TRUE(std::abs(1. - q.quaternion().squaredNorm()) < kTolerance)
          << "rotation vector: " << rot.transpose() << ", norm: " << rot.norm();

      // round-trip test: R(q) = R(ExpSO3(LogSO3(q)))
      SO3<Scalar> after_q(ExpSO3(LogSO3(q)));
      ASSERT_TRUE(std::abs(1. - after_q.quaternion().squaredNorm()) <
                  kTolerance);

      // we don't compare quaternions directly, since q and -q represent the
      // same rotation matrix R
      ASSERT_THAT(q.matrix(), IsApprox(after_q.matrix(), kTolerance))
          << "roundtrip error:\n"
          << q.matrix() - after_q.matrix();

      // test expSO3 against numeric matrix exponential: R(expSO3(rot)) =
      // matrixExp(SkewMatrix(rot))
      Matrix<Scalar, 3, 3> R = MatrixExp(SkewMatrix(rot), 100);
      ASSERT_THAT(q.matrix(), IsApprox(R, kTolerance)) << "matrix exp error:\n"
                                                       << q.matrix() - R;
    }
  }
}

Matrix<double, 1, 2> LogSO2DerivativeNumeric(const SO2d& q) {
  return VectorFieldNumericalDerivative<2, 1>(
      [](const Vector2d& raw) {
        SO2d q{raw.x(), raw.y()};
        Vector<double, 1> v = Vector<double, 1>::Zero();
        v[0] = LogSO2(q);
        return v;
      },
      Vector2d(q.cos_angle(), q.sin_angle()));
}

Matrix<double, 3, 4> LogSO3DerivativeNumeric(const Quaterniond& q) {
  return VectorFieldNumericalDerivative<4, 3>(
      [](const Vector<double, 4>& raw) {
        Quaterniond q{raw[0], raw[1], raw[2], raw[3]};
        q.normalize();
        return LogSO3(SO3d(q));
      },
      Vector<double, 4>(q.w(), q.x(), q.y(), q.z()));
}

struct LogSO3Functor {
  using InputType = Vector4d;
  using ValueType = Vector3d;

  template <typename T>
  void operator()(const Vector4<T>& q, Vector3<T>* value) const {
    *value = LogSO3(SO3<T>(Quaternion<T>(q[0], q[1], q[2], q[3])));
  }
};

Matrix<double, 3, 4> LogRiemannDerivativeNumeric(const Pose2d& p) {
  return VectorFieldNumericalDerivative<4, 3>(
      [](const Vector<double, 4>& raw) {
        return LogRiemann(
            Pose2d{Vector2d{raw[0], raw[1]}, SO2d{raw[2], raw[3]}});
      },
      Vector<double, 4>(p.translation().x(), p.translation().y(),
                        p.so2().cos_angle(), p.so2().sin_angle()));
}

Matrix<double, 6, 7> LogRiemannDerivativeNumeric(const Pose3d& p) {
  return VectorFieldNumericalDerivative<7, 6>(
      [](const Vector<double, 7>& raw) {
        return LogRiemann(Pose3d{Quaterniond{raw[3], raw[4], raw[5], raw[6]},
                                 Vector3d{raw[0], raw[1], raw[2]}});
      },
      MakeVector<double, 7>({p.translation().x(), p.translation().y(),
                             p.translation().z(), p.quaternion().w(),
                             p.quaternion().x(), p.quaternion().y(),
                             p.quaternion().z()}));
}

Matrix3d LogSE2DerivativeTangent0Numerical(const Pose2d& p) {
  // The current step size is chosen by trial-and-error.
  return VectorFieldNumericalDerivative<3, 3>(
      [p](const Vector3d& raw) { return LogSE2(p * ExpSE2(raw)); },
      Vector3d::Zero(), 1e-6);
}

void UnderstandNumericalErrorAssertEq(double lhs, double rhs) {
  if (lhs != rhs) {
    std::cerr << "UnderstandNumericalErrorAssertEq failed\n";
    std::terminate();
  }
}

double UnderstandNumericalErrorComputeDelta(double aa, double xx, double yy) {
  Pose2d p_a{{xx, yy}, aa};
  Pose2d p_u{{xx, yy}, aa};
  Matrix3d expected_a = LogSE2DerivativeTangent0Numerical(p_a);
  Matrix3d actual_a = LogSE2DerivativeTangent0(p_a);
  Matrix3d expected_u = LogSE2DerivativeTangent0Numerical(p_u);
  Matrix3d actual_u = LogSE2DerivativeTangent0(p_u);
  UnderstandNumericalErrorAssertEq(expected_a.rows(), actual_a.rows());
  UnderstandNumericalErrorAssertEq(expected_u.rows(), actual_u.rows());
  UnderstandNumericalErrorAssertEq(expected_a.rows(), expected_u.rows());
  UnderstandNumericalErrorAssertEq(expected_a.cols(), actual_a.cols());
  UnderstandNumericalErrorAssertEq(expected_u.cols(), actual_u.cols());
  UnderstandNumericalErrorAssertEq(expected_a.cols(), expected_u.cols());
  double local_max_delta = 0.0;
  for (int row = 0; row < expected_a.rows(); ++row) {
    for (int col = 0; col < expected_a.cols(); ++col) {
      UnderstandNumericalErrorAssertEq(expected_a(row, col),
                                       expected_u(row, col));
      UnderstandNumericalErrorAssertEq(actual_a(row, col), actual_u(row, col));
      const double delta = std::abs(expected_a(row, col) - actual_a(row, col));
      local_max_delta = std::max(local_max_delta, delta);
    }
  }
  return local_max_delta;
}

TEST(TestManifolds, ExpSE2Zero) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose2d pose2d_dist;
  Vector3d zero1{0, 0, 0};
  Vector3d zero2{0, 0, -0};
  for (int i = 0; i < 100; i++) {
    Pose2d pose = pose2d_dist(rnd_engine);
    Pose2d actual1 = pose * ExpSE2(zero1);
    EXPECT_THAT(pose, IsApprox(actual1));
    Pose2d actual2 = pose * ExpSE2(zero2);
    EXPECT_THAT(pose, IsApprox(actual2));
  }
}

TEST(TestManifolds, ExpSE2SmallAngle) {
  for (double q = 0.1; q > 1e-30; q = 0.5 * q) {
    Vector<double, 3> tangent{0, 0, q};
    Pose2d actual = ExpSE2(tangent);
    EXPECT_THAT(q, DoubleNear(actual.angle(), 1e-12));
  }
}

TEST(TestManifolds, LogSE2SmallAngle) {
  for (double q = 0.1; q > 1e-30; q = 0.5 * q) {
    Pose2d pose{{0, 0}, q};
    Vector<double, 3> actual = LogSE2(pose);
    EXPECT_THAT(q, DoubleNear(actual[2], 1e-12));
  }
}

TEST(TestManifolds, ExpMinusPose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  // Wraps the circle twice.
  NormalDistributionVector<double, 2> normal_vector2d_dist;
  UniformDistributionSO2d so2d_dist;
  for (int i = 0; i < 100; i++) {
    Vector2d translation_rnd_vect = 3.0 * normal_vector2d_dist(rnd_engine);
    double rnd_angle = so2d_dist(rnd_engine).angle();
    Vector3d tangent{translation_rnd_vect.x(), translation_rnd_vect.y(),
                     rnd_angle};
    Pose2d expected = ExpSE2(tangent);
    Pose2d actual = ExpSE2(-tangent).inverse();
    EXPECT_THAT(actual, IsApprox(expected, 1e-12));
  }
}

TEST(TestManifolds, LogMinusPose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose2d pose2d_dist;
  for (int i = 0; i < 100; i++) {
    Pose2d pose = pose2d_dist(rnd_engine);
    Vector3d expected = LogSE2(pose);
    Vector3d actual = -LogSE2(pose.inverse());
    EXPECT_THAT(actual, IsApprox(expected, 1e-12));
  }
}

TEST(TestManifolds, ExpLogPose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  NormalDistributionVector<double, 2> normal_vector2d_dist;
  UniformDistributionSO2d so2d_dist;
  for (int i = 0; i < 50; i++) {
    Vector2d translation_rnd_vect = 3.0 * normal_vector2d_dist(rnd_engine);
    double rnd_angle = so2d_dist(rnd_engine).angle();
    Vector3d expected_tangent{translation_rnd_vect.x(),
                              translation_rnd_vect.y(), rnd_angle};
    Vector3d actual_tangent = LogSE2(ExpSE2(expected_tangent));
    EXPECT_THAT(expected_tangent, IsApprox(actual_tangent, 5e-12));
  }
}

TEST(TestManifolds, LogExpPose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose2d pose2d_dist;
  for (int i = 0; i < 50; i++) {
    Pose2d expected_pose = pose2d_dist(rnd_engine);
    Pose2d actual_pose = ExpSE2(LogSE2(expected_pose));
    EXPECT_THAT(expected_pose, IsApprox(actual_pose, 5e-12));
  }
}

TEST(TestManifolds, ExpMultPose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  NormalDistributionVector<double, 2> normal_vector2d_dist;
  UniformDistributionSO2d so2d_dist;
  for (int i = 0; i < 50; i++) {
    Vector2d translation_rnd_vect = 3.0 * normal_vector2d_dist(rnd_engine);
    double rnd_angle = so2d_dist(rnd_engine).angle();
    Vector3d tangent{translation_rnd_vect.x(), translation_rnd_vect.y(),
                     rnd_angle};
    Pose2d expected_pose = ExpSE2(tangent) * ExpSE2(tangent);
    Pose2d actual_pose = ExpSE2(2.0 * tangent);
    EXPECT_THAT(expected_pose, IsApprox(actual_pose, 1e-12));
  }
}

TEST(TestManifolds, ExpLogRiemannPose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  NormalDistributionVector<double, 2> normal_vector2d_dist;
  UniformDistributionSO2d so2d_dist;
  for (int i = 0; i < 50; i++) {
    Vector2d translation_rnd_vect = 3.0 * normal_vector2d_dist(rnd_engine);
    double rnd_angle = so2d_dist(rnd_engine).angle();
    Vector3d expected_tangent{translation_rnd_vect.x(),
                              translation_rnd_vect.y(), rnd_angle};
    SO2d expected_rotation = ExpSO2(rnd_angle);
    Pose2d actual_pose = ExpRiemann(expected_tangent);
    EXPECT_THAT(translation_rnd_vect, IsApprox(actual_pose.translation()));
    EXPECT_THAT(expected_rotation, IsApprox(actual_pose.so2()));
    Vector3d actual_tangent = LogRiemann(actual_pose);
    EXPECT_THAT(expected_tangent, IsApprox(actual_tangent));
  }
}

TEST(TestManifolds, ExpLogRiemannPose3) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 3);
  NormalDistributionVector<double, 3> normal_vector3d_dist;
  UniformDistributionUnitVector<double, 3> unit_vector3d_dist;
  for (int i = 0; i < 50; i++) {
    Vector3d translation_rnd_vect = 3.0 * normal_vector3d_dist(rnd_engine);
    Vector3d axis_angle_rnd_vect =
        WrapAngle(dist(rnd_engine)) * unit_vector3d_dist(rnd_engine);
    Vector6d expected_tangent = MakeVector<double, 6>(
        {translation_rnd_vect.x(), translation_rnd_vect.y(),
         translation_rnd_vect.z(), axis_angle_rnd_vect.x(),
         axis_angle_rnd_vect.y(), axis_angle_rnd_vect.z()});
    SO3d expected_rotation = ExpSO3(axis_angle_rnd_vect);
    Pose3d actual_pose = ExpRiemann(expected_tangent);
    EXPECT_THAT(translation_rnd_vect, IsApprox(actual_pose.translation()));
    EXPECT_THAT(expected_rotation, IsApprox(actual_pose.so3(), 1e-15));
    Vector6d actual_tangent = LogRiemann(actual_pose);
    EXPECT_THAT(expected_tangent, IsApprox(actual_tangent));
  }
}

TEST(TestManifolds, ExpLogRiemannBasePose2) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  NormalDistributionVector<double, 2> normal_vector2d_dist;
  UniformDistributionSO2d so2d_dist;
  UniformDistributionPose2d pose2d_dist;
  for (int i = 0; i < 50; i++) {
    Vector2d translation_rnd_vect = 3.0 * normal_vector2d_dist(rnd_engine);
    double rnd_angle = so2d_dist(rnd_engine).angle();
    Vector3d expected_tangent{translation_rnd_vect.x(),
                              translation_rnd_vect.y(), rnd_angle};
    Pose2d pose = pose2d_dist(rnd_engine);
    Vector3d actual_tangent =
        LogRiemann(pose, ExpRiemann(pose, expected_tangent));
    EXPECT_THAT(actual_tangent, IsApprox(expected_tangent, 1e-14));
  }
}

TEST(TestManifolds, ExpLogRiemannBasePose3) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  std::normal_distribution<double> dist(0, 3);
  NormalDistributionVector<double, 3> normal_vector3d_dist;
  UniformDistributionUnitVector<double, 3> unit_vector3d_dist;
  UniformDistributionPose3d pose3d_dist;
  for (int i = 0; i < 50; i++) {
    Vector3d translation_rnd_vect = 3.0 * normal_vector3d_dist(rnd_engine);
    Vector3d axis_angle_rnd_vect =
        WrapAngle(dist(rnd_engine)) * unit_vector3d_dist(rnd_engine);
    Vector6d expected_tangent = MakeVector<double, 6>(
        {translation_rnd_vect.x(), translation_rnd_vect.y(),
         translation_rnd_vect.z(), axis_angle_rnd_vect.x(),
         axis_angle_rnd_vect.y(), axis_angle_rnd_vect.z()});
    Pose3d pose = pose3d_dist(rnd_engine);
    Vector6d actual_tangent =
        LogRiemann(pose, ExpRiemann(pose, expected_tangent));
    EXPECT_THAT(actual_tangent, IsApprox(expected_tangent, 1e-14));
  }
}

TEST(TestManifolds, LogSO2DerivativeIdentity) {
  SO2d q{0};
  Matrix<double, 1, 2> expected = MakeMatrix<double, 1, 2>({{0, 1}});
  Matrix<double, 1, 2> actual = LogSO2Derivative(q);
  EXPECT_THAT(expected, IsApprox(actual));
}

TEST(TestManifolds, LogSO2DerivativeNumeric) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionSO2d dist;
  for (int i = 0; i < 50; i++) {
    SO2d so2 = dist(rnd_engine);
    Matrix<double, 1, 2> expected = LogSO2DerivativeNumeric(so2);
    Matrix<double, 1, 2> actual = LogSO2Derivative(so2);
    EXPECT_THAT(actual, IsApprox(expected, 1e-5));
  }
}

TEST(TestManifolds, LogSO3DerivativeManifoldIdentity) {
  Quaterniond q{1, 0, 0, 0};
  Matrix<double, 3, 4> actual = LogSO3DerivativeManifold(q);
  Matrix<double, 3, 4> expected =
      MakeMatrix<double, 3, 4>({{0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}});
  EXPECT_THAT(expected, IsApprox(actual));
}

TEST(TestManifolds, LogSO3DerivativeManifoldNumeric) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionSO3d dist;
  for (int i = 0; i < 50; i++) {
    SO3d so3 = dist(rnd_engine);
    Matrix<double, 3, 4> expected = LogSO3DerivativeNumeric(so3.quaternion());
    Matrix<double, 3, 4> actual = LogSO3DerivativeManifold(so3.quaternion());
    EXPECT_THAT(actual, IsApprox(expected, 1e-5));
  }
}

TEST(TestManifolds, LogSO3DerivativeManifoldSmallNumeric) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionSO3d dist;
  for (int i = 0; i < 10; i++) {
    SO3d so3 = dist(rnd_engine);
    Matrix<double, 3, 4> expected = LogSO3DerivativeNumeric(so3.quaternion());
    Matrix<double, 3, 4> actual = LogSO3DerivativeManifold(so3.quaternion());
    EXPECT_THAT(actual, IsApprox(expected, 1e-5));
  }
}

void CheckExponentialJacobiansCloseToIdentity(const Vector3d& vec) {
  constexpr double kCoeffMatchTolerance = 1e-12;
  EXPECT_THAT(LeftExponentialJacobian(vec),
              IsApprox(Matrix3d::Identity(), kCoeffMatchTolerance));
  EXPECT_THAT(InverseLeftExponentialJacobian(vec),
              IsApprox(Matrix3d::Identity(), kCoeffMatchTolerance));
  EXPECT_THAT(RightExponentialJacobian(vec),
              IsApprox(Matrix3d::Identity(), kCoeffMatchTolerance));
  EXPECT_THAT(InverseRightExponentialJacobian(vec),
              IsApprox(Matrix3d::Identity(), kCoeffMatchTolerance));
}

TEST(TestManifolds, ExponentialJacobiansSmallMagnitude) {
  Vector3d zero_axis_angle = Vector3d::Zero();
  CheckExponentialJacobiansCloseToIdentity(zero_axis_angle);

  // Check that being slightly above the tolerance still works.
  Vector3d small_axis_angle = Vector3d::Zero();
  small_axis_angle(0) = 2.0 * Eigen::NumTraits<double>::dummy_precision();
  CheckExponentialJacobiansCloseToIdentity(small_axis_angle);

  // Check that being slightly below the tolerance still works.
  Vector3d tiny_axis_angle = Vector3d::Zero();
  tiny_axis_angle(0) = 0.5 * Eigen::NumTraits<double>::dummy_precision();
  CheckExponentialJacobiansCloseToIdentity(tiny_axis_angle);
}

TEST(TestManifolds, ExponentialJacobiansKillingForm) {
  constexpr double kCoeffMatchTolerance = 1e-6;
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionSO3d dist;
  for (int i = 0; i < 50; i++) {
    SO3d actual_so3 = dist(rnd_engine);
    Vector3d z = LogSO3(actual_so3);
    Matrix3d expected =
        LeftExponentialJacobian(z) * InverseRightExponentialJacobian(z);
    EXPECT_THAT(actual_so3.matrix(), IsApprox(expected, kCoeffMatchTolerance));
  }
}

TEST(TestManifolds, LogRiemannDerivative2Identity) {
  Pose2d q;
  Matrix<double, 3, 4> expected =
      MakeMatrix<double, 3, 4>({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}});
  Matrix<double, 3, 4> actual = LogRiemannDerivative(q);
  EXPECT_THAT(expected, IsApprox(actual));
}

TEST(TestManifolds, LogRiemannDerivative2Numeric) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose2d dist;
  for (int i = 0; i < 50; i++) {
    Pose2d p = dist(rnd_engine);
    Matrix<double, 3, 4> expected = LogRiemannDerivativeNumeric(p);
    Matrix<double, 3, 4> actual = LogRiemannDerivative(p);
    EXPECT_THAT(actual, IsApprox(expected, 1e-5));
  }
}

TEST(TestManifolds, LogRiemannDerivative3Identity) {
  Pose3d q{Quaterniond::Identity(), Vector3d::Identity()};
  Matrix<double, 6, 7> expected =
      MakeMatrix<double, 6, 7>({{1, 0, 0, 0, 0, 0, 0},
                                {0, 1, 0, 0, 0, 0, 0},
                                {0, 0, 1, 0, 0, 0, 0},
                                {0, 0, 0, 0, 2, 0, 0},
                                {0, 0, 0, 0, 0, 2, 0},
                                {0, 0, 0, 0, 0, 0, 2}});
  Matrix<double, 6, 7> actual = LogRiemannDerivative(q);
  EXPECT_THAT(expected, IsApprox(actual));
}

TEST(TestManifolds, LogRiemannDerivative3Numeric) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose3d dist;
  for (int i = 0; i < 50; i++) {
    Pose3d p = dist(rnd_engine);
    Matrix<double, 6, 7> expected = LogRiemannDerivativeNumeric(p);
    Matrix<double, 6, 7> actual = LogRiemannDerivative(p);
    EXPECT_THAT(actual, IsApprox(expected, 1.1e-5));
  }
}

TEST(TestManifolds, LogSE2DerivativeTangent0Identity) {
  Matrix3d J_id = LogSE2DerivativeTangent0(Pose2d{});
  EXPECT_THAT(Matrix3d::Identity(), IsApprox(J_id));
}

TEST(TestManifolds, DISABLED_UnderstandNumericalError) {
  // This is not really a test, but a way to explore the numerical
  // error behavior which makes the comparising between analytical
  // (logSE2DerivativeTangent0) and finite-difference solution
  // (LogSE2DerivativeTangent0Numerical) very brittle. You can enable
  // this test, pipe stderr into a file, and then plot things of
  // interest, e.g. with gnuplot. The data collection takes a
  // while. It helps to tail -f the data file to see progress.
  //
  // As no one is actually using logSE2DerivativeTangent0(), and we
  // see no upcoming use case, we decided to punt on this issue.

  TestGenerator rnd_engine(kGeneratorTestSeed);
  std::cerr
      << "# 1:angle 2:max_regular 3:max_randomized_xy 4:max_randomized_axy \n";
  for (double aa = -3.0; aa < 3.0; aa += 0.01) {
    std::cerr << aa;
    {  // Regular sampling: sweep X and Y at a fixed resolution.
      double max_regular = 0.0;
      for (double xx = -3.0; xx < 3.0; xx += 0.01) {
        for (double yy = -3.0; yy < 3.0; yy += 0.01) {
          max_regular = std::max(
              max_regular, UnderstandNumericalErrorComputeDelta(aa, xx, yy));
        }
      }
      std::cerr << "\t" << max_regular;
    }
    {  // Randomized XY sampling.
      std::uniform_real_distribution<double> distance_rnd(-3.0, 3.0);
      double max_randomized_xy = 0.0;
      for (int sample = 0; sample < 10000; ++sample) {
        const double xx = distance_rnd(rnd_engine);
        const double yy = distance_rnd(rnd_engine);
        max_randomized_xy =
            std::max(max_randomized_xy,
                     UnderstandNumericalErrorComputeDelta(aa, xx, yy));
      }
      std::cerr << "\t" << max_randomized_xy;
    }
    {  // Randomized AXY sampling. Resample angle, similarly to failing test.
      std::uniform_real_distribution<double> angle_rnd(-std::abs(aa),
                                                       std::abs(aa));
      std::uniform_real_distribution<double> distance_rnd(-3.0, 3.0);
      double max_randomized_axy = 0.0;
      for (int sample = 0; sample < 10000; ++sample) {
        const double aa_resampled = angle_rnd(rnd_engine);
        const double xx = distance_rnd(rnd_engine);
        const double yy = distance_rnd(rnd_engine);
        const double local_max_delta =
            UnderstandNumericalErrorComputeDelta(aa_resampled, xx, yy);
        max_randomized_axy = std::max(max_randomized_axy, local_max_delta);
        if (local_max_delta > 1e5) {
          std::cout << aa_resampled << "\t" << xx << "\t" << yy << "\t"
                    << local_max_delta << "\n";
        }
      }
      std::cerr << "\t" << max_randomized_axy;
    }
    std::cerr << "\n";
  }
}

TEST(TestManifolds, LogSE2DerivativeTangent0Numerical) {
  // Numerical derivative can be quite inaccurate.
  constexpr double kFailureThreshold = 1e-3;
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose2d pose2d_dist;
  for (int i = 0; i < 100; i++) {
    Pose2d p = pose2d_dist(rnd_engine);
    Matrix3d expected = LogSE2DerivativeTangent0Numerical(p);
    Matrix3d actual = LogSE2DerivativeTangent0(p);
    EXPECT_THAT(actual, IsApprox(expected, kFailureThreshold))
        << "at pose " << p;
  }
}

TEST(TestManifolds, LogSE2DerivativeTangent0NumericalSmallAngle) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  UniformDistributionPose2d pose2d_dist;
  for (int i = 0; i < 100; i++) {
    Pose2d p = pose2d_dist(rnd_engine);
    Matrix3d expected = LogSE2DerivativeTangent0Numerical(p);
    Matrix3d actual = LogSE2DerivativeTangent0(p);
    EXPECT_THAT(actual, IsApprox(expected, 1e-5)) << "at pose " << p;
  }
}

}  // namespace
}  // namespace eigenmath
