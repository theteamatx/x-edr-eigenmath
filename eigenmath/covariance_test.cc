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

#include "covariance.h"

#include "Eigen/Geometry"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "test_constants.h"

namespace eigenmath {
namespace {

template <typename Matrix>
class SqrtInformationTest : public ::testing::Test {};

using eigenmath::testing::IsApprox;
using ::testing::DoubleNear;
using SqrtInformationTestTypes =
    ::testing::Types<Matrix2d, Matrix3d, Matrix4d, Matrix<double, 5, 5>,
                     Matrix6d>;
TYPED_TEST_SUITE(SqrtInformationTest, SqrtInformationTestTypes);

TYPED_TEST(SqrtInformationTest, LLTCovariancesOfDifferentSizes) {
  const TypeParam covariance =
      testing::TestCovarianceMatrix6d()
          .block<TypeParam::RowsAtCompileTime, TypeParam::RowsAtCompileTime>(0,
                                                                             0);
  absl::StatusOr<TypeParam> sqrt_info = GetSqrtInformationLLT(covariance);
  ASSERT_TRUE(sqrt_info.status().ok());
  EXPECT_THAT(sqrt_info->transpose() * (*sqrt_info) * covariance,
              IsApprox(TypeParam::Identity()));
}

Matrix2d CloseToSingularCovarianceMatrix() {
  Matrix2d covariance = Matrix2d::Zero();
  // clang-format off
  covariance <<
    2, 6,
    6, 18+1e-9;
  // clang-format on
  return covariance;
}

Matrix2d IndefiniteCovarianceMatrix() {
  Matrix2d covariance = Matrix2d::Zero();
  // clang-format off
  covariance <<
    2, 6,
    6, 18-1e-9;
  // clang-format on
  return covariance;
}

TEST(SqrtInformationTest, CloseToSingular) {
  const Matrix2d covariance = CloseToSingularCovarianceMatrix();
  absl::StatusOr<Matrix2d> sqrt_info_llt = GetSqrtInformationLLT(covariance);
  ASSERT_TRUE(sqrt_info_llt.status().ok());
  EXPECT_THAT(sqrt_info_llt->transpose() * (*sqrt_info_llt) * covariance,
              IsApprox(Matrix2d::Identity(), 1e-5));
}

TEST(SqrtInformationTestDeathTest, IndefiniteMatrix) {
  const Matrix2d covariance = IndefiniteCovarianceMatrix();
  EXPECT_DEATH(*GetSqrtInformationLLT(covariance),
               "Covariance matrix need to be a positive definite matrix.");
}

TEST(EllipsoidCovarianceTest, GetEllipsoid) {
  const SO3d rotation(Quaterniond(
      Eigen::AngleAxisd(-0.5, Vector3d(1.1, 2.2, 3.3).normalized())));
  const Vector3d scale(1, 2, 3);
  // Construct an axis-aligned covariance matrix and rotate it.
  const Matrix3d covariance = Eigen::DiagonalMatrix<double, 3>(scale);
  const Matrix3d rotated_covariance =
      rotation.matrix() * covariance * rotation.matrix().transpose();
  constexpr double kSqrtChiSqValue3Dof = 1.2345;
  const auto ellipsoid = GetEllipsoid(rotated_covariance, kSqrtChiSqValue3Dof);
  ASSERT_TRUE(ellipsoid.status().ok());
  EXPECT_THAT(
      ellipsoid->scale,
      IsApprox(Vector3d(scale.array().sqrt().matrix() * kSqrtChiSqValue3Dof)));
  EXPECT_THAT(ellipsoid->origin_rotation_ellipsoid, IsApprox(rotation));
}

// Some eigendecomposition algorithms return a reflection matrix instead of a
// rotation matrix since it satisfies the decomposition requirements. Therefore,
// GetEllipsoid needs to handle such cases and convert the reflection matrix to
// a rotation matrix.
TEST(EllipsoidCovarianceTest, ReflectionMatrix) {
  Matrix3d covariance = Matrix3d::Zero();
  // clang-format off
  covariance <<
    0.828111, -0.110297, -0.589187,
   -0.110297,   1.39998,  0.513806,
   -0.589187,  0.513806,  0.593931;
  // clang-format on
  const auto ellipsoid =
      GetEllipsoid(covariance, /*sqrt_chi_squared_value_3dof=*/1.2345);
  ASSERT_TRUE(ellipsoid.status().ok());
  EXPECT_THAT(ellipsoid->origin_rotation_ellipsoid.matrix().determinant(),
              DoubleNear(1.0, /*max_abs_error=*/1e-10));
}

}  // namespace
}  // namespace eigenmath
