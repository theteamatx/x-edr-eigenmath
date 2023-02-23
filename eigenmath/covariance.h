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

#ifndef EIGENMATH_EIGENMATH_COVARIANCE_H_
#define EIGENMATH_EIGENMATH_COVARIANCE_H_

#include <limits>

#include "Eigen/Cholesky"  // IWYU pragma: keep
#include "absl/status/statusor.h"
#include "so3.h"
#include "types.h"

namespace eigenmath {

// Returns the square root information form of the given positive definite
// covariance matrix. Given a positive definite covariance matrix C, its square
// root information form is a matrix L satisfying L^TL = C^{-1}. Returns error
// if the given matrix is not a positive definite matrix. Invalid if the given
// matrix is singular.
template <typename Scalar, int N, int Options>
absl::StatusOr<Matrix<Scalar, N, N>> GetSqrtInformationLLT(
    const Matrix<Scalar, N, N, Options>& covariance) {
  Eigen::LLT<Matrix<Scalar, N, N>> cov_llt(covariance);

  // Note: the cov_llt.info() does not catch singular case.
  if (cov_llt.info() != Eigen::Success) {
    return absl::FailedPreconditionError(
        "Covariance matrix need to be a positive definite matrix.");
  }
  return cov_llt.matrixL().solve(Matrix<Scalar, N, N, Options>::Identity());
}

struct Ellipsoid3d {
  SO3d origin_rotation_ellipsoid;
  Vector3d scale;
};
absl::StatusOr<Ellipsoid3d> GetEllipsoid(const Matrix3d& covariance,
                                         double sqrt_chi_squared_value_3dof);

absl::StatusOr<Ellipsoid3d> GetEllipsoid(const Matrix2d& covariance,
                                         double sqrt_chi_squared_value_2dof,
                                         double z_scale);

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_COVARIANCE_H_
