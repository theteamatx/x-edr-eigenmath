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

#include "Eigen/Eigenvalues"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "so2.h"

namespace eigenmath {

namespace {
template <int N>
absl::StatusOr<std::pair<Vectord<N>, Matrixd<N>>> GetScaleAndRotation(
    const Matrixd<N> covariance, double sqrt_chi_squared_value) {
  Eigen::SelfAdjointEigenSolver<Matrixd<N>> eig;
  eig.computeDirect(covariance);
  if (eig.info() != Eigen::Success) {
    return absl::InvalidArgumentError("Failed to decompose covariance matrix.");
  }
  Matrixd<N> rot = eig.eigenvectors();
  if (eig.eigenvectors().determinant() < 0) {
    rot.col(0) *= -1.0;
  }
  return std::pair<Vectord<N>, Matrixd<N>>{
      sqrt_chi_squared_value * eig.eigenvalues().array().sqrt(), rot};
}
}  // namespace

absl::StatusOr<Ellipsoid3d> GetEllipsoid(const eigenmath::Matrix3d& covariance,
                                         double sqrt_chi_squared_value_3dof) {
  auto scale_rot_pair =
      GetScaleAndRotation(covariance, sqrt_chi_squared_value_3dof);
  if (!scale_rot_pair.ok()) return scale_rot_pair.status();
  return Ellipsoid3d{.origin_rotation_ellipsoid = SO3d(scale_rot_pair->second),
                     .scale = scale_rot_pair->first};
}

absl::StatusOr<Ellipsoid3d> GetEllipsoid(const eigenmath::Matrix2d& covariance,
                                         double sqrt_chi_squared_value_2dof,
                                         double z_scale) {
  auto scale_rot_pair =
      GetScaleAndRotation(covariance, sqrt_chi_squared_value_2dof);
  if (!scale_rot_pair.ok()) return scale_rot_pair.status();
  return Ellipsoid3d{
      .origin_rotation_ellipsoid =
          SO3d(SO2d(scale_rot_pair->second).angle(), Vector3d::UnitZ()),
      .scale = {scale_rot_pair->first.x(), scale_rot_pair->first.y(), z_scale}};
}

}  // namespace eigenmath
