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

#ifndef EIGENMATH_EIGENMATH_PLANE_CONVERSIONS_H_
#define EIGENMATH_EIGENMATH_PLANE_CONVERSIONS_H_

#include "absl/log/check.h"
#include "pose3.h"
#include "types.h"

namespace eigenmath {

// Converts a pose `ref_pose_plane` to a normal representation of its x-y plane.
template <typename T, int Options>
Plane3<T, kDefaultOptions> PlaneFromPose(
    const Pose3<T, Options>& ref_pose_plane) {
  return {ref_pose_plane.quaternion()
              .matrix()
              .template topRightCorner<3, 1>(),  // normal
          ref_pose_plane.translation()};         // point on plane
}

namespace plane_conversion_details {
// Projects y onto x.
template <typename T>
Vector3<T> ParallelProjection(const Vector3<T>& x, const Vector3<T>& y) {
  return x.dot(y) * x;
}

// Projects y onto the plane normal to x.
template <typename T>
Vector3<T> OrthogonalProjection(const Vector3<T>& x, const Vector3<T>& y) {
  return y - ParallelProjection(x, y);
}

// Creates an orthonormal basis from a normalized z-basis vector and a x-basis
// vector hint.
// Returns the bases as a rotation matrix.
//
// Note: the vectors normalhint_x and orthonormal_z must not be parallel.
template <typename T>
Matrix3<T> OrthonormalBasisXZ(const Vector3<T>& normalhint_x,
                              const Vector3<T>& orthonormal_z) {
  using std::abs;
  CHECK_LT(abs(orthonormal_z.squaredNorm() - T(1)),
           Eigen::NumTraits<T>::dummy_precision())
      << "Must be normalized";

  const Vector3<T> orthonormal_x =
      plane_conversion_details::OrthogonalProjection(orthonormal_z,
                                                     normalhint_x)
          .normalized();
  const Vector3<T> orthonormal_y = orthonormal_z.cross(orthonormal_x);
  Matrix3<T> R;
  R << orthonormal_x, orthonormal_y, orthonormal_z;
  return R;
}

// Creates a rotation matrix from a normal vector.  Extends `normal` to an
// orthonormal basis with `normal` as the z axis.
template <typename T>
Matrix3<T> RotationMatrixFromNormal(const Vector3<T>& normal) {
  // We create a rotation matrix from a normal, but don't really care which one.
  // We need to make sure that our x-basis hint is not parrallel to normal, thus
  // their dot product should not be 1. Here we use a threshold of 0.9 which is
  // sufficiently far away from 1 to avoid numerical issues.
  if (std::abs(normal.x()) < T(0.9)) {
    return OrthonormalBasisXZ({T(1), T(0), T(0)}, normal);
  } else {
    return OrthonormalBasisXZ({T(0), T(1), T(0)}, normal);
  }
}

// Creates a rotation quaternion from a normal.  Extends `normal` to an
// orthonormal basis with `normal` as the z axis.
//
// Similar to RotationMatrixFromNormal, returning the rotation as a quaternion.
template <typename T>
Quaternion<T, kDefaultOptions> RotationQuaternionFromNormal(
    const Vector3<T>& normal) {
  return Quaternion<T, kDefaultOptions>(RotationMatrixFromNormal(normal));
}
}  // namespace plane_conversion_details

// Transforms plane representation to a pose representation using an origin
// hint.  The plane normal becomes the z-axis of the pose.
template <typename T, int Options>
Pose3<T, kDefaultOptions> PoseFromPlane(const Plane3<T, Options>& plane_a,
                                        const Vector3<T>& originhint_a) {
  Vector3<T> origin_a = plane_a.projection(originhint_a);
  Vector3<T> normal = plane_a.normal().eval();
  Quaternion<T, kDefaultOptions> a_R_plane =
      plane_conversion_details::RotationQuaternionFromNormal(normal);
  Pose3<T, kDefaultOptions> a_pose_plane(a_R_plane, origin_a);
  return a_pose_plane;
}

// Transforms plane representation to a pose representation, using the normal as
// the z-axis.
template <typename T, int Options>
Pose3<T, kDefaultOptions> PoseFromPlane(const Plane3<T, Options>& plane_a) {
  Vector3<T> zero = Vector3<T>::Zero();
  return PoseFromPlane(plane_a, zero);
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_PLANE_CONVERSIONS_H_
