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

#include "plane_conversions.h"

#include <iostream>
#include <random>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "distribution.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "pose3.h"
#include "sampling.h"
#include "types.h"

namespace eigenmath {

using ::testing::DoubleNear;
using testing::IsApprox;

TEST(PoseFromPlane, InverseOfPlaneFromPoseFromNormal) {
  TestGenerator generator(kGeneratorTestSeed);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  UniformDistributionUnitVector3d vec_dist;

  for (int i = 0; i < 100; ++i) {
    const Vector3d normal = vec_dist(generator);
    const Plane3d plane_world(normal, dist(generator));
    EXPECT_THAT(PlaneFromPose(PoseFromPlane(plane_world)),
                IsApprox(plane_world));
  }
}

TEST(PoseFromPlane, InverseOfPlaneFromPoseFromNormalAndHint) {
  TestGenerator generator(kGeneratorTestSeed);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  UniformDistributionUnitVector3d unit_vec_dist;
  UniformDistributionVector3d vec_dist;

  for (int i = 0; i < 100; ++i) {
    const Vector3d normal = unit_vec_dist(generator);
    const Plane3d plane_world(normal, dist(generator));
    const Vector3d hint = vec_dist(generator);
    EXPECT_THAT(PlaneFromPose(PoseFromPlane(plane_world, hint)),
                IsApprox(plane_world));
  }
}

TEST(PlaneUnit, PoseFromToPlane) {
  const Vector3d normals[] = {Vector3d(1., 0., 0.), Vector3d(1., 1., -1.),
                              Vector3d(0., 1., 1.), Vector3d(1., -2., 3.)};
  const Vector3d origin_hints[] = {Vector3d(0., 0., 0.), Vector3d(4., 1., -1.),
                                   Vector3d(0., -5., 1.),
                                   Vector3d(1., -2., 3.)};
  for (Vector3d normal : normals) {
    normal.normalize();
    for (const Vector3d& originhint : origin_hints) {
      for (double dist : {0., 1., -0.5, 100.}) {
        const Plane3d plane_world(normal, dist);
        const Pose3d world_pose_plane = PoseFromPlane(plane_world, originhint);

        const Matrix3d world_R_plane = world_pose_plane.quaternion().matrix();
        // world_R_plane must be a rotation matrix
        EXPECT_THAT(world_R_plane.determinant(), DoubleNear(1, 1e-5));
        EXPECT_THAT(world_R_plane.transpose() * world_R_plane,
                    IsApprox(Matrix3d::Identity(), 1e-5));
        EXPECT_THAT(world_R_plane.col(2), IsApprox(normal, 1e-5));

        const Plane3d plane2_world = PlaneFromPose(world_pose_plane);
        EXPECT_THAT(plane2_world.normal(), IsApprox(normal, 1e-5));
        EXPECT_THAT(plane2_world.normal().norm(), DoubleNear(1, 1e-5));

        EXPECT_THAT(plane2_world, IsApprox(plane_world));
      }
    }
  }
}

}  // namespace eigenmath
