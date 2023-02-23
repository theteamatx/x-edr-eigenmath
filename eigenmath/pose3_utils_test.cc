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

#include "pose3_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "mean_and_covariance.h"
#include "pose3_test_utils.h"
#include "rotation_utils.h"

namespace eigenmath {

using eigenmath::testing::IsApprox;

TEST(Pose3UtilsTest, Adjoint) {
  const Pose3d original_transform = TestPose(1.0);
  const Vector6d target(0.1, 0.2, 0.3, 0.01, 0.02, 0.03);
  const Pose3d target_pose = ExpRiemann(target);
  const Vector6d adj_transform_target = AdjointSE3(original_transform) * target;
  const Pose3d adj_transform_target_pose = ExpRiemann(adj_transform_target);
  EXPECT_THAT(original_transform * target_pose * original_transform.inverse(),
              IsApprox(adj_transform_target_pose,
                       /*threshold_norm_translation*/ 0.01,
                       /*threshold_angle*/ 0.001));
}

TEST(Pose3UtilsTest, EquivalenceRotationOnlyTransformAndRotateCovariance) {
  const Pose3d x_pose_y(TestPose(1.0).so3(), Vector3d::Zero());
  const Matrix6d y_pose_z_covariance = TestCovariance(1.0);
  EXPECT_THAT(TransformCovariance(x_pose_y, y_pose_z_covariance),
              IsApprox(RotateCovariance(y_pose_z_covariance, x_pose_y.so3())));
  EXPECT_THAT(
      TransformCovariance(x_pose_y, y_pose_z_covariance).topLeftCorner(3, 3),
      IsApprox(RotateCovariance(
          Matrix3d(y_pose_z_covariance.topLeftCorner(3, 3)), x_pose_y.so3())));
}

TEST(Pose3UtilsTest, EquivalentMahalanobisDistance) {
  const Pose3d x_pose_y = TestPose(1.0);
  // y_pose_z and x_pose_z are normal transforms. We don't need them for the
  // test so they have not been specified.
  const Matrix6d y_pose_z_covariance = TestCovariance(1.0);
  const Matrix6d x_pose_z_covariance =
      TransformCovariance(x_pose_y, y_pose_z_covariance);

  // We expect the same small deviation from x_pose_z and y_pose_z to
  // have the same Mahalanobis distance.
  const Pose3d y_pose_y = TestPose(0.01);
  const Vector6d y_error_z = LogRiemann(y_pose_y);
  const Pose3d x_pose_x = x_pose_y * y_pose_y * x_pose_y.inverse();
  const Vector6d x_error_z = LogRiemann(x_pose_x);
  EXPECT_NEAR(y_error_z.transpose() * y_pose_z_covariance.inverse() * y_error_z,
              x_error_z.transpose() * x_pose_z_covariance.inverse() * x_error_z,
              /*abs_error=*/0.001);
}

}  // namespace eigenmath
