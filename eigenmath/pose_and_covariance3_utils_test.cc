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

#include "pose_and_covariance3_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "pose3_test_utils.h"
#include "rotation_utils.h"

namespace eigenmath {

using eigenmath::testing::IsApprox;
using eigenmath::testing::IsApproxMeanAndCovariance;
using ::testing::Lt;

PoseAndCovariance3d TestNormalTransform(const double t) {
  return {/*m=*/TestPose(t), /*c=*/TestCovariance(t)};
}

TEST(PoseAndCovariance3UtilsTest, Compound) {
  const PoseAndCovariance3d a_pose_b = TestNormalTransform(1.0);
  const PoseAndCovariance3d b_pose_c = TestNormalTransform(2.0);

  // We expect the means to be the same.
  EXPECT_THAT(Compound(a_pose_b.mean, b_pose_c).mean,
              IsApprox(Compound(a_pose_b, b_pose_c).mean));

  // However, the resulting covariance will increase.
  EXPECT_THAT(Compound(a_pose_b.mean, b_pose_c).covariance.determinant(),
              Lt(Compound(a_pose_b, b_pose_c).covariance.determinant()));

  // We expect that compounding a normal transform with no uncertainty is
  // equivalent to compounding with a deterministic transform.
  const PoseAndCovariance3d a_pose_b_zero_covariance(/*m=*/a_pose_b.mean,
                                                     /*c=*/Matrix6d::Zero());
  EXPECT_THAT(
      Compound(a_pose_b.mean, b_pose_c),
      IsApproxMeanAndCovariance(Compound(a_pose_b_zero_covariance, b_pose_c)));
}

}  // namespace eigenmath
