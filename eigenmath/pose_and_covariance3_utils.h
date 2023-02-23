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

#ifndef EIGENMATH_EIGENMATH_POSE_AND_COVARIANCE3_UTILS_H_
#define EIGENMATH_EIGENMATH_POSE_AND_COVARIANCE3_UTILS_H_

#include "mean_and_covariance.h"
#include "pose3.h"
#include "pose3_utils.h"

namespace eigenmath {

// Returns the result of a normal transformation compounded with a deterministic
// transformation (assuming that `b_pose_c` is perturbed on the left).
inline PoseAndCovariance3d Compound(const Pose3d& a_pose_b,
                                    const PoseAndCovariance3d& b_pose_c) {
  return {/*m=*/a_pose_b * b_pose_c.mean,
          /*c=*/TransformCovariance(a_pose_b, b_pose_c.covariance)};
}

// Returns a_pose_c given `a_pose_b` and `b_pose_c` assuming that the associated
// covariances are not correlated.
inline PoseAndCovariance3d Compound(const PoseAndCovariance3d& a_pose_b,
                                    const PoseAndCovariance3d& b_pose_c) {
  return {/*m=*/a_pose_b.mean * b_pose_c.mean,
          /*c=*/a_pose_b.covariance +
              TransformCovariance(a_pose_b.mean, b_pose_c.covariance)};
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_POSE_AND_COVARIANCE3_UTILS_H_
