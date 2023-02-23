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

#include "test_constants.h"

namespace eigenmath {
namespace testing {

Matrix6d TestCovarianceMatrix6d() {
  Matrix6d covariance = Matrix6d::Zero();
  // clang-format off
  covariance <<
    8.593172, -0.133665, -1.021232, -0.474583,  0.192796, -0.162058,
   -0.133665,  8.230687,  0.204496, -0.871275,  0.118158, -0.502941,
   -1.021232,  0.204496,  8.280552, -2.077038, -0.189275,  1.263804,
   -0.474583, -0.871275, -2.077038,  7.973196, -0.017817, -1.138728,
    0.192796,  0.118158, -0.189275, -0.017817,  8.916864, -0.490197,
   -0.162058, -0.502941,  1.263804, -1.138728, -0.490197,  7.813304;
  // clang-format on
  return covariance;
}

}  // namespace testing
}  // namespace eigenmath
