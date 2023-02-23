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

#include "quasi_random_vector.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace eigenmath {
namespace {

TEST(QuasiRandomVectorTest, ReturnsResultWithinLimits) {
  constexpr std::size_t kNumRandomConfigurations = 100;
  eigenmath::Vector2d lower_limits(-0.5, -0.5);
  eigenmath::Vector2d upper_limits(0.5, 0.5);
  int sequence_index = 0;
  for (std::size_t i = 0; i < kNumRandomConfigurations; ++i) {
    eigenmath::VectorXd joint_configuration_within_limits =
        GetQuasiRandomVector(lower_limits, upper_limits, &sequence_index);

    EXPECT_GE((joint_configuration_within_limits - lower_limits).minCoeff(),
              0.0);
    EXPECT_GE((upper_limits - joint_configuration_within_limits).minCoeff(),
              0.0);
  }
}

TEST(QuasiRandomVectorTest, HigherJointIndicesNotCollinear) {
  // Test that the consecutive samples do not suffer from colinearity in higher
  // indices.
  constexpr int kDimension = 7;
  using VectorType = eigenmath::Vector<double, kDimension>;
  const VectorType min = VectorType::Constant(0.0);
  const VectorType max = VectorType::Constant(1.0);
  std::vector<VectorType> samples;
  int index = 0;
  samples.reserve(100);
  for (int i = 0; i < 100; ++i) {
    samples.push_back(GetQuasiRandomVector(min, max, &index));
  }

  // Look at strides of higher indices.  Ensure that their span does not
  // collapse.
  const int kStride = 3;
  using MatrixType = eigenmath::Matrix<double, kStride, kStride>;
  MatrixType span = MatrixType::Zero();
  for (int i = 0; i < kStride; ++i) {
    span.col(i) = samples[i].tail<kStride>();
  }
  // Check the rank of the span of kStride consecutive samples.
  for (int i = kStride; i < 100; ++i) {
    span.col(i % kStride) = samples[i].tail<kStride>();
    Eigen::FullPivLU<MatrixType> lu_decomp(span);
    EXPECT_THAT(lu_decomp.rank(), testing::Ge(kStride - 1))
        << "Samples " << (i + 1 - kStride) << " to " << i << " are:\n"
        << span;
  }
}

TEST(QuasiRandomVectorDeathTest, DiesOnInvalidInput) {
  EXPECT_DEATH(GetQuasiRandomVector(eigenmath::Vector2d::Zero(),
                                    eigenmath::Vector2d::Zero(), nullptr),
               "Check failed");
  int index = 0;
  EXPECT_DEATH(GetQuasiRandomVector(eigenmath::Vector2d::Zero(),
                                    eigenmath::Vector3d::Zero(), &index),
               "Check failed");
}

TEST(QuasiRandomVectorGeneratorTest, ReturnsResultWithinLimits) {
  constexpr std::size_t kNumRandomConfigurations = 100;
  eigenmath::Vector2d lower_limits(-0.5, -0.5);
  eigenmath::Vector2d upper_limits(0.5, 0.5);
  QuasiRandomVectorGenerator generator(lower_limits, upper_limits);
  for (std::size_t i = 0; i < kNumRandomConfigurations; ++i) {
    eigenmath::VectorXd joint_configuration_within_limits = generator();

    EXPECT_GE((joint_configuration_within_limits - lower_limits).minCoeff(),
              0.0);
    EXPECT_GE((upper_limits - joint_configuration_within_limits).minCoeff(),
              0.0);
  }
}

}  // namespace
}  // namespace eigenmath
