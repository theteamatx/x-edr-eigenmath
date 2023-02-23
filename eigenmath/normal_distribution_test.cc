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

#include "normal_distribution.h"

#include <algorithm>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "matchers.h"
#include "mean_and_covariance.h"

namespace eigenmath {
namespace {

using ::testing::DoubleEq;
using testing::IsApprox;

TEST(TestNormalDistribution, Dimension) {
  {
    auto mvn = EuclideanNormalDistribution<double, 2>::CreateStandard();
    EXPECT_EQ(2, mvn.Dimension());
  }
  {
    auto mvn = EuclideanNormalDistribution<double, 3>::CreateStandard();
    EXPECT_EQ(3, mvn.Dimension());
  }
  {
    auto mvn = EuclideanNormalDistributionX<double>::CreateStandard(21);
    EXPECT_EQ(21, mvn.Dimension());
  }
  {
    auto mvn = EuclideanNormalDistribution<double, 3>(
        Vector3d{1, 2, 3},
        CreateCovarianceAngleAxis(30.0 / 180.0 * M_PI, {1, 1, 1}, {2, 1, 3}));
    EXPECT_EQ(3, mvn.Dimension());
  }
}

TEST(TestNormalDistribution, Mean) {
  {
    auto mvn = EuclideanNormalDistribution<double, 3>::CreateStandard();
    EXPECT_THAT(Vector3d::Zero(), IsApprox(mvn.Mean()));
  }
  {
    auto mvn = EuclideanNormalDistributionX<double>::CreateStandard(17);
    EXPECT_THAT(VectorXd::Zero(17), IsApprox(mvn.Mean()));
  }
  {
    Vector3d mean{1, 2, 3};
    Matrix3d covariance =
        CreateCovarianceAngleAxis(30.0 / 180.0 * M_PI, {1, 1, 1}, {2, 1, 3});
    auto mvn = EuclideanNormalDistribution<double, 3>(mean, covariance);
    EXPECT_THAT(mean, IsApprox(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<SO2d>::CreateStandard();
    EXPECT_THAT(SO2d{}, IsApprox(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<SO3d>::CreateStandard();
    EXPECT_THAT(SO3d{}, IsApprox(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<Pose2d>::CreateStandard();
    EXPECT_THAT(Pose2d::Identity(), IsApprox(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<Pose3d>::CreateStandard();
    EXPECT_THAT(Pose3d::Identity(), IsApprox(mvn.Mean()));
  }
}

TEST(TestNormalDistribution, Covariance) {
  {
    auto mvn = EuclideanNormalDistribution<double, 3>::CreateStandard();
    EXPECT_THAT(Matrix3d::Identity(), IsApprox(mvn.Covariance()));
  }
  {
    auto mvn = EuclideanNormalDistributionX<double>::CreateStandard(17);
    EXPECT_THAT(MatrixXd::Identity(17, 17), IsApprox(mvn.Covariance()));
  }
  {
    Vector3d mean{1, 2, 3};
    Matrix3d covariance =
        CreateCovarianceAngleAxis(30.0 / 180.0 * M_PI, {1, 1, 1}, {2, 1, 3});
    auto mvn = EuclideanNormalDistribution<double, 3>(mean, covariance);
    EXPECT_THAT(covariance, IsApprox(mvn.Covariance()));
  }
  {
    auto mvn = NormalDistribution<SO2d>::CreateStandard();
    EXPECT_THAT(1.0, DoubleEq(mvn.Covariance()(0, 0)));
  }
  {
    auto mvn = NormalDistribution<SO3d>::CreateStandard();
    EXPECT_THAT(Matrix3d::Identity(), IsApprox(mvn.Covariance()));
  }
  {
    auto mvn = NormalDistribution<Pose2d>::CreateStandard();
    EXPECT_THAT(Matrix3d::Identity(), IsApprox(mvn.Covariance()));
  }
  {
    auto mvn = NormalDistribution<Pose3d>::CreateStandard();
    EXPECT_THAT(Matrix6d::Identity(), IsApprox(mvn.Covariance()));
  }
}

TEST(TestNormalDistribution, StandardMeanProbability) {
  // Tests function probability at distribution mean.
  {
    auto mvn = EuclideanNormalDistribution<double, 2>::CreateStandard();
    EXPECT_DOUBLE_EQ(1.0 / (2.0 * M_PI), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = EuclideanNormalDistribution<double, 3>::CreateStandard();
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -1.5), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = EuclideanNormalDistribution<double, 4>::CreateStandard();
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -2.0), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = EuclideanNormalDistributionX<double>::CreateStandard(17);
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -8.5), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<SO2d>::CreateStandard();
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -0.5), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<SO3d>::CreateStandard();
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -1.5), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<Pose2d>::CreateStandard();
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -1.5), mvn.Probability(mvn.Mean()));
  }
  {
    auto mvn = NormalDistribution<Pose3d>::CreateStandard();
    EXPECT_DOUBLE_EQ(std::pow(2.0 * M_PI, -3.0), mvn.Probability(mvn.Mean()));
  }
}

TEST(TestNormalDistribution, MahalanobisDistance) {
  // Tests mahalanobisDistance and mahalanobisDistanceSquared using some hand
  // test cases.
  {
    EuclideanNormalDistribution<double, 3> mvn =
        EuclideanNormalDistribution<double, 3>::CreateStandard();
    EXPECT_DOUBLE_EQ(0, mvn.MahalanobisDistance(Vector3d::Zero()));
    EXPECT_DOUBLE_EQ(0, mvn.MahalanobisDistance({0, 0, 0}));
    EXPECT_DOUBLE_EQ(0, mvn.MahalanobisDistanceSquared({0, 0, 0}));
    EXPECT_DOUBLE_EQ(std::sqrt(14), mvn.MahalanobisDistance({2, 1, 3}));
    EXPECT_DOUBLE_EQ(14, mvn.MahalanobisDistanceSquared({2, 1, 3}));
  }
  {
    EuclideanNormalDistribution<double, 3> mvn =
        EuclideanNormalDistribution<double, 3>(
            Vector3d{1, 2, 3}, Eigen::DiagonalMatrix<double, 3>(2, 1, 3));
    EXPECT_DOUBLE_EQ(std::sqrt(7.5), mvn.MahalanobisDistance({0, 0, 0}));
    EXPECT_DOUBLE_EQ(7.5, mvn.MahalanobisDistanceSquared({0, 0, 0}));
    EXPECT_DOUBLE_EQ(0, mvn.MahalanobisDistance({1, 2, 3}));
    EXPECT_DOUBLE_EQ(0, mvn.MahalanobisDistanceSquared({1, 2, 3}));
  }
}

}  // namespace
}  // namespace eigenmath
