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

#include "mean_and_covariance.h"

#include <algorithm>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/SVD"
#include "distribution.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "sampling.h"
#include "types.h"

namespace eigenmath {
namespace {

constexpr double kToleranceDbl = 1e-14;
using eigenmath::testing::IsApprox;
using eigenmath::testing::IsApproxMeanAndCovariance;

TEST(TestMeanAndCovariance, EuclideanMeanAndCovariance) {
  {
    EuclideanMeanAndCovariance<double, 4> emac;
    EXPECT_EQ(4, emac.mean.rows());
    EXPECT_EQ(1, emac.mean.cols());
    EXPECT_EQ(4, emac.covariance.rows());
    EXPECT_EQ(4, emac.covariance.cols());
  }
  {
    EuclideanMeanAndCovariance<double, Eigen::Dynamic> emac;
    EXPECT_EQ(0, emac.mean.rows());
    EXPECT_EQ(1, emac.mean.cols());
    EXPECT_EQ(0, emac.covariance.rows());
    EXPECT_EQ(0, emac.covariance.cols());
  }
}

TEST(TestPoseAndCovariance, PoseAndCovariance2) {
  {
    PoseAndCovariance2<double> pac;
    EXPECT_EQ(3, pac.covariance.rows());
    EXPECT_EQ(3, pac.covariance.cols());
  }
}

TEST(TestPoseAndCovariance, PoseAndCovariance3) {
  {
    PoseAndCovariance3<double> pac;
    EXPECT_EQ(6, pac.covariance.rows());
    EXPECT_EQ(6, pac.covariance.cols());
  }
}

TEST(TestMeanAndCovariance, CreateCovariance) {
  {
    Eigen::Matrix3d expected_rotation =
        Eigen::AngleAxisd(30.0 / 180.0 * M_PI,
                          Eigen::Vector3d{1, 1, 1}.normalized())
            .matrix();
    Eigen::Vector3d expected_scale{3, 2, 1};
    Eigen::Matrix3d covariance =
        CreateCovariance(expected_rotation, expected_scale);
    auto svd = covariance.jacobiSvd(Eigen::ComputeFullU);  // NOLINT
    Eigen::Matrix3d actual_rotation = svd.matrixU();
    EXPECT_THAT(expected_rotation, IsApprox(actual_rotation, kToleranceDbl));
    Eigen::Vector3d actual_scale = svd.singularValues().array().sqrt().matrix();
    EXPECT_THAT(expected_scale, IsApprox(actual_scale, kToleranceDbl));
  }
}

TEST(TestMeanAndCovariance, CreateCovarianceAngleAxis) {
  {
    double angle = 30.0 / 180.0 * M_PI;
    Eigen::Vector3d axis{1, 1, 1};
    Eigen::Vector3d scale{3, 2, 1};
    Eigen::Matrix3d expected_covariance = CreateCovariance(
        Eigen::AngleAxis<double>(angle, axis.normalized()).matrix(), scale);
    Eigen::Matrix3d actual_covariance =
        CreateCovarianceAngleAxis(angle, axis, scale);
    EXPECT_THAT(expected_covariance, IsApprox(actual_covariance));
  }
}

TEST(TestMeanAndCovariance, CreateMeanAndCovariance) {
  {
    Eigen::Vector3d expected_mean{2, 1, 3};
    Eigen::Matrix3d expected_rotation =
        Eigen::AngleAxisd(30.0 / 180.0 * M_PI,
                          Eigen::Vector3d{1, 1, 1}.normalized())
            .matrix();
    Eigen::Vector3d expected_scale{3, 2, 1};
    Eigen::Matrix3d expected_covariance =
        CreateCovariance(expected_rotation, expected_scale);
    EXPECT_THAT(CreateEuclideanMeanAndCovariance(
                    expected_mean, expected_rotation, expected_scale),
                IsApproxMeanAndCovariance(EuclideanMeanAndCovariance<double, 3>(
                    expected_mean, expected_covariance)));
  }
}

TEST(TestMeanAndCovariance, FitNormalDistribution) {
  TestGenerator rnd_engine(kGeneratorTestSeed);
  {
    auto mvn = NormalDistributionVector2d();
    std::vector<Vector2d> samples(500000);
    std::generate(samples.begin(), samples.end(),
                  [&mvn, &rnd_engine]() { return mvn(rnd_engine); });
    auto fit = SampleMeanAndCovariance(samples.begin(), samples.end());
    EXPECT_THAT(fit.mean, IsApprox(Vector2d::Zero(), 0.006));
    EXPECT_THAT(fit.covariance, IsApprox(Matrix2d::Identity(), 0.006));
  }
  {
    auto mvn = NormalDistributionVector3d();
    std::vector<Vector3d> samples(500000);
    std::generate(samples.begin(), samples.end(),
                  [&mvn, &rnd_engine]() { return mvn(rnd_engine); });
    auto fit = SampleMeanAndCovariance(samples.begin(), samples.end());
    EXPECT_THAT(fit.mean, IsApprox(Vector3d::Zero(), 0.01));
    EXPECT_THAT(fit.covariance, IsApprox(Matrix3d::Identity(), 0.02));
  }
}

}  // namespace
}  // namespace eigenmath
