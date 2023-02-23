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

#include "quintic_spline.h"

#include <cmath>
#include <fstream>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "matchers.h"
#include "quintic_spline_segment.h"
#include "sampling.h"
#include "spline_factories.h"

namespace eigenmath {
namespace {

// Fills with randomly sampled values in [0, 1].
template <typename T, typename Generator>
void FillWithRandomValues(T& matrix, Generator& generator) {
  std::uniform_real_distribution<> dist;
  // NOLINTNEXTLINE
  for (auto& coefficient : matrix.reshaped()) {
    coefficient = dist(generator);
  }
}

// Generate a random spline.
template <typename Generator>
QuinticSpline RandomSpline(unsigned num_points, Generator& generator) {
  Eigen::VectorXd ctrl_points(num_points * 3);
  FillWithRandomValues(ctrl_points, generator);
  return QuinticSpline(ctrl_points);
}

// Ensure that two splines are equal.
void ExpectSplineEquality(const QuinticSpline& lhs, const QuinticSpline& rhs) {
  EXPECT_EQ(lhs.NumDof(), rhs.NumDof());
  EXPECT_EQ(lhs.NumSegments(), rhs.NumSegments());
  EXPECT_NEAR(lhs.MaxSplineIndex(), rhs.MaxSplineIndex(), 1e-8);

  // check each control point
  for (int i = 0; i <= lhs.NumSegments(); ++i) {
    EXPECT_THAT(lhs.Sample<0>(i), testing::IsApprox(rhs.Sample<0>(i)));
    EXPECT_THAT(lhs.Sample<1>(i), testing::IsApprox(rhs.Sample<1>(i)));
    EXPECT_THAT(lhs.Sample<2>(i), testing::IsApprox(rhs.Sample<2>(i)));
    EXPECT_THAT(lhs.Sample<3>(i), testing::IsApprox(rhs.Sample<3>(i)));
  }
}

// Ensure that what we put in is what we get out.
TEST(QuinticSpline, BasicTest) {
  const int num_points = 3;
  Eigen::VectorXd ctrl_points(num_points * 3);
  TestGenerator generator(kGeneratorTestSeed);
  FillWithRandomValues(ctrl_points, generator);

  QuinticSpline spline(ctrl_points);

  ASSERT_EQ(spline.NumDof(), 1);
  ASSERT_EQ(spline.NumSegments(), num_points - 1);

  double abs_error = 1e-8;

  for (int point = 0; point < num_points; ++point) {
    EXPECT_NEAR(spline.Sample<0>(point)(0), ctrl_points(point * 3 + 0),
                abs_error);
    EXPECT_NEAR(spline.Sample<1>(point)(0), ctrl_points(point * 3 + 1),
                abs_error);
    EXPECT_NEAR(spline.Sample<2>(point)(0), ctrl_points(point * 3 + 2),
                abs_error);
  }
}

// Ensure that discretization works.
TEST(QuinticSpline, Discretize) {
  TestGenerator generator(kGeneratorTestSeed);
  QuinticSpline spline = RandomSpline(3, generator);

  const int num_points = 5;

  std::vector<Eigen::VectorXd> deriv0 = spline.Discretize<0>(num_points);
  std::vector<Eigen::VectorXd> deriv1 = spline.Discretize<1>(num_points);
  std::vector<Eigen::VectorXd> deriv2 = spline.Discretize<2>(num_points);
  std::vector<Eigen::VectorXd> deriv3 = spline.Discretize<3>(num_points);

  EXPECT_THAT(deriv0.front(), testing::IsApprox(spline.Sample<0>(0)));
  EXPECT_THAT(deriv1.front(), testing::IsApprox(spline.Sample<1>(0)));
  EXPECT_THAT(deriv2.front(), testing::IsApprox(spline.Sample<2>(0)));
  EXPECT_THAT(deriv3.front(), testing::IsApprox(spline.Sample<3>(0)));

  double last_index = spline.MaxSplineIndex();
  EXPECT_THAT(deriv0.back(), testing::IsApprox(spline.Sample<0>(last_index)));
  EXPECT_THAT(deriv1.back(), testing::IsApprox(spline.Sample<1>(last_index)));
  EXPECT_THAT(deriv2.back(), testing::IsApprox(spline.Sample<2>(last_index)));
  EXPECT_THAT(deriv3.back(), testing::IsApprox(spline.Sample<3>(last_index)));

  int middle = num_points / 2;
  double middle_index = last_index / 2.0;

  EXPECT_THAT(deriv0[middle],
              testing::IsApprox(spline.Sample<0>(middle_index)));
  EXPECT_THAT(deriv1[middle],
              testing::IsApprox(spline.Sample<1>(middle_index)));
  EXPECT_THAT(deriv2[middle],
              testing::IsApprox(spline.Sample<2>(middle_index)));
  EXPECT_THAT(deriv3[middle],
              testing::IsApprox(spline.Sample<3>(middle_index)));
}

// Ensure that discretize() and sample() return the same results.
TEST(QuinticSpline, MultiSample) {
  TestGenerator generator(kGeneratorTestSeed);
  QuinticSpline spline = RandomSpline(2, generator);

  const int num_points = 5;
  std::vector<double> sample_indices(num_points);
  for (int i = 0; i < num_points; ++i) {
    sample_indices[i] = spline.MaxSplineIndex() * static_cast<double>(i) /
                        static_cast<double>(num_points - 1);
  }

  std::vector<Eigen::VectorXd> discr0 = spline.Discretize<0>(num_points);
  std::vector<Eigen::VectorXd> discr1 = spline.Discretize<1>(num_points);
  std::vector<Eigen::VectorXd> discr2 = spline.Discretize<2>(num_points);
  std::vector<Eigen::VectorXd> discr3 = spline.Discretize<3>(num_points);
  std::vector<Eigen::VectorXd> samples0 = spline.Sample<0>(sample_indices);
  std::vector<Eigen::VectorXd> samples1 = spline.Sample<1>(sample_indices);
  std::vector<Eigen::VectorXd> samples2 = spline.Sample<2>(sample_indices);
  std::vector<Eigen::VectorXd> samples3 = spline.Sample<3>(sample_indices);
  for (int i = 0; i < num_points; ++i) {
    EXPECT_THAT(discr0[i], testing::IsApprox(samples0[i]));
    EXPECT_THAT(discr1[i], testing::IsApprox(samples1[i]));
    EXPECT_THAT(discr2[i], testing::IsApprox(samples2[i]));
    EXPECT_THAT(discr3[i], testing::IsApprox(samples3[i]));
  }
}

TEST(QuinticSpline, MakeSplineConstantVelocity) {
  Eigen::VectorXd start(1);
  Eigen::VectorXd end(1);
  start(0) = 0.0;
  end(0) = 1.0;
  auto spline = MakeSplineConstantVelocity(start, end);

  double abs_error = 1e-8;

  // check end-point positions
  EXPECT_NEAR(spline.Sample<0>(0.0)(0), start(0), abs_error);
  EXPECT_NEAR(spline.Sample<0>(1.0)(0), end(0), abs_error);

  // check end-point accelerations
  EXPECT_NEAR(spline.Sample<2>(0.0)(0), 0.0, abs_error);
  EXPECT_NEAR(spline.Sample<2>(1.0)(0), 0.0, abs_error);

  // check for constant velocity
  double expected_vel = spline.Sample<1>(0.0)(0);
  for (double i = 0; i < 1.0; i += 0.01) {
    EXPECT_NEAR(expected_vel, spline.Sample<1>(i)(0), abs_error);
  }
}

TEST(QuinticSpline, MakeSplineZeroStartEndVelAcc) {
  Eigen::VectorXd start(1);
  Eigen::VectorXd end(1);
  start(0) = 0.0;
  end(0) = 1.0;
  auto spline = MakeSplineZeroStartEndVelAcc(start, end);

  double abs_error = 1e-8;

  // check end-point positions
  EXPECT_NEAR(spline.Sample<0>(0.0)(0), start(0), abs_error);
  EXPECT_NEAR(spline.Sample<0>(1.0)(0), end(0), abs_error);

  // check end-point velocities
  EXPECT_NEAR(spline.Sample<1>(0.0)(0), 0.0, abs_error);
  EXPECT_NEAR(spline.Sample<1>(1.0)(0), 0.0, abs_error);

  // check end-point accelerations
  EXPECT_NEAR(spline.Sample<2>(0.0)(0), 0.0, abs_error);
  EXPECT_NEAR(spline.Sample<2>(1.0)(0), 0.0, abs_error);
}

TEST(QuinticSpline, MakeSplineNaturalWithDirections) {
  Eigen::VectorXd start(2);
  Eigen::VectorXd end(2);
  start(0) = 0.0;
  start(1) = 0.0;
  end(0) = 0.5;
  end(1) = 1.0;
  Eigen::VectorXd start_dir(2);
  Eigen::VectorXd end_dir(2);
  start_dir(0) = 1.0;
  start_dir(1) = 0.0;
  end_dir(0) = std::cos(-1.0);
  end_dir(1) = std::sin(-1.0);
  const QuinticSpline spline =
      MakeSplineNaturalWithDirections(start, start_dir, end, end_dir);

  const double abs_error = 1e-8;

  // check end-point positions
  EXPECT_NEAR(spline.Sample<0>(0.0)(0), start(0), abs_error);
  EXPECT_NEAR(spline.Sample<0>(0.0)(1), start(1), abs_error);
  EXPECT_NEAR(spline.Sample<0>(1.0)(0), end(0), abs_error);
  EXPECT_NEAR(spline.Sample<0>(1.0)(1), end(1), abs_error);

  // check end-point velocities
  const Eigen::VectorXd start_perp_v =
      spline.Sample<1>(0.0) - spline.Sample<1>(0.0).dot(start_dir) * start_dir;
  EXPECT_NEAR(start_perp_v(0), 0.0, abs_error);
  EXPECT_NEAR(start_perp_v(1), 0.0, abs_error);
  const Eigen::VectorXd end_perp_v =
      spline.Sample<1>(1.0) - spline.Sample<1>(1.0).dot(end_dir) * end_dir;
  EXPECT_NEAR(end_perp_v(0), 0.0, abs_error);
  EXPECT_NEAR(end_perp_v(1), 0.0, abs_error);

  // check end-point accelerations
  const Eigen::VectorXd start_perp_a =
      spline.Sample<2>(0.0) - spline.Sample<2>(0.0).dot(start_dir) * start_dir;
  EXPECT_NEAR(start_perp_a(0), 0.0, abs_error);
  EXPECT_NEAR(start_perp_a(1), 0.0, abs_error);
  const Eigen::VectorXd end_perp_a =
      spline.Sample<2>(1.0) - spline.Sample<2>(1.0).dot(end_dir) * end_dir;
  EXPECT_NEAR(end_perp_a(0), 0.0, abs_error);
  EXPECT_NEAR(end_perp_a(1), 0.0, abs_error);
}

TEST(QuinticSpline, MakeSplineFromWaypoints) {
  int num_dof = 2;
  int num_waypoints = 10;
  TestGenerator generator(kGeneratorTestSeed);
  std::vector<Eigen::VectorXd> waypoints;
  waypoints.reserve(num_waypoints);
  for (int i = 0; i < num_waypoints; ++i) {
    waypoints.emplace_back(num_dof);
    FillWithRandomValues(waypoints.back(), generator);
  }

  auto spline = MakeSplineFromWaypoints(waypoints);

  // check each waypoint for required conditions
  for (int i = 0; i < num_waypoints; ++i) {
    EXPECT_THAT(spline.Sample<0>(i), testing::IsApprox(waypoints[i]));
    EXPECT_TRUE(spline.Sample<1>(i).isZero());
    EXPECT_TRUE(spline.Sample<2>(i).isZero());
  }
}

TEST(QuinticSpline, MakeSplineFromWaypointsAndDirections) {
  int num_dof = 2;
  int num_waypoints = 10;
  TestGenerator generator(kGeneratorTestSeed);
  std::vector<Eigen::VectorXd> waypoints;
  std::vector<Eigen::VectorXd> directions;
  for (int i = 0; i < num_waypoints; ++i) {
    waypoints.emplace_back(num_dof);
    FillWithRandomValues(waypoints.back(), generator);
    directions.emplace_back(num_dof);
    FillWithRandomValues(directions.back(), generator);
  }

  const QuinticSpline spline =
      MakeSplineFromWaypointsAndDirections(waypoints, directions);

  // check each waypoint for required conditions
  for (int i = 0; i < num_waypoints; ++i) {
    EXPECT_THAT(spline.Sample<0>(i), testing::IsApprox(waypoints[i]))
        << "Got: " << spline.Sample<0>(i).transpose()
        << ", but expected: " << waypoints[i].transpose();
    const Eigen::VectorXd proj_v =
        (spline.Sample<1>(i).dot(directions[i]) / directions[i].squaredNorm()) *
        directions[i];
    EXPECT_THAT(proj_v, testing::IsApprox(spline.Sample<1>(i)))
        << "Got: " << spline.Sample<1>(i).transpose()
        << ", but expected collinearity with: " << directions[i].transpose()
        << ", projected vector: " << proj_v.transpose();
    const Eigen::VectorXd proj_a =
        (spline.Sample<2>(i).dot(directions[i]) / directions[i].squaredNorm()) *
        directions[i];
    EXPECT_THAT(proj_a, testing::IsApprox(spline.Sample<2>(i)))
        << "Got: " << spline.Sample<2>(i).transpose()
        << ", but expected collinearity with: " << directions[i].transpose()
        << ", projected vector: " << proj_a.transpose();
  }
}

TEST(QuinticSpline, MakeSplineFromControlPoints) {
  const int num_points = 3;
  const int num_dof = 2;
  TestGenerator generator(kGeneratorTestSeed);
  Eigen::MatrixXd ctrl_points(num_points * 3, num_dof);
  FillWithRandomValues(ctrl_points, generator);
  std::vector<Eigen::VectorXd> control_point_vector(num_dof);
  for (int i = 0; i < num_dof; ++i) {
    control_point_vector[i] = ctrl_points.col(i);
  }

  auto spline_from_constructor = QuinticSpline(ctrl_points);
  auto spline_from_matrix_factory = MakeSplineFromControlPoints(ctrl_points);
  auto spline_from_vector_factory =
      MakeSplineFromControlPoints(control_point_vector);

  ExpectSplineEquality(spline_from_matrix_factory, spline_from_constructor);
  ExpectSplineEquality(spline_from_vector_factory, spline_from_constructor);
}

// Ensure that what we put in is what we get out.
TEST(QuinticSplineSegment, TestEndpoints) {
  constexpr int N_DOF = 2;
  TestGenerator generator(kGeneratorTestSeed);
  Eigen::Matrix<double, 2, 6> input_params =
      Eigen::Matrix<double, 2, 6>::Zero();
  FillWithRandomValues(input_params, generator);
  for (double duration : {0.5, 1.0, 2.0}) {
    auto segment = QuinticSplineSegment<N_DOF>{input_params.col(0),
                                               input_params.col(1),
                                               input_params.col(2),
                                               input_params.col(3),
                                               input_params.col(4),
                                               input_params.col(5),
                                               duration};
    EXPECT_THAT(segment.SamplePosition(0.0),
                testing::IsApprox(input_params.col(0)));
    EXPECT_THAT(segment.SampleVelocity(0.0),
                testing::IsApprox(input_params.col(1)));
    EXPECT_THAT(segment.SampleAcceleration(0.0),
                testing::IsApprox(input_params.col(2)));
    EXPECT_THAT(segment.SamplePosition(duration),
                testing::IsApprox(input_params.col(3)));
    EXPECT_THAT(segment.SampleVelocity(duration),
                testing::IsApprox(input_params.col(4)));
    EXPECT_THAT(segment.SampleAcceleration(duration),
                testing::IsApprox(input_params.col(5)));
  }
}

}  // namespace
}  // namespace eigenmath
