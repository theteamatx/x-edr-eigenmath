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

#include "spline_fit.h"

#include <vector>

#include "gtest/gtest.h"
#include "matchers.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

constexpr double kTolerance = 1.0e-10;

TEST(SplineFit, FitOneSplineSegmentToTwoWaypoints) {
  Vector2d start = Vector2d::Zero();
  Vector2d end = Vector2d::Ones();
  std::vector<VectorXd> waypoints = {Vector2d::Zero(), Vector2d::Ones()};
  auto spline = FitSplineToWaypoints(waypoints, 1);

  EXPECT_EQ(1, spline.NumSegments());
  EXPECT_EQ(2, spline.NumDof());
  EXPECT_THAT(spline.Sample<0>(0.0), IsApprox(start, kTolerance));
  EXPECT_THAT(spline.Sample<0>(1.0), IsApprox(end, kTolerance));
}

TEST(SplineFit, FitManySplineSegmentsToTwoWaypoints) {
  Vector2d start = Vector2d::Zero();
  Vector2d end = Vector2d::Ones();
  std::vector<VectorXd> waypoints = {start, end};
  const int num_segments = 10;
  auto spline = FitSplineToWaypoints(waypoints, num_segments);

  EXPECT_EQ(num_segments, spline.NumSegments());
  EXPECT_EQ(2, spline.NumDof());
  EXPECT_THAT(spline.Sample<0>(0.0), IsApprox(start, kTolerance));
  EXPECT_THAT(spline.Sample<0>(num_segments), IsApprox(end, kTolerance));
}

TEST(SplineFit, FitManySplineSegmentsToManyWaypoints) {
  std::vector<VectorXd> waypoints;
  waypoints.reserve(21);
  for (int i = 0; i <= 10; ++i) {
    waypoints.push_back(Vector2d::Constant(i));
  }
  for (int i = 9; i >= 0; --i) {
    waypoints.push_back(Vector2d::Constant(i));
  }

  const int num_segments = 8;
  const int mid_waypoint = waypoints.size() / 2;
  const int mid_sample_point = num_segments / 2;

  // start fitting with a low acceleration weight
  double acceleration_weight = 0.1;
  auto spline =
      FitSplineToWaypoints(waypoints, num_segments, acceleration_weight);

  // both end-points should match
  EXPECT_THAT(spline.Sample<0>(0.0), IsApprox(waypoints.front(), kTolerance));
  EXPECT_THAT(spline.Sample<0>(num_segments),
              IsApprox(waypoints.back(), kTolerance));

  // convenience function to get mid-point distance
  auto getMidPointDistance = [&waypoints,
                              mid_waypoint](const QuinticSpline& spline) {
    return (spline.Sample<0>(mid_sample_point) - waypoints[mid_waypoint])
        .norm();
  };

  double mid_dist_1 = getMidPointDistance(spline);

  // now fit with higher acceleration weight, and ensure that distance increases
  acceleration_weight = 1.0;
  spline = FitSplineToWaypoints(waypoints, num_segments, acceleration_weight);
  double mid_dist_2 = getMidPointDistance(spline);
  EXPECT_GT(mid_dist_2, mid_dist_1);

  // once more
  acceleration_weight = 10.0;
  spline = FitSplineToWaypoints(waypoints, num_segments, acceleration_weight);
  double mid_dist_3 = getMidPointDistance(spline);
  EXPECT_GT(mid_dist_3, mid_dist_2);

  // now increase the via weight for midpoint and try again
  std::vector<double> via_weights(waypoints.size(), 1.0);
  via_weights[mid_waypoint] = 10.0;
  spline = FitSplineToWaypoints(waypoints, num_segments, acceleration_weight,
                                via_weights);
  double mid_dist_4 = getMidPointDistance(spline);
  EXPECT_LT(mid_dist_4, mid_dist_3);
}

TEST(SplineFit, FitSplineWithConstraints) {
  std::vector<VectorXd> waypoints;
  waypoints.reserve(21);
  for (int i = 0; i <= 10; ++i) {
    waypoints.push_back(Vector2d::Constant(i));
  }
  for (int i = 9; i >= 0; --i) {
    waypoints.push_back(Vector2d::Constant(i));
  }

  double acceleration_weight = 0.1;
  const int num_segments = 8;
  int num_waypoints = waypoints.size();
  auto waypointToSplineIndex = [num_waypoints](int waypoint_index) {
    return (static_cast<double>(waypoint_index) /
            static_cast<double>(num_waypoints - 1)) *
           num_segments;
  };
  double spline_index_1 = waypointToSplineIndex(1);
  double spline_index_2 = waypointToSplineIndex(2);

  std::vector<VariableSubstitution> position_constraints;
  position_constraints.push_back({1, Vector2d::Constant(0.5)});
  std::vector<VariableSubstitution> velocity_constraints;
  velocity_constraints.push_back({2, Vector2d::Constant(0.0)});
  std::vector<VariableSubstitution> acceleration_constraints;
  acceleration_constraints.push_back({2, Vector2d::Constant(0.0)});
  auto spline = FitSplineToWaypoints(
      waypoints, num_segments, acceleration_weight, {}, position_constraints,
      velocity_constraints, acceleration_constraints);

  // check that constraints were satisfied
  EXPECT_THAT(spline.Sample<0>(spline_index_1),
              IsApprox(position_constraints[0].value, kTolerance));
  EXPECT_THAT(spline.Sample<1>(spline_index_2),
              IsApprox(velocity_constraints[0].value, kTolerance));
  EXPECT_THAT(spline.Sample<2>(spline_index_2),
              IsApprox(acceleration_constraints[0].value, kTolerance));
}

TEST(SplineFit, FitSplineSafe) {
  std::vector<VectorXd> waypoints;
  waypoints.reserve(21);
  for (int i = 0; i <= 10; ++i) {
    waypoints.push_back(Vector2d::Constant(i));
  }
  for (int i = 9; i >= 0; --i) {
    waypoints.push_back(Vector2d::Constant(i));
  }
  const int num_segments = 8;
  const int mid_waypoint = waypoints.size() / 2;

  auto validity_checker = [mid_waypoint, &waypoints](
                              const std::vector<VectorXd>& spline_points) {
    std::vector<bool> validity(waypoints.size(), true);
    // if the mid-point is too far away from the desired, we call it invalid
    validity[mid_waypoint] =
        ((spline_points[mid_waypoint] - waypoints[mid_waypoint]).norm() < 0.1);
    return validity;
  };

  // a high acceleration weight should cause an invalid spline
  double acceleration_weight = 10.0;
  auto bad_spline =
      FitSplineToWaypoints(waypoints, num_segments, acceleration_weight);
  auto via_validity =
      validity_checker(bad_spline.Discretize<0>(waypoints.size()));
  EXPECT_FALSE(via_validity[mid_waypoint]);

  // the iterative algorithm should produce a valid spline
  QuinticSpline good_spline;
  bool success = FitSplineToWaypointsSafe(waypoints, num_segments,
                                          validity_checker, acceleration_weight,
                                          {}, {}, {}, {}, good_spline);
  EXPECT_TRUE(success);
  via_validity = validity_checker(good_spline.Discretize<0>(waypoints.size()));
  EXPECT_TRUE(via_validity[mid_waypoint]);
}

}  // namespace
}  // namespace eigenmath
