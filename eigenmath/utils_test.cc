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

#include "utils.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include "Eigen/Core"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "line_utils.h"
#include "matchers.h"
#include "types.h"
#include "vector_utils.h"

namespace eigenmath {
namespace {

constexpr double kEpsilon = 1e-6;

using testing::IsApprox;

TEST(EigenMathUtils, MakeVectorStaticDouble) {
  auto vector = MakeVector<double, 2>({1.0, 2.0});  // NOLINT
  static_assert(vector.RowsAtCompileTime == 2, "Incorrect number of rows");
  EXPECT_EQ(2, vector.size());
  EXPECT_EQ(1.0, vector(0));
  EXPECT_EQ(2.0, vector(1));
}

TEST(EigenMathUtils, MakeVectorDynamicDouble) {
  auto vector = MakeVector({1.0, 2.0});  // NOLINT
  static_assert(vector.RowsAtCompileTime == Eigen::Dynamic,
                "Number of rows wasn't dynamic");
  EXPECT_EQ(2, vector.size());
  EXPECT_EQ(1.0, vector(0));
  EXPECT_EQ(2.0, vector(1));
}

TEST(EigenMathUtils, MakeVectorStaticFloat) {
  auto vector = MakeVector<float, 2>({1.0f, 2.0f});  // NOLINT
  static_assert(vector.RowsAtCompileTime == 2, "Incorrect number of rows");
  EXPECT_EQ(2, vector.size());
  EXPECT_EQ(1.0f, vector(0));
  EXPECT_EQ(2.0f, vector(1));
}

TEST(EigenMathUtils, MakeVectorDynamicFloat) {
  auto vector = MakeVector({1.0f, 2.0f});  // NOLINT
  static_assert(vector.RowsAtCompileTime == Eigen::Dynamic,
                "Number of rows wasn't dynamic");
  EXPECT_EQ(2, vector.size());
  EXPECT_EQ(1.0f, vector(0));
  EXPECT_EQ(2.0f, vector(1));
}

template <typename Derived>
void CheckMatrixValuesd(const Eigen::MatrixBase<Derived>& matrix) {
  EXPECT_EQ(3, matrix.rows());
  EXPECT_EQ(2, matrix.cols());
  EXPECT_DOUBLE_EQ(1.0, matrix(0, 0));
  EXPECT_DOUBLE_EQ(2.0, matrix(0, 1));
  EXPECT_DOUBLE_EQ(3.0, matrix(1, 0));
  EXPECT_DOUBLE_EQ(4.0, matrix(1, 1));
  EXPECT_DOUBLE_EQ(5.0, matrix(2, 0));
  EXPECT_DOUBLE_EQ(6.0, matrix(2, 1));
}

template <typename Derived>
void CheckMatrixValuesf(const Eigen::MatrixBase<Derived>& matrix) {
  EXPECT_EQ(3, matrix.rows());
  EXPECT_EQ(2, matrix.cols());
  EXPECT_FLOAT_EQ(1.0f, matrix(0, 0));
  EXPECT_FLOAT_EQ(2.0f, matrix(0, 1));
  EXPECT_FLOAT_EQ(3.0f, matrix(1, 0));
  EXPECT_FLOAT_EQ(4.0f, matrix(1, 1));
  EXPECT_FLOAT_EQ(5.0f, matrix(2, 0));
  EXPECT_FLOAT_EQ(6.0f, matrix(2, 1));
}

TEST(EigenMathUtils, MakeMatrixStaticDouble) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<double, 3, 2>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  static_assert(matrix.RowsAtCompileTime == 3, "Incorrect number of rows");
  static_assert(matrix.ColsAtCompileTime == 2, "Incorrect number of cols");
  CheckMatrixValuesd(matrix);
}

TEST(EigenMathUtils, MakeMatrixDynamicDouble) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<double>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  static_assert(matrix.RowsAtCompileTime == Eigen::Dynamic,
                "Number of rows wasn't dynamic");
  static_assert(matrix.ColsAtCompileTime == Eigen::Dynamic,
                "Number of cols wasn't dynamic");
  CheckMatrixValuesd(matrix);
}

TEST(EigenMathUtils, MakeMatrixStaticColsDouble) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<double, Eigen::Dynamic, 2>(
      {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  static_assert(matrix.RowsAtCompileTime == Eigen::Dynamic,
                "Number of rows wasn't dynamic");
  static_assert(matrix.ColsAtCompileTime == 2, "Incorrect number of cols");
  CheckMatrixValuesd(matrix);
}

TEST(EigenMathUtils, MakeMatrixStaticRowsDouble) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<double, 3, Eigen::Dynamic>(
      {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  static_assert(matrix.RowsAtCompileTime == 3, "Incorrect number of rows");
  static_assert(matrix.ColsAtCompileTime == Eigen::Dynamic,
                "Number of cols wasn't dynamic");
  CheckMatrixValuesd(matrix);
}

TEST(EigenMathUtils, MakeMatrixStaticFloat) {
  // NOLINTNEXTLINE
  auto matrix =
      MakeMatrix<float, 3, 2>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  static_assert(matrix.RowsAtCompileTime == 3, "Incorrect number of rows");
  static_assert(matrix.ColsAtCompileTime == 2, "Incorrect number of cols");
  CheckMatrixValuesf(matrix);
}

TEST(EigenMathUtils, MakeMatrixDynamicFloat) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<float>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  static_assert(matrix.RowsAtCompileTime == Eigen::Dynamic,
                "Number of rows wasn't dynamic");
  static_assert(matrix.ColsAtCompileTime == Eigen::Dynamic,
                "Number of cols wasn't dynamic");
  CheckMatrixValuesf(matrix);
}

TEST(EigenMathUtils, MakeMatrixStaticColsFloat) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<float, Eigen::Dynamic, 2>(
      {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  static_assert(matrix.RowsAtCompileTime == Eigen::Dynamic,
                "Number of rows wasn't dynamic");
  static_assert(matrix.ColsAtCompileTime == 2, "Incorrect number of cols");
  CheckMatrixValuesf(matrix);
}

TEST(EigenMathUtils, MakeMatrixStaticRowsFloat) {
  // NOLINTNEXTLINE
  auto matrix = MakeMatrix<float, 3, Eigen::Dynamic>(
      {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  static_assert(matrix.RowsAtCompileTime == 3, "Incorrect number of rows");
  static_assert(matrix.ColsAtCompileTime == Eigen::Dynamic,
                "Number of cols wasn't dynamic");
  CheckMatrixValuesf(matrix);
}

TEST(EigenMathUtils, SkewMatrixCheckDouble) {
  const Vector3d skew_vector{1, 2, 3};
  const Matrix3d skew = SkewMatrix(skew_vector);
  EXPECT_THAT(skew(0, 0), 0);
  EXPECT_THAT(skew(0, 1), -skew_vector.z());
  EXPECT_THAT(skew(0, 2), skew_vector.y());
  EXPECT_THAT(skew(1, 0), skew_vector.z());
  EXPECT_THAT(skew(1, 1), 0);
  EXPECT_THAT(skew(1, 2), -skew_vector.x());
  EXPECT_THAT(skew(2, 0), -skew_vector.y());
  EXPECT_THAT(skew(2, 1), skew_vector.x());
  EXPECT_THAT(skew(2, 2), 0);
}

TEST(EigenMathUtils, SkewMatrixCheckFloat) {
  const Vector3f skew_vector{1, 2, 3};
  const Matrix3f skew = SkewMatrix(skew_vector);
  EXPECT_THAT(skew(0, 0), 0);
  EXPECT_THAT(skew(0, 1), -skew_vector.z());
  EXPECT_THAT(skew(0, 2), skew_vector.y());
  EXPECT_THAT(skew(1, 0), skew_vector.z());
  EXPECT_THAT(skew(1, 1), 0);
  EXPECT_THAT(skew(1, 2), -skew_vector.x());
  EXPECT_THAT(skew(2, 0), -skew_vector.y());
  EXPECT_THAT(skew(2, 1), skew_vector.x());
  EXPECT_THAT(skew(2, 2), 0);
}

TEST(EigenMathUtils, SkewMatrixCrossProductCheck) {
  const Vector3d skew_vector{1, 2, 3};
  const Matrix3d skew = SkewMatrix(skew_vector);
  const Vector3d test_vector{4, 5, 6};
  EXPECT_THAT(skew * test_vector, IsApprox(skew_vector.cross(test_vector)));
}

TEST(EigenMathUtils, SkewMatrixSquaredDouble) {
  const Vector3d skew_vector{1, 2, 3};
  const Matrix3d skew = SkewMatrix(skew_vector);
  const Matrix3d skew_sqr = SkewMatrixSquared(skew_vector);
  EXPECT_THAT(skew_sqr, IsApprox(skew * skew));
}

TEST(EigenMathUtils, SkewMatrixSquaredCheckFloat) {
  const Vector3f skew_vector{1, 2, 3};
  const Matrix3f skew = SkewMatrix(skew_vector);
  const Matrix3f skew_sqr = SkewMatrixSquared(skew_vector);
  EXPECT_THAT(skew_sqr, IsApprox(skew * skew));
}

TEST(EigenMathUtils, SkewMatrixSquaredCrossProductCheck) {
  const Vector3d skew_vector{1, 2, 3};
  const Matrix3d skew_sqr = SkewMatrixSquared(skew_vector);
  const Vector3d test_vector{4, 5, 6};
  EXPECT_THAT(skew_sqr * test_vector,
              IsApprox(skew_vector.cross(skew_vector.cross(test_vector))));
}

TEST(EigenMathUtils, DistanceFromLineSegment2d) {
  Vector2d p0(1.0, 2.0);
  Vector2d p1(2.0, 1.0);
  Vector2d pq(3.0, 3.0);
  double dist, s;
  DistanceFromLineSegment(p0, p1, pq, &dist, &s);
  EXPECT_NEAR(s, 0.5, kEpsilon);
  EXPECT_NEAR(dist, 2.1213203, kEpsilon);
  DistanceFromLineSegment(p0, p1, p0, &dist, &s);
  EXPECT_NEAR(s, 0.0, kEpsilon);
  EXPECT_NEAR(dist, 0.0, kEpsilon);
  DistanceFromLineSegment(p0, p1, p1, &dist, &s);
  EXPECT_NEAR(s, 1.0, kEpsilon);
  EXPECT_NEAR(dist, 0.0, kEpsilon);
  Vector2d p0_offset = p0 + Vector2d(0.0, 1.0);
  DistanceFromLineSegment(p0, p1, p0_offset, &dist, &s);
  EXPECT_NEAR(s, -0.5, kEpsilon);
  EXPECT_NEAR(dist, 1.0, kEpsilon);
  Vector2d p1_offset = p1 + Vector2d(1.0, 0.0);
  DistanceFromLineSegment(p0, p1, p1_offset, &dist, &s);
  EXPECT_NEAR(s, 1.5, kEpsilon);
  EXPECT_NEAR(dist, 1.0, kEpsilon);

  DistanceFromLineSegment(p0, p1, p0_offset, &dist);
  EXPECT_NEAR(dist, 1.0, kEpsilon);

  DistanceFromLineSegment(p0, p0, pq, &dist, &s);
  EXPECT_NEAR(dist, (p0 - pq).norm(), kEpsilon);
  EXPECT_NEAR(s, 0.5, kEpsilon);

  // Line parameter == nullptr should be OK.
  DistanceFromLineSegment(p0, p0, pq, &dist, static_cast<double*>(nullptr));
  // distance == nullptr should panic.
  EXPECT_DEATH(
      DistanceFromLineSegment(p0, p0, pq, static_cast<double*>(nullptr)),
      "nullptr");
}

template <typename T>
class LineSegmentsIntersectTest : public ::testing::Test {};

using NumericTestTypes = ::testing::Types<float, double, int>;
TYPED_TEST_SUITE(LineSegmentsIntersectTest, NumericTestTypes);

TYPED_TEST(LineSegmentsIntersectTest, NonintersectingSegments) {
  using Vector = Vector2<TypeParam>;
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 0}, Vector{0, 1},
                                     Vector{4, 1}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 4}, Vector{0, 1},
                                     Vector{4, 5}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 4}, Vector{0, 4},
                                     Vector{3, 4}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 0}, Vector{3, 2},
                                     Vector{5, -1}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{2, 0}, Vector{3, 0},
                                     Vector{4, 0}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 4}, Vector{0, 1},
                                     Vector{2, 3}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{1, 1}, Vector{2, 2},
                                     Vector{3, 3}));
  EXPECT_FALSE(LineSegmentsIntersect(Vector{0, 0}, Vector{2, 2}, Vector{3, 3},
                                     Vector{0, 1}));

  if (std::is_floating_point<TypeParam>::value) {
    constexpr TypeParam smallest_positive =
        std::numeric_limits<TypeParam>::min();
    EXPECT_FALSE(LineSegmentsIntersect(
        Vector{0, 0}, Vector{0, smallest_positive},
        Vector{smallest_positive, 0}, Vector{2 * smallest_positive, 0}));
    EXPECT_FALSE(
        LineSegmentsIntersect(Vector{0, 0}, Vector{0, smallest_positive},
                              Vector{smallest_positive, 0},
                              Vector{smallest_positive, smallest_positive}));
  }
}

TYPED_TEST(LineSegmentsIntersectTest, IntersectingSegments) {
  using Vector = Vector2<TypeParam>;
  EXPECT_TRUE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 0}, Vector{0, 0},
                                    Vector{0, 4}));
  EXPECT_TRUE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 0}, Vector{4, 0},
                                    Vector{5, 0}));
  EXPECT_TRUE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 4}, Vector{2, 1},
                                    Vector{1, 2}));
  EXPECT_TRUE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 4}, Vector{3, 5},
                                    Vector{5, 3}));
  EXPECT_TRUE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 0}, Vector{3, 2},
                                    Vector{5, -2}));
  EXPECT_TRUE(LineSegmentsIntersect(Vector{0, 0}, Vector{4, 0}, Vector{3, 2},
                                    Vector{5, -3}));
  if (std::is_floating_point<TypeParam>::value) {
    constexpr TypeParam smallest_positive =
        std::numeric_limits<TypeParam>::min();
    EXPECT_TRUE(LineSegmentsIntersect(
        Vector{smallest_positive, smallest_positive},
        Vector{0, 2 * smallest_positive},
        Vector{smallest_positive, smallest_positive}, Vector{0, 0}));
    EXPECT_TRUE(
        LineSegmentsIntersect(Vector{smallest_positive, smallest_positive},
                              Vector{0, 2 * smallest_positive},
                              Vector{smallest_positive, smallest_positive},
                              Vector{2 * smallest_positive, 0}));
  }
}

class DistanceBetweenLineSegmentsTest : public ::testing::Test {
 protected:
  // Checks all permutations of end points and line segments to produce the
  // expected distance within the desired tolerance.
  void TestLineSegmentDistanceWithAllPermutations(const LineSegment2d& segment1,
                                                  const LineSegment2d& segment2,
                                                  double expected_distance,
                                                  double tolerance) {
    TestLineSegmentDistancePermutingEndPoints(segment1, segment2,
                                              expected_distance, tolerance);
    TestLineSegmentDistancePermutingEndPoints(segment2, segment1,
                                              expected_distance, tolerance);
  }
  void TestLineSegmentDistancePermutingEndPoints(const LineSegment2d& segment1,
                                                 const LineSegment2d& segment2,
                                                 double expected_distance,
                                                 double tolerance) {
    if (HasFailure()) return;
    EXPECT_NEAR(DistanceBetweenLineSegments(segment1.from, segment1.to,
                                            segment2.from, segment2.to),
                expected_distance, tolerance);
    if (HasFailure()) return;
    EXPECT_NEAR(DistanceBetweenLineSegments(segment1.to, segment1.from,
                                            segment2.from, segment2.to),
                expected_distance, tolerance);
    if (HasFailure()) return;
    EXPECT_NEAR(DistanceBetweenLineSegments(segment1.from, segment1.to,
                                            segment2.to, segment2.from),
                expected_distance, tolerance);
    if (HasFailure()) return;
    EXPECT_NEAR(DistanceBetweenLineSegments(segment1.to, segment1.from,
                                            segment2.to, segment2.from),
                expected_distance, tolerance);
  }
};

TEST_F(DistanceBetweenLineSegmentsTest, IntersectingSegments) {
  const Vector2d points[] = {
      {-1, -1},
      {1, 2},
      {-1, 1},
      {1, -3},
  };
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[1]}, {points[2], points[3]}, 0.0, kEpsilon);
}

TEST_F(DistanceBetweenLineSegmentsTest, ParallelSegments) {
  // Segments on a line.
  const Vector2d points[] = {
      {0, 0},
      {0, 2},
      {0, 4},
      {0, 5},
  };
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[1]}, {points[2], points[3]}, 2.0, kEpsilon);
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[2]}, {points[1], points[3]}, 0.0, kEpsilon);
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[3]}, {points[1], points[2]}, 0.0, kEpsilon);

  // Segments shifted to a parallel lines.
  const Vector2d shifted[] = {
      {1, 0},
      {1, 2},
      {1, 4},
      {1, 5},
  };
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[1]}, {shifted[2], shifted[3]}, std::sqrt(5), kEpsilon);
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[2]}, {shifted[1], shifted[3]}, 1.0, kEpsilon);
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[3]}, {shifted[1], shifted[2]}, 1.0, kEpsilon);
}

TEST_F(DistanceBetweenLineSegmentsTest, GeneralConfiguration) {
  const Vector2d points[] = {
      {1, 1},
      {2, 2},
      {2, 1},
      {3, -1},
  };
  TestLineSegmentDistanceWithAllPermutations(
      {points[0], points[1]}, {points[2], points[3]}, std::sqrt(0.5), kEpsilon);
}

TEST(EigenMathUtils, DistanceFromArc) {
  Vector2d p0(1.0, 2.0);
  Vector2d t0(1.0, 0.0);
  double dist, s;
  // Check for straight-line, closer to the end.
  DistanceFromArc(p0, t0, 0.0, 1.0, Vector2d(3.0, 3.0), &dist, &s);
  EXPECT_NEAR(s, 1.0, kEpsilon);
  EXPECT_NEAR(dist, std::sqrt(2.0), kEpsilon);
  // Check for straight-line, closer to the start.
  DistanceFromArc(p0, t0, 0.0, 1.0, Vector2d(0.0, 3.0), &dist, &s);
  EXPECT_NEAR(s, 0.0, kEpsilon);
  EXPECT_NEAR(dist, std::sqrt(2.0), kEpsilon);
  // Check for straight-line, zero length segment.
  DistanceFromArc(p0, t0, 0.0, 0.0, Vector2d(3.0, 3.0), &dist, &s);
  EXPECT_NEAR(s, 0.5, kEpsilon);
  EXPECT_NEAR(dist, std::sqrt(5.0), kEpsilon);
  // Check for arc, zero length segment.
  DistanceFromArc(p0, t0, M_PI, 0.0, Vector2d(3.0, 3.0), &dist, &s);
  EXPECT_NEAR(s, 0.5, kEpsilon);
  EXPECT_NEAR(dist, std::sqrt(5.0), kEpsilon);
  // Check for right-turning arc, closer to the start.
  DistanceFromArc(p0, t0, M_PI, 1.0, Vector2d(0.9, 2.0), &dist, &s);
  EXPECT_NEAR(s, 0.0, kEpsilon);
  EXPECT_NEAR(dist, 0.1, kEpsilon);
  // Check for right-turning arc, closer to the end.
  DistanceFromArc(p0, t0, M_PI, 1.0, Vector2d(0.9, 2.0 + 2.0 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(s, 1.0, kEpsilon);
  EXPECT_NEAR(dist, 0.1, kEpsilon);
  // Check for left-turning arc, closer to the start.
  DistanceFromArc(p0, t0, -M_PI, 1.0, Vector2d(0.9, 2.0), &dist, &s);
  EXPECT_NEAR(s, 0.0, kEpsilon);
  EXPECT_NEAR(dist, 0.1, kEpsilon);
  // Check for left-turning arc, closer to the end.
  DistanceFromArc(p0, t0, -M_PI, 1.0, Vector2d(0.9, 2.0 - 2.0 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(s, 1.0, kEpsilon);
  EXPECT_NEAR(dist, 0.1, kEpsilon);
  // Check for right-turning arc, query point at center of the arc.
  DistanceFromArc(p0, t0, M_PI, 1.0, Vector2d(1.0, 2.0 + 1.0 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(dist, 1.0 / M_PI, kEpsilon);
  // Check for left-turning arc, query point at center of the arc.
  DistanceFromArc(p0, t0, -M_PI, 1.0, Vector2d(1.0, 2.0 - 1.0 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(dist, 1.0 / M_PI, kEpsilon);
  // Check for right-turning arc, outside the apogee of the arc.
  DistanceFromArc(p0, t0, M_PI, 1.0, Vector2d(2.0, 2.0 + 1.0 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(s, 0.5, kEpsilon);
  EXPECT_NEAR(dist, 1.0 - 1.0 / M_PI, kEpsilon);
  // Check for left-turning arc, outside the apogee of the arc.
  DistanceFromArc(p0, t0, -M_PI, 1.0, Vector2d(2.0, 2.0 - 1.0 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(s, 0.5, kEpsilon);
  EXPECT_NEAR(dist, 1.0 - 1.0 / M_PI, kEpsilon);
  // Check for right-turning arc, inside the arc, slight off-center.
  DistanceFromArc(p0, t0, M_PI, 1.0, Vector2d(0.0, 2.0 + 0.9 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(s, 0.0, kEpsilon);
  EXPECT_NEAR(dist, 1.0402260133, kEpsilon);
  // Check for left-turning arc, inside the arc, slight off-center.
  DistanceFromArc(p0, t0, -M_PI, 1.0, Vector2d(0.0, 2.0 - 0.9 / M_PI), &dist,
                  &s);
  EXPECT_NEAR(s, 0.0, kEpsilon);
  EXPECT_NEAR(dist, 1.0402260133, kEpsilon);
}

// For the test of ProjectPointOutsideVertex, call the region between the
// normals the 'free' region, the one opposite it the 'opposite' region, and the
// other parts 'close to edge'.
// Use value parametrized tests using an inner angle at the vertex (in the
// interval [0, pi)) as the parameter.  The edges have the half-angle to the
// y-axis, and the vertex is at the origin.
class ProjectPointOutsideVertexTest : public ::testing::TestWithParam<double> {
 protected:
  void SetUp() override {
    // Use angle relative to y axis.
    const double angle = GetParam() / 2;
    ASSERT_THAT(angle,
                ::testing::AllOf(::testing::Ge(0.0), ::testing::Le(M_PI / 2)));
    edge1_ = {2 * -std::sin(angle), 2 * std::cos(angle)};
    edge2_ = {7 * std::sin(angle), 7 * std::cos(angle)};
  }
  Vector2d edge1_;
  Vector2d edge2_;
};

INSTANTIATE_TEST_SUITE_P(ProjectPointOutsideVertexTests,
                         ProjectPointOutsideVertexTest,
                         ::testing::Values(0.0, 0.1, 1.0, M_PI / 2,
                                           (3 / 4) * M_PI, M_PI - 0.1, M_PI));

TEST_P(ProjectPointOutsideVertexTest, ZeroProjection) {
  // Point in the opposite region.
  const Vector2d point(0.0, 5.0);

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point),
              Vector2d::Zero());
}

TEST_P(ProjectPointOutsideVertexTest, UnchangedPoint) {
  const Vector2d point(0.0, -5.0);

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point), point);
}

TEST_P(ProjectPointOutsideVertexTest, PointCloseToIncomingEdge) {
  const Vector2d point(-5.0, 1.0);

  const Vector2d orthonormal = RightOrthogonal(edge1_.normalized());
  const double dot_product = orthonormal.dot(point);
  const Vector2d expected = (dot_product > 0)
                                ? Vector2d(orthonormal * dot_product)
                                : Vector2d::Zero();

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point),
              IsApprox(expected));
}

TEST_P(ProjectPointOutsideVertexTest, PointCloseToOutgoingEdge) {
  const Vector2d point(5.0, 1.0);

  const Vector2d orthonormal = LeftOrthogonal(edge2_.normalized());
  const double dot_product = orthonormal.dot(point);
  const Vector2d expected = (dot_product > 0)
                                ? Vector2d(orthonormal * dot_product)
                                : Vector2d::Zero();

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point),
              IsApprox(expected));
}

TEST_P(ProjectPointOutsideVertexTest, PointOnOutsideNormalToIncomingEdge) {
  const Vector2d point = 2 * RightOrthogonal(edge1_);

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point), point);
}

TEST_P(ProjectPointOutsideVertexTest, PointOnOutsideNormalToOutgoingEdge) {
  const Vector2d point = 2 * LeftOrthogonal(edge2_);

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point), point);
}

TEST_P(ProjectPointOutsideVertexTest, PointOnInsideNormalToIncomingEdge) {
  const Vector2d orthonormal1 = RightOrthogonal(edge1_.normalized());
  const Vector2d point = -2 * orthonormal1;

  // Do not use orthonormal to get around relative approximate comparison for
  // close to zero cases.
  const Vector2d orthogonal2 = LeftOrthogonal(edge2_);
  const double dot_product = orthogonal2.dot(point);
  const Vector2d expected =
      (dot_product > 0) ? Vector2d(orthogonal2 * (orthogonal2.dot(point) /
                                                  orthogonal2.squaredNorm()))
                        : Vector2d::Zero();

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point),
              IsApprox(expected));
}

TEST_P(ProjectPointOutsideVertexTest, PointOnInsideNormalToOutgoingEdge) {
  const Vector2d orthonormal2 = LeftOrthogonal(edge2_.normalized());
  const Vector2d point = -2 * orthonormal2;

  // const Vector2d orthogonal = LeftOrthogonal(edge1_);
  const Vector2d orthonormal1 = RightOrthogonal(edge1_.normalized());
  const double dot_product = orthonormal1.dot(point);
  const Vector2d expected =
      (dot_product > 0) ? Vector2d(orthonormal1 * orthonormal1.dot(point))
                        : Vector2d::Zero();

  EXPECT_THAT(ProjectPointOutsideVertex(edge1_, edge2_, point),
              IsApprox(expected));
}

// Use full opposite half plane.
TEST(ProjectPointOutsideVertex, ColinearEdges) {
  const Vector2d edge1(0.0, 4.0);
  const Vector2d edge2 = 3 * edge1;

  {
    const Vector2d point(2.0, -1.0);
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point), point);
    EXPECT_THAT(ProjectPointOutsideVertex(edge2, edge1, point), point);
  }

  {
    const Vector2d point(2.0, 1.0);
    const Vector2d expected(2.0, 0.0);
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point),
                IsApprox(expected));
    EXPECT_THAT(ProjectPointOutsideVertex(edge2, edge1, point),
                IsApprox(expected));
  }

  {
    const Vector2d point(-2.0, -1.0);
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point), point);
    EXPECT_THAT(ProjectPointOutsideVertex(edge2, edge1, point), point);
  }

  {
    const Vector2d point(-2.0, 1.0);
    const Vector2d expected(-2.0, 0.0);
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point),
                IsApprox(expected));
    EXPECT_THAT(ProjectPointOutsideVertex(edge2, edge1, point),
                IsApprox(expected));
  }
}

// Project point onto line.
TEST(ProjectPointOutsideVertex, OppositeColinearEdges) {
  const Vector2d edge1(-5.0, 0.0);
  const Vector2d edge2 = -3 * edge1;

  {
    const Vector2d point(2.0, 1.0);
    const Vector2d expected = Vector2d::Zero();
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point), expected);
  }

  {
    const Vector2d point(-2.0, 1.0);
    const Vector2d expected = Vector2d::Zero();
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point), expected);
  }

  {
    const Vector2d point(2.0, -1.0);
    const Vector2d expected(0.0, -1.0);
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point), expected);
  }

  {
    const Vector2d point(-2.0, -1.0);
    const Vector2d expected(0.0, -1.0);
    EXPECT_THAT(ProjectPointOutsideVertex(edge1, edge2, point), expected);
  }
}

}  // namespace
}  // namespace eigenmath
