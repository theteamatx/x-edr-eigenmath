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

#ifndef EIGENMATH_EIGENMATH_UTILS_H_
#define EIGENMATH_EIGENMATH_UTILS_H_

#include <limits>
#include <type_traits>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "types.h"
#include "vector_utils.h"

namespace eigenmath {

// Projects a homogeneous N-vector with N+1 components to its Cartesian
// counterpart.
// `homogeneous_point` represents a homogeneous N-vector with N+1 components.
template <class Scalar, int NPlusOne, int Options>
Vector<Scalar, NPlusOne - 1> Project(
    const Vector<Scalar, NPlusOne, Options>& homogeneous_point) {
  constexpr int N = NPlusOne - 1;
  static_assert(N >= 1, "must be static dimension > 0");
  return homogeneous_point.template head<N>() / homogeneous_point[N];
}

// Derivative of project function wrt. to homogeneous point
// `homogeneous_point` represents a homogeneous N-vector with N+1 components.
template <class Scalar, int NPlusOne, int Options>
Matrix<Scalar, NPlusOne - 1, NPlusOne> ProjectDerivative(
    const Vector<Scalar, NPlusOne, Options>& homogeneous_point) {
  constexpr int N = NPlusOne - 1;
  static_assert(N >= 1, "must be static dimension > 0");
  Matrix<Scalar, N, NPlusOne> tmp;
  tmp.setIdentity();
  tmp.col(N) = -Project(homogeneous_point);
  return Scalar(1) / homogeneous_point[N] * tmp;
}

// Transforms a Cartesian N-vector to its homogeneous representation.
// `cartesian_point` is a cartesian N-vector
// Returns a homogeneous representation.
template <class Scalar, int N, int Options>
Vector<Scalar, N + 1> Unproject(
    const Vector<Scalar, N, Options>& cartesian_point) {
  static_assert(N >= 1, "must be static dimension > 0");
  Vector<Scalar, N + 1> homogeneous_point;
  homogeneous_point.template head<N>() = cartesian_point;
  homogeneous_point[N] = Scalar(1);
  return homogeneous_point;
}

// Creates a Vector(d|f|i) from the given coefficients.
//
// Creates a dynamically allocated vector by default, but can produce a
// fixed-size matrix by providing the size as a template argument.
//
// For example:
// VectorXd dynamic = MakeVector<float>({1, 2, 3, 4});
// Vector4d fixed   = MakeVector<float, 4>({1, 2, 3, 4});
template <class T, int Size = Eigen::Dynamic>
Vector<T, Size> MakeVector(std::initializer_list<T> values) {
  int size = Size;
  if constexpr (Size == Eigen::Dynamic) {
    size = values.size();
  }
  CHECK_GE(size, 1) << "Vector size must be >= 1";
  CHECK_EQ(size, values.size()) << "Input must have " << size << " elements";
  Vector<T, Size> vector;
  vector.resize(size);
  vector = Eigen::Map<const Vector<T, Size, Eigen::DontAlign>>(values.begin(),
                                                               size, 1);
  return vector;
}

// Creates a Matrix(d|f|i) from the given coefficients.
//
// This creates a dynamically allocated matrix by default, but can be made to
// produce a fixed-size matrix by providing the number of rows and columns as
// template parameters. For example:
//
// auto dynamic = MakeMatrix<float>({{1, 2}, {3, 4}});
// auto fixed   = MakeMatrix<float, 2, 2>({{1, 2}, {3, 4}});
//
// The `values` parameter are nested initializer lists, where the inner lists
// correspond to rows of the matrix.
template <class T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
Matrix<T, Rows, Cols> MakeMatrix(
    const std::initializer_list<std::initializer_list<T>>& values) {
  int num_rows = Rows;
  if (Rows == Eigen::Dynamic) {
    num_rows = values.size();
  }
  CHECK_GE(num_rows, 1) << "num_rows must be >= 1";
  CHECK_EQ(num_rows, values.size())
      << "Input must have " << num_rows << " rows";
  int num_cols = Cols;
  if (num_cols == Eigen::Dynamic) {
    num_cols = values.begin()->size();
  }
  CHECK_GE(num_cols, 1) << "num_cols must be >= 1";
  Matrix<T, Rows, Cols> matrix;
  matrix.resize(num_rows, num_cols);
  int row_index = 0;
  for (const auto& row : values) {
    CHECK_EQ(row.size(), num_cols)
        << "All input rows must have " << num_cols << " cols";
    matrix.row(row_index) =
        Eigen::Map<const Eigen::Matrix<T, 1, Cols>>(row.begin(), 1, num_cols);
    ++row_index;
  }
  return matrix;
}

// Creates a skew-symmetrix matrix matrix from a 3 Vector, such that the cross
// product between two 3-vectors can be evaluated as: a x b = SkewMatrix(a) * b
template <typename Scalar, int Options>
Matrix<Scalar, 3, 3, Options> SkewMatrix(
    const Matrix<Scalar, 3, 1, Options>& v) {
  Matrix<Scalar, 3, 3, Options> skew_matrix;
  skew_matrix(0, 0) = Scalar{0};
  skew_matrix(0, 1) = -v.z();
  skew_matrix(0, 2) = v.y();
  skew_matrix(1, 0) = v.z();
  skew_matrix(1, 1) = Scalar{0};
  skew_matrix(1, 2) = -v.x();
  skew_matrix(2, 0) = -v.y();
  skew_matrix(2, 1) = v.x();
  skew_matrix(2, 2) = Scalar{0};
  return skew_matrix;
}

// Creates a skew-symmetrix matrix square from a 3 Vector, equivalent to
// SkewMatrix(a) * SkewMatrix(a), but cheaper to compute.
// The result is a negative semi-definite matrix.
template <typename Scalar, int Options>
Matrix<Scalar, 3, 3, Options> SkewMatrixSquared(
    const Matrix<Scalar, 3, 1, Options>& v) {
  Matrix<Scalar, 3, 3, Options> result;
  const Scalar xx = v.x() * v.x();
  const Scalar yy = v.y() * v.y();
  const Scalar zz = v.z() * v.z();
  result(0, 0) = -yy - zz;
  result(0, 1) = v.x() * v.y();
  result(0, 2) = v.x() * v.z();
  result(1, 0) = result(0, 1);
  result(1, 1) = -xx - zz;
  result(1, 2) = v.y() * v.z();
  result(2, 0) = result(0, 2);
  result(2, 1) = result(1, 2);
  result(2, 2) = -xx - yy;
  return result;
}

// Returns the distance and line parameter of a point against a line segment.
// When projected on the line segment direction, if the query point falls
// between the two end points of the line segment, then the normalized line
// parameter will fall between 0 and 1, indicating where the projection point
// lies between p0 and p1. Otherwise, the line parameter exceeds the [0,1]
// bounds by the amount necessary to reach the projection point.
// The distance value corresponds to the perpendicular distance between the
// query point and the line if its projection falls within the two end points.
// Otherwise, the distance value is the distance to the nearest end point.
//
// The `line_p0` parameter is the first end-point of the line segment.
// The `line_p1` parameter is the second end-point of the line segment.
// The `p_query` parameter is the query point whose distance we are looking for.
// The `distance` parameter is the output distance value.
// The `normalized_line_param` parameter is the line coordinate from the first
// end-point where the projected query point falls, normalized by the length of
// the line segment.
template <typename T, int N, int Options>
void DistanceFromLineSegment(const Vector<T, N, Options>& line_p0,
                             const Vector<T, N, Options>& line_p1,
                             const Vector<T, N, Options>& p_query, T* distance,
                             T* normalized_line_param) {
  static_assert(
      std::is_floating_point<T>::value,
      "This calculation does not make sense for non-floating point types.");
  CHECK_NE(distance, nullptr);

  typedef Vector<T, N, Options> VectorT;
  const VectorT line_tan = line_p1 - line_p0;
  const VectorT p0_to_p_query = p_query - line_p0;
  const T line_tan_squared_norm = line_tan.squaredNorm();
  if (line_tan_squared_norm < std::numeric_limits<T>::epsilon()) {
    if (normalized_line_param) {
      *normalized_line_param = 0.5;
    }
    *distance = (0.5 * line_p0 + 0.5 * line_p1 - p_query).norm();
  } else {
    const T s = p0_to_p_query.dot(line_tan) / line_tan.squaredNorm();
    if (normalized_line_param) {
      *normalized_line_param = s;
    }
    if (s <= 0.0) {
      *distance = (p_query - line_p0).norm();
    } else if (s >= 1.0) {
      *distance = (p_query - line_p1).norm();
    } else {
      *distance = (p0_to_p_query - s * line_tan).norm();
    }
  }
}

// Same as DistanceFromLineSegment but ignoring the line coordinate.
template <typename T, int N, int Options>
void DistanceFromLineSegment(const Vector<T, N, Options>& line_p0,
                             const Vector<T, N, Options>& line_p1,
                             const Vector<T, N, Options>& p_query,
                             T* distance) {
  DistanceFromLineSegment(line_p0, line_p1, p_query, distance,
                          static_cast<T*>(nullptr));
}

// Same as above but returning the result.
template <typename T, int N, int Options>
T DistanceFromLineSegment(const Vector<T, N, Options>& line_p0,
                          const Vector<T, N, Options>& line_p1,
                          const Vector<T, N, Options>& p_query) {
  T distance;
  DistanceFromLineSegment(line_p0, line_p1, p_query, &distance);
  return distance;
}

namespace utils_detail {
// Dimension-dependent dispatching for line segment intersection test.
template <typename T, int N, int Options>
struct LineSegmentsIntersectDispatcher {
  static_assert(N == 2,
                "Line segment intersection for dimension other than 2 is not "
                "yet supported.");
};

template <typename T, int Options>
struct LineSegmentsIntersectDispatcher<T, 2, Options> {
  bool operator()(const Vector<T, 2, Options>& line0_p0,
                  const Vector<T, 2, Options>& line0_p1,
                  const Vector<T, 2, Options>& line1_p0,
                  const Vector<T, 2, Options>& line1_p1) const {
    using VectorType = Vector<T, 2, Options>;
    auto segment_intersects_line =
        [&](const VectorType& l0, const VectorType& l1, const VectorType& s0,
            const VectorType& s1) {
          const VectorType tangent{l1 - l0};
          const auto v0_cross_line = CrossProduct(tangent, VectorType{s0 - l0});
          const auto v1_cross_line = CrossProduct(tangent, VectorType{s1 - l0});
          return v0_cross_line * v1_cross_line <= 0;
        };
    const Eigen::AlignedBox<T, 2> aabb0 =
        Eigen::AlignedBox<T, 2>(line0_p0).extend(line0_p1);
    const Eigen::AlignedBox<T, 2> aabb1 =
        Eigen::AlignedBox<T, 2>(line1_p0).extend(line1_p1);
    return aabb0.intersects(aabb1) &&
           segment_intersects_line(line0_p0, line0_p1, line1_p0, line1_p1) &&
           segment_intersects_line(line1_p0, line1_p1, line0_p0, line0_p1);
  }
};
}  // namespace utils_detail

// Checks for intersection between two line segments.
template <typename T, int N, int Options>
bool LineSegmentsIntersect(const Vector<T, N, Options>& line0_p0,
                           const Vector<T, N, Options>& line0_p1,
                           const Vector<T, N, Options>& line1_p0,
                           const Vector<T, N, Options>& line1_p1) {
  return utils_detail::LineSegmentsIntersectDispatcher<T, N, Options>()(
      line0_p0, line0_p1, line1_p0, line1_p1);
}

// Returns the distance between two line segments.
template <typename T, int N, int Options>
T DistanceBetweenLineSegments(const Vector<T, N, Options>& line0_p0,
                              const Vector<T, N, Options>& line0_p1,
                              const Vector<T, N, Options>& line1_p0,
                              const Vector<T, N, Options>& line1_p1) {
  if (LineSegmentsIntersect(line0_p0, line0_p1, line1_p0, line1_p1)) {
    return T{0};
  }

  return std::min<T>({DistanceFromLineSegment(line0_p0, line0_p1, line1_p0),
                      DistanceFromLineSegment(line0_p0, line0_p1, line1_p1),
                      DistanceFromLineSegment(line1_p0, line1_p1, line0_p0),
                      DistanceFromLineSegment(line1_p0, line1_p1, line0_p1)});
}

// Returns the distance and arc parameter of a point against an arc.
// If the closest point to the query point falls between the two end points
// of the circular arc, then the normalized arc parameter will fall
// between 0 and 1, indicating where the projection point lies between p0 and
// p1. Otherwise, the arc parameter is clamped to the [0,1] bounds to reach
// the closest end point.
// The distance value corresponds to the distance between the query point and
// its closest point on the circle if that point falls within the arc.
// Otherwise, the distance value is the distance to the nearest end point.
//
// The `arc_p0` parameter is the first end-point of the arc.
// The `arc_t0` parameter is the tangent (normalized) at the first end-point of
// the arc.
// The `arc_kappa` parameter is the signed curvature of the arc (positive: turns
// right).
// The `arc_length` parameter is the arc-length of the arc.
// The `p_query` parameter is the query point whose distance we are looking for.
// The `distance` parameter is the output distance value.
// The `normalized_arc_param` parameter is the arc coordinate from the first
// end-point where the query point falls, normalized by the length of the arc.
template <typename T, int Options>
void DistanceFromArc(const Vector<T, 2, Options>& arc_p0,
                     const Vector<T, 2, Options>& arc_t0, T arc_kappa,
                     T arc_length, const Vector<T, 2, Options>& p_query,
                     T* distance, T* normalized_arc_param) {
  CHECK_NE(distance, nullptr);
  typedef Vector<T, 2, Options> VectorT;
  using std::abs;
  using std::atan2;
  using std::cos;
  using std::sin;

  // Check for a straight-line segment, use distanceFromLineSegment, if so.
  if (abs(arc_kappa) < std::numeric_limits<T>::epsilon()) {
    DistanceFromLineSegment<T, 2, Options>(arc_p0, arc_p0 + arc_t0 * arc_length,
                                           p_query, distance,
                                           normalized_arc_param);
    if (normalized_arc_param) {
      if (*normalized_arc_param < 0.0) {
        *normalized_arc_param = 0.0;
      } else if (*normalized_arc_param > 1.0) {
        *normalized_arc_param = 1.0;
      }
    }
    return;
  }

  // Check for a zero-length segment, trivial solution, if so.
  if (abs(arc_length) < std::numeric_limits<T>::epsilon()) {
    *distance = (arc_p0 - p_query).norm();
    if (normalized_arc_param) {
      *normalized_arc_param = 0.5;
    }
    return;
  }

  const T arc_radius = T(1.0) / abs(arc_kappa);
  const T arc_angle = arc_length / arc_radius;
  const VectorT arc_normal =
      (arc_kappa > 0.0 ? RightOrthogonal(arc_t0) : LeftOrthogonal(arc_t0));
  const VectorT arc_center = arc_p0 + arc_radius * arc_normal;
  // Define v as the vector from the arc's center to the query point.
  const VectorT v = p_query - arc_center;
  const T v_norm = v.norm();
  const T v_dot_t = v.dot(arc_t0);
  const T v_dot_n = v.dot(arc_normal);
  T v_angle = atan2(v_dot_t, -v_dot_n);
  if (v_angle < 0.0) {
    v_angle += 2.0 * M_PI;  // Bring angles to [0, 2*pi).
  }
  if (v_angle <= arc_angle) {
    *distance = abs(v_norm - arc_radius);
    if (normalized_arc_param) {
      *normalized_arc_param = v_angle / arc_angle;
    }
  } else {
    if (abs(2.0 * M_PI - v_angle) < abs(v_angle - arc_angle)) {
      // Closer to start of arc.
      *distance = (arc_p0 - p_query).norm();
      if (normalized_arc_param) {
        *normalized_arc_param = 0.0;
      }
    } else {
      // Closer to end of arc. Compute distance to end point.
      const T arc_sin = sin(arc_angle);
      const T arc_cos = cos(arc_angle);
      const VectorT arc_p1 =
          arc_center + (arc_t0 * arc_sin - arc_normal * arc_cos) * arc_radius;
      *distance = (arc_p1 - p_query).norm();
      if (normalized_arc_param) {
        *normalized_arc_param = 1.0;
      }
    }
  }
}

// Finds the closest point in the area between the outside normals of two edges.
// The point is assumed to be relative to the vertex (so that vertex corresponds
// to the origin), and the edges are assumed to be directed away from the
// vertex.
// The `edge1` parameter is the incoming edge.
// The `edge2` parameter is the outgoing edge.
// The `relative_point` parameter is the point to be projected.
// Note: The edge vectors cannot have a norm close to zero.
template <class Scalar, int Options>
Vector2<Scalar, Options> ProjectPointOutsideVertex(
    const Vector2<Scalar, Options>& edge1,
    const Vector2<Scalar, Options>& edge2,
    const Vector2<Scalar, Options>& relative_point) {
  using VectorType = Vector2<Scalar, Options>;
  // Determine the quadrant containing the point, using half-planes relative to
  // the normals.
  const bool along_edge1 = edge1.dot(relative_point) > 0;
  const bool along_edge2 = edge2.dot(relative_point) > 0;

  // Call the area between the two outside normals the 'free' region.
  const bool in_free_region = !along_edge1 && !along_edge2;
  const VectorType outside_normal1 = RightOrthogonal(edge1);
  const VectorType outside_normal2 = LeftOrthogonal(edge2);
  const bool along_normal1 = outside_normal1.dot(relative_point) > 0;
  const bool along_normal2 = outside_normal2.dot(relative_point) > 0;
  if (in_free_region) {
    return relative_point;
  } else if (along_edge1 && along_normal1) {
    // Project point onto outside normal
    const double squared_norm = edge1.squaredNorm();
    CHECK_GT(squared_norm, std::numeric_limits<double>::epsilon());
    const double scaling = outside_normal1.dot(relative_point) / squared_norm;
    return scaling * outside_normal1;
  } else if (along_edge2 && along_normal2) {
    // Project point onto outside normal
    const double squared_norm = edge2.squaredNorm();
    CHECK_GT(squared_norm, std::numeric_limits<double>::epsilon());
    const double scaling = outside_normal2.dot(relative_point) / squared_norm;
    return scaling * outside_normal2;
  } else {
    return VectorType::Zero();
  }
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_UTILS_H_
