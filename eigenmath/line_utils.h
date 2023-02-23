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

#ifndef EIGENMATH_EIGENMATH_LINE_UTILS_H_
#define EIGENMATH_EIGENMATH_LINE_UTILS_H_

#include <ostream>

#include "absl/types/optional.h"
#include "pose2.h"
#include "rotation_utils.h"

namespace eigenmath {

// Stores an n-dimensional line segment.
template <class Scalar, int N, int Options = kDefaultOptions>
struct LineSegment {
  using VectorType = Vector<Scalar, N, Options>;

  LineSegment() = default;
  LineSegment(const VectorType& from_, const VectorType& to_)
      : from(from_), to(to_) {}

  friend std::ostream& operator<<(std::ostream& os,
                                  const LineSegment& line_segment) {
    return os << "{{" << line_segment.from.x() << ", " << line_segment.from.y()
              << "}, {" << line_segment.to.x() << ", " << line_segment.to.y()
              << "}}";
  }

  VectorType from, to;
};

// 2d line segment using Scalar
template <class Scalar, int Options = kDefaultOptions>
using LineSegment2 = LineSegment<Scalar, 2, Options>;

using LineSegment2f = LineSegment2<float>;
using LineSegment2d = LineSegment2<double>;

// 3d line segment using Scalar
template <class Scalar, int Options = kDefaultOptions>
using LineSegment3 = LineSegment<Scalar, 3, Options>;

using LineSegment3f = LineSegment3<float>;
using LineSegment3d = LineSegment3<double>;

// Transform a line.
template <typename Scalar, int OptionsPose, int OptionsLine>
Line2<Scalar, OptionsLine> operator*(
    const Pose2<Scalar, OptionsPose>& transform,
    const Line2<Scalar, OptionsLine>& line) {
  const Vector2<Scalar> normal = transform.so2() * line.normal();
  return Line2<Scalar, OptionsLine>(
      normal, line.coeffs()(2) - normal.dot(transform.translation()));
}

// Compute intersection of two lines. If the lines are parallel the function
// still returns a point on the line.
template <typename Scalar, int Options>
bool IntersectLines(const Line2<Scalar, Options>& line_a,
                    const Line2<Scalar, Options>& line_b,
                    Vector2<Scalar>* intersection,
                    Scalar intersection_threshold = 1e-5) {
  const Vector3<Scalar> cross_prod = line_a.coeffs().cross(line_b.coeffs());
  *intersection = cross_prod.template head<2>();
  if (std::abs(cross_prod(2)) < intersection_threshold) {
    return false;  // Lines are parallel.
  }
  *intersection /= cross_prod(2);
  return true;
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_LINE_UTILS_H_
