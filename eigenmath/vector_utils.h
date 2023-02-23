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

#ifndef EIGENMATH_EIGENMATH_VECTOR_UTILS_H_
#define EIGENMATH_EIGENMATH_VECTOR_UTILS_H_

#include <algorithm>
#include <limits>
#include <type_traits>

#include "absl/log/check.h"
#include "genit/filter_iterator.h"
#include "genit/iterator_range.h"
#include "scalar_utils.h"
#include "types.h"

namespace eigenmath {

// Return cross product of two 2D vectors.
template <class Scalar, int Options>
Scalar CrossProduct(const Vector2<Scalar, Options> &aa,
                    const Vector2<Scalar, Options> &bb) {
  return aa.x() * bb.y() - aa.y() * bb.x();
}

// Return dot product of two 2D vectors.
template <class Scalar, int Options>
Scalar DotProduct(const Vector2<Scalar, Options> &aa,
                  const Vector2<Scalar, Options> &bb) {
  return aa.x() * bb.x() + aa.y() * bb.y();
}

// Return triple product of two 2D vectors, (A x B) x C.
template <class Scalar, int Options>
Vector2<Scalar, Options> TripleProduct(const Vector2<Scalar, Options> &aa,
                                       const Vector2<Scalar, Options> &bb,
                                       const Vector2<Scalar, Options> &cc) {
  return bb * DotProduct(cc, aa) - aa * DotProduct(cc, bb);
}

// Return the right-orthogonal vector to a given 2D vector.
template <class Scalar, int Options>
Vector2<Scalar, Options> RightOrthogonal(const Vector2<Scalar, Options> &aa) {
  return Vector2<Scalar, Options>(-aa.y(), aa.x());
}

// Return the left-orthogonal vector to a given 2D vector.
template <class Scalar, int Options>
Vector2<Scalar, Options> LeftOrthogonal(const Vector2<Scalar, Options> &aa) {
  return Vector2<Scalar, Options>(aa.y(), -aa.x());
}

// Builds a (deterministic) orthonormal basis.  Given a vector u, finds vectors
// v and w such that v, w, and the normalized u form an orthonormal basis.
//
// Assumes that the passed in vector has a sufficiently large norm.
//
// The implementation is based on
// http://cs.brown.edu/research/pubs/pdfs/1999/Hughes-1999-BAO.pdf
// See implementation details below in the implementation section.
template <class Scalar>
std::array<Vector3<Scalar>, 2> ExtendToOrthonormalBasis(
    const Vector3<Scalar> &u);

// Return type for the ScaleVectorToLimits function below.
enum class ScaleVectorResult {
  INFEASIBLE,       // Invalid pre-conditions, can't scale.
  VECTOR_SCALED,    // Successfully scaled the whole vector, keeping direction.
  ELEMENTS_SCALED,  // Could scale within limits, but not keeping direction.
  UNCHANGED         // Scaling was unnecessary.
};

// Returns the string corresponding to the given ScaleVectorResult.
const char *ToString(const ScaleVectorResult result);

// Scales down a vector (if needed) to be within the given lower and upper
// limits. This function either:
// 1) Scales inout so it is within [lower, upper] maintaining vector
//    direction and returns ScaleVectorResult::VECTOR_SCALED,
// 2) Does nothing if already in limits and returns
//    ScaleVectorResult::UNCHANGED,
// 3) If any of the upper limits is lower than the lower limits, returns the
//    average for those elements and returns ScaleVectorResult::INFEASIBLE,
// 4) Clamps input within upper and lower bounds if direction preserving
//    scaling is not feasible and returns ScaleVectorResult::ELEMENTS_SCALED.
// Asserts that vector dimensions match.
// See implementation details below in the implementation section.
template <class Scalar, int N, int MaxSize>
ScaleVectorResult ScaleVectorToLimits(
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &lower,
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &upper,
    VectorFixedOrDynamic<Scalar, N, MaxSize> *inout);

// Scales down a vector (if needed) to be within the given symmetric lower and
// upper limits. The elements of the limits vector must be strictly greater than
// zero. This guarantees that scaling will always keep the direction of the
// vector. If the vector is already within the limits, the same vector is
// returned. If some element is outside the limits, the whole vector is scaled
// down to be within the lmits, therefore changing magnitude but not direction.
// Asserts that all elements of symmetric_limits are positive.
// If the range defined by the limits is not symmetric you should use
// ScaleVectorToLimits() instead.
// See implementation details below in the implementation section.
template <class Scalar, int N, int MaxSize>
VectorFixedOrDynamic<Scalar, N, MaxSize> ScaleDownToLimits(
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &vector,
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &symmetric_limits);

// Applies a deadband to `input` and returns the result.
template <class Scalar, int N, int MaxSize>
eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> ApplyDeadband(
    const eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> &input,
    const eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> &band);

///////////////////////////////////////////////////////////////////////////////
// Implementation details below.
///////////////////////////////////////////////////////////////////////////////

template <class Scalar>
std::array<Vector3<Scalar>, 2> ExtendToOrthonormalBasis(
    const Vector3<Scalar> &u) {
  static_assert(std::is_floating_point<Scalar>::value,
                "The operation requires real number arithmetic.");
  using VectorType = Vector3<Scalar>;
  auto compare_absolute_value = [](const Scalar &lhs, const Scalar &rhs) {
    using std::abs;
    return abs(lhs) < abs(rhs);
  };
  // Construct an orthogonal vector v.
  VectorType v = u;
  {
    auto it = std::min_element(v.data(), v.data() + 3, compare_absolute_value);
    *it = Scalar{0};
    auto is_not_it = [&](const Scalar &x) { return &x != &*it; };
    const auto values = genit::FilterRange(
        genit::IteratorRange(v.data(), v.data() + 3), std::cref(is_not_it));
    const auto first = values.begin();
    const auto second = std::next(first);
    std::swap(*first, *second);
    *first = -*first;
    v.normalize();
  }

  VectorType w = u.cross(v).normalized();
  return {v, w};
}

template <class Scalar, int N, int MaxSize>
ScaleVectorResult ScaleVectorToLimits(
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &lower,
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &upper,
    VectorFixedOrDynamic<Scalar, N, MaxSize> *inout) {
  static_assert(std::is_floating_point<Scalar>::value,
                "The operation requires real number arithmetic.");
  constexpr Scalar kTiny = 1e-10;

  CHECK(nullptr != inout);
  CHECK_EQ(upper.rows(), inout->rows());
  CHECK_EQ(lower.rows(), inout->rows());
  const int dim = inout->rows();

  Scalar scale_max = std::numeric_limits<Scalar>::max();
  Scalar scale_min = std::numeric_limits<Scalar>::lowest();
  for (int dof = 0; dof < dim; dof++) {
    Scalar &v = (*inout)[dof];
    const Scalar &u = upper[dof];
    const Scalar &l = lower[dof];
    if (u < l) {  // signal infeasible case below.
      scale_max = std::numeric_limits<Scalar>::lowest();
      scale_min = std::numeric_limits<Scalar>::max();
      break;
    }
    if (std::abs(v) < kTiny) {
      if (u < 0) {
        scale_max = std::min(scale_max, 0.0);
      }
      if (l > 0) {
        scale_min = std::max(scale_min, 0.0);
      }
    } else {
      Scalar scale_u = u / v;
      Scalar scale_l = l / v;
      if (scale_u < scale_l) std::swap(scale_u, scale_l);
      scale_max = std::min(scale_max, scale_u);
      scale_min = std::max(scale_min, scale_l);
    }
  }
  if (scale_max < 1 || scale_min > 1) {
    // add some safety in down-scaling
    constexpr Scalar kSafetyFactor =
        (1.0 - std::numeric_limits<Scalar>::epsilon() * 1e2);
    if (scale_min > scale_max) {
      bool unhandled_case = false;
      for (int dof = 0; dof < dim; dof++) {
        if (upper[dof] < lower[dof]) {
          unhandled_case = true;
          (*inout)[dof] = 0.5 * (lower[dof] + upper[dof]);
          continue;
        }
        (*inout)[dof] = std::clamp((*inout)[dof], lower[dof] * kSafetyFactor,
                                   upper[dof] * kSafetyFactor);
      }
      if (unhandled_case) {
        return ScaleVectorResult::INFEASIBLE;
      }
      return ScaleVectorResult::ELEMENTS_SCALED;
    }
    Scalar scale = 1.0;
    if (scale < scale_min) scale = scale_min;
    if (scale > scale_max) scale = scale_max;
    (*inout) = (*inout) * scale * kSafetyFactor;
    // Saturate to within limits. This makes sure
    // lower <= (inout) <= upper  is strictly maintained.
    // Without it, there can be very slight violations
    // due to floating point issues.
    (*inout) = inout->cwiseMin(upper).cwiseMax(lower);

    return ScaleVectorResult::VECTOR_SCALED;
  }
  return ScaleVectorResult::UNCHANGED;
}

template <class Scalar, int N, int MaxSize>
VectorFixedOrDynamic<Scalar, N, MaxSize> ScaleDownToLimits(
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &vector,
    const VectorFixedOrDynamic<Scalar, N, MaxSize> &symmetric_limits) {
  CHECK(symmetric_limits.minCoeff() > 0.0) << "Limits must be all positive.";
  VectorFixedOrDynamic<Scalar, N, MaxSize> output = vector;
  VectorFixedOrDynamic<Scalar, N, MaxSize> upper = symmetric_limits;
  VectorFixedOrDynamic<Scalar, N, MaxSize> lower = -symmetric_limits;
  ScaleVectorResult result = ScaleVectorToLimits(lower, upper, &output);
  CHECK(result == ScaleVectorResult::VECTOR_SCALED ||
        result == ScaleVectorResult::UNCHANGED);
  return output;
}

// Applies a deadband to `input` and returns the result.
template <class Scalar, int N, int MaxSize>
eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> ApplyDeadband(
    const eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> &input,
    const eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> &band) {
  CHECK_EQ(input.size(), band.size());
  eigenmath::VectorFixedOrDynamic<Scalar, N, MaxSize> result(input.size());
  for (int index = 0; index < input.size(); ++index) {
    if (input[index] > band[index]) {
      result[index] = input[index] - band[index];
    } else if (input[index] < -band[index]) {
      result[index] = input[index] + band[index];
    } else {
      result[index] = 0.0;
    }
  }
  return result;
}
}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_VECTOR_UTILS_H_
