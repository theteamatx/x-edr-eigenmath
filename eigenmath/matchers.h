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

#ifndef EIGENMATH_EIGENMATH_MATCHERS_H_
#define EIGENMATH_EIGENMATH_MATCHERS_H_

#include <type_traits>
#include <utility>

#include "Eigen/Core"
#include "genit/iterator_range.h"
#include "genit/transform_iterator.h"
#include "gmock/gmock.h"
#include "pose2.h"
#include "pose3.h"
#include "type_checks.h"
#include "types.h"

namespace eigenmath {
namespace testing {
namespace matchers_internal {
// Return whether the type T has a difference operator.
template <typename T, typename = void>
static constexpr bool HasDifferenceOp = false;
template <typename T>
static constexpr bool HasDifferenceOp<
    T, std::void_t<decltype(std::declval<T>() - std::declval<T>())>> = true;

template <typename ArgType, typename ExpectedType, typename Scalar>
bool GenericIsApprox(ArgType&& arg, ExpectedType&& expected,
                     Scalar first_tolerance, Scalar second_tolerance,
                     ::testing::MatchResultListener* result_listener) {
  using TestType = std::decay_t<ArgType>;
  if constexpr (IsPose<TestType>) {
    if (expected.isApprox(arg, first_tolerance, second_tolerance)) {
      return true;
    }
    decltype(arg) delta = arg * expected.inverse();
    *result_listener << "\n difference:\n" << delta;
    if constexpr (std::decay_t<
                      decltype(delta.translation())>::RowsAtCompileTime == 2) {
      const decltype(arg.translation()) translation_diff =
          (arg.translation() - expected.translation()).eval();
      *result_listener << "\n translation difference = "
                       << translation_diff.transpose()
                       << " (norm = " << translation_diff.norm() << " m)";
      const auto so2_diff = (expected.so2().inverse() * arg.so2());
      *result_listener << "\n rotation difference = " << so2_diff
                       << " (norm = " << so2_diff.norm() << " rad)";
    } else {
      const decltype(arg.translation()) translation_diff =
          (arg.translation() - expected.translation()).eval();
      *result_listener << "\n translation difference = "
                       << translation_diff.transpose()
                       << " (norm = " << translation_diff.norm() << " m)";
      const auto so3_diff = (expected.so3().inverse() * arg.so3());
      *result_listener << "\n rotation difference = " << so3_diff
                       << " (norm = " << so3_diff.norm() << " rad)";
    }
    return false;
  } else if constexpr (IsQuaternion<TestType>) {
    const TestType relative_quaternion = arg.inverse() * expected;
    if (relative_quaternion.isApprox(TestType::Identity(), first_tolerance)) {
      return true;
    }
    *result_listener << "relative quaternion: " << relative_quaternion;
    *result_listener << ", tolerance= " << first_tolerance;
    return false;
  } else if constexpr (HasDifferenceOp<TestType>) {
    // Ensure that a comparison against a zero matrix/vector does not fail. This
    // is, because isApprox() compares (arg - expected) to the minimum norm of
    // arg and expected, which is 0 in case of a zero vector/matrix.
    if ((arg - expected).cwiseAbs().maxCoeff() < first_tolerance) return true;
    return expected.isApprox(arg, first_tolerance);
  } else {
    return expected.isApprox(arg, first_tolerance);
  }
}
}  // namespace matchers_internal

// Returns true if the arg matches the given expected value within the given
// tolerance.
//
// Usage:
// const double kApproxTolerance = 1e-12;
// eigenmath::Pose3d actual_world_t_target;
// eigenmath::Pose3d expected_world_t_target;
//
// EXPECT_THAT(actual_world_t_target,
//             testing::IsApprox(expected_world_t_target, kApproxTolerance));
MATCHER_P2(IsApprox, expected, tolerance,
           "is approximately equal to:\n" + ::testing::PrintToString(expected) +
               "\n with tolerance " + ::testing::PrintToString(tolerance)) {
  return matchers_internal::GenericIsApprox(arg, expected, tolerance, tolerance,
                                            result_listener);
}

// Returns true if the arg matches the given expected value within the default
// tolerance.
//
// Usage:
// eigenmath::Pose3d actual_world_t_target;
// eigenmath::Pose3d expected_world_t_target;
//
// EXPECT_THAT(actual_world_t_target,
//             testing::IsApprox(expected_world_t_target));
MATCHER_P(IsApprox, expected,
          "is approximately equal to\n" + ::testing::PrintToString(expected)) {
  using Scalar = ScalarTypeOf<std::decay_t<decltype(arg)>>;
  return matchers_internal::GenericIsApprox(
      arg, expected, Eigen::NumTraits<Scalar>::dummy_precision(),
      Eigen::NumTraits<Scalar>::dummy_precision(), result_listener);
}

template <typename MeanAndCovarianceType>
auto IsApproxMeanAndCovariance(const MeanAndCovarianceType& expected) {
  return ::testing::AllOf(
      ::testing::Field("mean", &MeanAndCovarianceType::mean,
                       IsApprox(expected.mean)),
      ::testing::Field("covariance", &MeanAndCovarianceType::covariance,
                       IsApprox(expected.covariance)));
}

template <typename Func, typename Container>
auto TransformRangeToVector(Func&& f, const Container& orig) {
  using TransformedType = decltype(f(orig.front()));
  return genit::CopyRange<std::vector<TransformedType>>(
      genit::TransformRange(orig, std::cref(f)));
}

// Returns a matcher that checks if some actual range matches the expected
// range of values using the IsApprox matcher for each element.
template <typename Range>
auto ElementsAreApprox(Range&& expected, double tolerance) {
  return ::testing::ElementsAreArray(TransformRangeToVector(
      [tolerance](const auto& elem) { return IsApprox(elem, tolerance); },
      expected));
}

template <typename Range>
auto UnorderedElementsAreApprox(Range&& expected, double tolerance) {
  return ::testing::UnorderedElementsAreArray(TransformRangeToVector(
      [tolerance](const auto& elem) { return IsApprox(elem, tolerance); },
      expected));
}

// Returns true if the arg is approximately the same eigenvector as the given
// expected vector within the given tolerance.
//
// What makes two vectors the same eigenvector is if they:
//  - have the same magnitude
//  - are colinear
// In other words, a flip in sign of all components is allowed.
//
// Usage:
// const double kApproxTolerance = 1e-12;
// eigenmath::Vector3d actual_eigenvector;
// eigenmath::Vector3d expected_eigenvector;
//
// EXPECT_THAT(actual_eigenvector,
//     testing::IsApproxEigenVector(expected_eigenvector, kApproxTolerance));
MATCHER_P2(IsApproxEigenVector, expected, tolerance,
           "is approximately the same eigenvector as\n" +
               ::testing::PrintToString(expected) + " with tolerance " +
               ::testing::PrintToString(tolerance)) {
  if (arg.dot(expected) < 0.0) {
    return expected.isApprox(-arg, tolerance);
  } else {
    return expected.isApprox(arg, tolerance);
  }
}

// Returns true if the two-tuple arg's members match each other within the given
// tolerance.
//
// This is particularly useful for matching the contents of collections:
//
// const double kApproxTolerance = 1e-12;
// std::vector<eigenmath::Pose3d> actual_collection;
// std::vector<eigenmath::Pose3d> expected_collection;
//
// EXPECT_THAT(actual_collection,
//   ::testing::Pointwise(
//     ::IsApproxTuple(kApproxTolerance),
//     expected_collection))

MATCHER_P(IsApproxTuple, tolerance, "") {
  return matchers_internal::GenericIsApprox(std::get<0>(arg), std::get<1>(arg),
                                            tolerance, tolerance,
                                            result_listener);
}

// Returns true if the two-tuple arg's members match each other within the given
// tolerance.
//
// This is particularly useful for matching the contents of collections:
//
//  std::vector<eigenmath::Pose3d> actual_collection;
//  std::vector<eigenmath::Pose3d> expected_collection;
//
//  EXPECT_THAT(actual_collection,
//    ::testing::Pointwise(
//      ::IsApproxTuple(), expected_collection));
MATCHER(IsApproxTuple, "") {
  using Scalar = ScalarTypeOf<std::decay_t<decltype(std::get<0>(arg))>>;
  return matchers_internal::GenericIsApprox(
      std::get<0>(arg), std::get<1>(arg),
      Eigen::NumTraits<Scalar>::dummy_precision(),
      Eigen::NumTraits<Scalar>::dummy_precision(), result_listener);
}

// Matcher function to compare two poses. Two thresholds are provided, the first
// one determines the threshold for the norm of the delta translation. The
// second one determines the threshold of the delta absolute angle.
//
// eigenmath::Pose3d a;
// eigenmath::Pose3d b;
// double threshold_translation = 0.5;
// double threshold_angle = 0.4;
//
// EXPECT_THAT(a, testing::IsApprox(b,threshold_translation, threshold_angle));
MATCHER_P3(IsApprox, expected, threshold_norm_translation, threshold_angle,
           std::string(negation ? "isn't" : "is") +
               " approximately equal to:\n" +
               ::testing::PrintToString(expected) + "\n with tolerance " +
               ::testing::PrintToString(threshold_norm_translation) + "m and " +
               ::testing::PrintToString(threshold_angle) + "rad") {
  return matchers_internal::GenericIsApprox(arg, expected,
                                            threshold_norm_translation,
                                            threshold_angle, result_listener);
}

// Matches two poses that are part of a collection. This is useful in cases
// where IsApprox doesn't fit. For example:
//
// std::map<int, eigenmath::Pose3d> a, b;
// EXPECT_THAT(a, UnorderedPointwise(FieldPairsAre(Eq(), ApproxEq()), b));
MATCHER(ApproxEq, "") {
  return ::testing::Value(::testing::get<0>(arg),
                          IsApprox(::testing::get<1>(arg)));
}

// Matches two undirected line segments (allows swapped endpoints).
MATCHER_P(IsApproxUndirected, tolerance,
          "has endpoints approximately equal to with tolerance " +
              ::testing::PrintToString(tolerance)) {
  using std::get;
  return (get<1>(arg).from.isApprox(get<0>(arg).from, tolerance) &&
          get<1>(arg).to.isApprox(get<0>(arg).to, tolerance)) ||
         (get<1>(arg).to.isApprox(get<0>(arg).from, tolerance) &&
          get<1>(arg).from.isApprox(get<0>(arg).to, tolerance));
}
template <typename T>
auto IsApproxUndirected(T&& expected, double tolerance)
    -> decltype(::testing::internal::MatcherBindSecond(
        IsApproxUndirected(tolerance), std::forward<T>(expected))) {
  return ::testing::internal::MatcherBindSecond(IsApproxUndirected(tolerance),
                                                std::forward<T>(expected));
}

// Matches two directed line segments.
MATCHER_P(IsApproxDirected, tolerance,
          "has endpoints approximately equal to with tolerance " +
              ::testing::PrintToString(tolerance)) {
  using std::get;
  return (get<1>(arg).from.isApprox(get<0>(arg).from, tolerance) &&
          get<1>(arg).to.isApprox(get<0>(arg).to, tolerance));
}
template <typename T>
auto IsApproxDirected(T&& expected, double tolerance)
    -> decltype(::testing::internal::MatcherBindSecond(
        IsApproxDirected(tolerance), std::forward<T>(expected))) {
  return ::testing::internal::MatcherBindSecond(IsApproxDirected(tolerance),
                                                std::forward<T>(expected));
}

}  // namespace testing
}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_MATCHERS_H_
