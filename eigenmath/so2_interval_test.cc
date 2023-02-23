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

#include "so2_interval.h"

#include <ostream>
#include <vector>

#include "absl/random/distributions.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "sampling.h"
#include "so2.h"

namespace eigenmath {

// ADL stream operator for test output.
std::ostream& operator<<(std::ostream& os, const SO2dInterval& interval) {
  if (interval.IsEmpty()) {
    return os << "empty";
  }
  if (interval.IsFullCircle()) {
    return os << "full circle";
  }
  return os << "[" << interval.FromAngle() << ", " << interval.ToAngle() << "]";
}

namespace {
using ::testing::DoubleEq;

// Number of samples for benchmarks.
constexpr int kSamples = 10000;
constexpr double kTolerance = 1e-12;
constexpr double kTwoPi = 2 * M_PI;

MATCHER_P(IntervalEquals, expected, "") {
  if ((arg.IsEmpty() != expected.IsEmpty()) ||
      (arg.IsFullCircle() != expected.IsFullCircle())) {
    return false;
  }
  if (arg.IsEmpty() || arg.IsFullCircle()) {
    return true;
  }

  if (arg.FromAngle() == expected.FromAngle() &&
      arg.ToAngle() == expected.ToAngle()) {
    return true;
  }
  return false;
}

MATCHER_P2(ApproxInterval, expected, tolerance, "") {
  if ((arg.IsEmpty() != expected.IsEmpty()) ||
      (arg.IsFullCircle() != expected.IsFullCircle())) {
    return false;
  }
  if (arg.IsEmpty() || arg.IsFullCircle()) {
    return true;
  }

  if (std::abs(arg.FromAngle() - expected.FromAngle()) < tolerance &&
      std::abs(arg.ToAngle() - expected.ToAngle()) < tolerance) {
    return true;
  }
  return false;
}

TEST(Constructor, DefaultConstructorYieldsEmptyInterval) {
  const SO2dInterval default_constructed;

  EXPECT_THAT(default_constructed, IntervalEquals(SO2dInterval::Empty()));
}

TEST(Constructor, FromAnglesEqualsSO2) {
  const SO2d first(0.1);
  const SO2d second(-0.3);
  const SO2dInterval x{first, second};
  const SO2dInterval y{first.angle(), second.angle()};
  const SO2dInterval z{0.1, -0.3};
  EXPECT_THAT(y, ApproxInterval(z, kTolerance));
  EXPECT_THAT(x, ApproxInterval(y, kTolerance));
}

TEST(Constructor, FromAngularSegmentWithSO2PointsForward) {
  const SO2d first(0.1);
  const SO2d second(-0.3);
  const SO2dInterval x{first, second, 100.0};
  const SO2dInterval y{first, second};
  EXPECT_THAT(x, IntervalEquals(y));
}

TEST(Constructor, FromAngularSegmentWithSO2PointsBackward) {
  const SO2d first(0.1);
  const SO2d second(-0.3);
  const SO2dInterval x{first, second, -100.0};
  const SO2dInterval y{second, first};
  EXPECT_THAT(x, IntervalEquals(y));
}

TEST(Constructor, FromAngularSegmentWithAnglePointsBackward) {
  const double first = 0.1;
  const double second = -0.3;
  const SO2dInterval x{first, second, -100.0};
  const SO2dInterval y{second, first};
  EXPECT_THAT(x, IntervalEquals(y));
}

TEST(IsEmpty, OnlyEmptyIntervalIsEmpty) {
  EXPECT_TRUE(SO2dInterval::Empty().IsEmpty());
  EXPECT_FALSE(SO2dInterval::FullCircle().IsEmpty());
  EXPECT_FALSE(SO2dInterval(1.0, 2.0).IsEmpty());
  EXPECT_FALSE(SO2dInterval(2.0, 1.0).IsEmpty());
}

TEST(IsFullCircle, OnlyFullCircleIsFullCircle) {
  EXPECT_TRUE(SO2dInterval::FullCircle().IsFullCircle());
  EXPECT_FALSE(SO2dInterval::Empty().IsFullCircle());
  EXPECT_FALSE(SO2dInterval(1.0, 2.0).IsFullCircle());
  EXPECT_FALSE(SO2dInterval(2.0, 1.0).IsFullCircle());
}

TEST(Length, EmptyIntervalHasZeroLength) {
  EXPECT_THAT(SO2dInterval::Empty().Length(), DoubleEq(0));
}

TEST(Length, FullCircleHasLengthTwoPi) {
  EXPECT_THAT(SO2dInterval::FullCircle().Length(), DoubleEq(kTwoPi));
}

TEST(Length, LengthYieldsArcLength) {
  EXPECT_THAT(SO2dInterval(1.0, 2.0).Length(), DoubleEq(1.0));
  EXPECT_THAT(SO2dInterval(kTwoPi - 1.0, 1.0).Length(), DoubleEq(2.0));
}

TEST(ContainsPoint, EmptyIntervalDoesNotContainAnyPoint) {
  const SO2dInterval empty;

  for (const double angle : {-10.0, -2.5, 0.0, 1.5, 10.0}) {
    EXPECT_FALSE(empty.Contains(SO2d{angle}, kTolerance));
    EXPECT_FALSE(empty.Contains(angle, kTolerance));
  }
}

TEST(ContainsPoint, FullCircleContainsSampledPoints) {
  const SO2dInterval full_circle = SO2dInterval::FullCircle();

  for (const double angle : {-10.0, -2.5, 0.0, 1.5, 10.0}) {
    EXPECT_TRUE(full_circle.Contains(SO2d{angle}, kTolerance));
    EXPECT_TRUE(full_circle.Contains(angle, kTolerance));
  }
}

TEST(ContainsPoint, IntervalContainsOnlyPointsWithinBounds) {
  const SO2dInterval interval(1.0, 2.0);

  for (const double angle : {-1.0, 0.0, 0.5, 2.5, kTwoPi + 0.5}) {
    EXPECT_FALSE(interval.Contains(SO2d{angle}, kTolerance));
    EXPECT_FALSE(interval.Contains(angle, kTolerance));
  }
  for (const double angle : {1.0, 1.1, 1.5, 1.9, 2.0}) {
    EXPECT_TRUE(interval.Contains(SO2d{angle}, kTolerance));
    EXPECT_TRUE(interval.Contains(angle, kTolerance));
  }
}

TEST(ContainsPoint, ReverseIntervalContainsOnlyPointsWithinBounds) {
  const SO2dInterval interval(2.0, 1.0);

  for (const double angle : {-1.0, 0.0, 0.5, 1.0, 2.0, 2.5, kTwoPi + 1.0}) {
    EXPECT_TRUE(interval.Contains(SO2d{angle}, kTolerance));
    EXPECT_TRUE(interval.Contains(angle, kTolerance));
  }
  for (const double angle : {1.1, 1.5, 1.9}) {
    EXPECT_FALSE(interval.Contains(SO2d{angle}, kTolerance));
    EXPECT_FALSE(interval.Contains(angle, kTolerance));
  }
}

TEST(ContainsInterval, AboutFullIntervalContainsFullCircle) {
  const SO2dInterval about_full(1.0, 1.0 - kTolerance);
  EXPECT_TRUE(about_full.Contains(SO2dInterval::FullCircle(), kTolerance));
}

TEST(ContainsInterval, IntervalContainsPointsWithinTolerance) {
  const SO2dInterval near_bound(kTolerance, 1.0);
  EXPECT_TRUE(near_bound.Contains(0.0, kTolerance));
  EXPECT_TRUE(near_bound.Contains(1.0 + kTolerance, kTolerance));
}

TEST(ContainsInterval, EmptyIntervalContainsNoInterval) {
  const SO2dInterval empty;
  EXPECT_FALSE(empty.Contains(empty, kTolerance));
  EXPECT_FALSE(empty.Contains(SO2dInterval::FullCircle(), kTolerance));
  EXPECT_FALSE(empty.Contains(SO2dInterval{1.0, 2.0}, kTolerance));
  EXPECT_FALSE(empty.Contains(SO2dInterval{2.0, 1.0}, kTolerance));
}

TEST(ContainsInterval, FullCircleContainsAllIntervals) {
  const SO2dInterval full_circle = SO2dInterval::FullCircle();
  EXPECT_TRUE(full_circle.Contains(full_circle, kTolerance));
  EXPECT_TRUE(full_circle.Contains(SO2dInterval::Empty(), kTolerance));
  EXPECT_TRUE(full_circle.Contains(SO2dInterval{1.0, 2.0}, kTolerance));
  EXPECT_TRUE(full_circle.Contains(SO2dInterval{2.0, 1.0}, kTolerance));
}

TEST(ContainsInterval, IntervalContainsOnlySmallerIntervals) {
  const SO2dInterval interval(1.0, 2.0);

  EXPECT_TRUE(interval.Contains(SO2dInterval::Empty(), kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval::FullCircle(), kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{0.1, 0.3}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{0.5, 1.5}, kTolerance));
  EXPECT_TRUE(interval.Contains(SO2dInterval{1.1, 1.5}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{1.5, 2.5}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{2.1, 2.5}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{2.5, 0.5}, kTolerance));
}

TEST(ContainsInterval, ReverseIntervalContainsOnlySmallerIntervals) {
  const SO2dInterval interval(2.0, 1.0);

  EXPECT_TRUE(interval.Contains(SO2dInterval::Empty(), kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval::FullCircle(), kTolerance));
  EXPECT_TRUE(interval.Contains(SO2dInterval{0.1, 0.3}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{0.5, 1.5}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{1.1, 1.5}, kTolerance));
  EXPECT_FALSE(interval.Contains(SO2dInterval{1.5, 2.5}, kTolerance));
  EXPECT_TRUE(interval.Contains(SO2dInterval{2.1, 2.5}, kTolerance));
  EXPECT_TRUE(interval.Contains(SO2dInterval{2.5, 0.5}, kTolerance));
}

TEST(ContainsInterval, AboutFullIntervalContainsIntervalAcrossBounds) {
  const SO2dInterval about_full(1.0, 1.0 - kTolerance);
  EXPECT_TRUE(about_full.Contains(SO2dInterval{0.5, 1.5}, kTolerance));
}

TEST(ContainsInterval, IntervalContainsSlightlyLargerIntervalWithinTolerance) {
  const SO2dInterval near_bound(kTolerance, 1.0);
  EXPECT_TRUE(
      near_bound.Contains(SO2dInterval{-kTolerance, 1.0}, 2 * kTolerance));
}

TEST(ContainsAngularSegment, EmptyIntervalContainsNoAngularSegment) {
  const SO2dInterval empty;
  EXPECT_FALSE(empty.Contains(0.1, 0.3, kTolerance));
  EXPECT_FALSE(empty.Contains(100.0, -20.0, kTolerance));
  EXPECT_FALSE(empty.Contains(-1.0, 20.0, kTolerance));
}

TEST(ContainsAngularSegment, FullIntervalContainsAllAngularSegments) {
  const SO2dInterval full_circle = SO2dInterval::FullCircle();
  EXPECT_TRUE(full_circle.Contains(0.1, 0.3, kTolerance));
  EXPECT_TRUE(full_circle.Contains(100.0, -20.0, kTolerance));
  EXPECT_TRUE(full_circle.Contains(-1.0, 20.0, kTolerance));
}

TEST(ContainsAngularSegment, AboutFullIntervalContainsAllAngularSegments) {
  const SO2dInterval about_full{1.0, 1.0 - kTolerance};
  EXPECT_TRUE(about_full.Contains(0.1, 0.3, kTolerance));
  EXPECT_TRUE(about_full.Contains(100.0, -20.0, kTolerance));
  EXPECT_TRUE(about_full.Contains(-1.0, 20.0, kTolerance));
}

TEST(ContainsAngularSegment, IntervalContainsOnlyContainedSegments) {
  const SO2dInterval interval(1.0, 2.0);
  EXPECT_TRUE(interval.Contains(1.0, 0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.0, -0.3, kTolerance));
  EXPECT_TRUE(interval.Contains(1.8, -0.7, kTolerance));
  EXPECT_FALSE(interval.Contains(1.8, 0.7, kTolerance));

  EXPECT_FALSE(interval.Contains(0.5, 0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(0.5, -0.3, kTolerance));
  EXPECT_TRUE(interval.Contains(1.1, 0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.1, -0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.1, 1.2 + kTwoPi, kTolerance));
}

TEST(ContainsAngularSegment, ReverseIntervalContainsOnlyContainedSegments) {
  const SO2dInterval interval(2.0, 1.0);
  EXPECT_FALSE(interval.Contains(1.0, 0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(2.0, -0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.8, -0.7, kTolerance));
  EXPECT_FALSE(interval.Contains(1.8, 0.7, kTolerance));

  EXPECT_TRUE(interval.Contains(0.5, 0.3, kTolerance));
  EXPECT_TRUE(interval.Contains(0.5, -0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.1, 0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.1, -0.3, kTolerance));
  EXPECT_FALSE(interval.Contains(1.1, 1.2 + kTwoPi, kTolerance));
}

TEST(Intersect, EmptyIntervalIntersectsToEmptyInterval) {
  const SO2dInterval empty;
  EXPECT_THAT(empty.Intersect(empty), IntervalEquals(empty));
  EXPECT_THAT(empty.Intersect(SO2dInterval::FullCircle()),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(empty.Intersect(SO2dInterval{1.0, 2.0}), IntervalEquals(empty));
  EXPECT_THAT(empty.Intersect(SO2dInterval{2.0, 1.0}), IntervalEquals(empty));
}

TEST(Intersect, FullCircleIntersectsToOtherInterval) {
  const SO2dInterval full_circle = SO2dInterval::FullCircle();
  EXPECT_THAT(full_circle.Intersect(full_circle), IntervalEquals(full_circle));
  EXPECT_THAT(full_circle.Intersect(SO2dInterval::Empty()),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(full_circle.Intersect(SO2dInterval{1.0, 2.0}),
              IntervalEquals(SO2dInterval{1.0, 2.0}));
  EXPECT_THAT(full_circle.Intersect(SO2dInterval{2.0, 1.0}),
              IntervalEquals(SO2dInterval{2.0, 1.0}));
}

TEST(Intersect, Interval) {
  const SO2dInterval reference(1.0, 2.0);
  EXPECT_THAT(reference.Intersect(SO2dInterval::FullCircle()),
              IntervalEquals(reference));
  EXPECT_THAT(reference.Intersect(SO2dInterval::Empty()),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(reference.Intersect(reference), IntervalEquals(reference));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.0, 1.0}),
              IntervalEquals(SO2dInterval{1.0}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.0}),
              IntervalEquals(SO2dInterval{1.0}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.0}),
              IntervalEquals(SO2dInterval{2.0}));

  EXPECT_THAT(reference.Intersect(SO2dInterval{0.1, 0.3}),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(reference.Intersect(SO2dInterval{0.5, 1.5}),
              IntervalEquals(SO2dInterval{1.0, 1.5}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.1, 1.5}),
              IntervalEquals(SO2dInterval{1.1, 1.5}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.5, 2.5}),
              IntervalEquals(SO2dInterval{1.5, 2.0}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.1, 2.5}),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.5, 0.5}),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.8, 1.2}),
              IntervalEquals(SO2dInterval{1.0, 1.2}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{0.8, 2.2}),
              IntervalEquals(reference));
}

TEST(Intersect, ReverseInterval) {
  const SO2dInterval reference(2.0, 1.0);
  EXPECT_THAT(reference.Intersect(SO2dInterval::FullCircle()),
              IntervalEquals(reference));
  EXPECT_THAT(reference.Intersect(SO2dInterval::Empty()),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(reference.Intersect(reference), IntervalEquals(reference));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.0, 2.0}),
              IntervalEquals(SO2dInterval{2.0}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.0}),
              IntervalEquals(SO2dInterval{1.0}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.0}),
              IntervalEquals(SO2dInterval{2.0}));

  EXPECT_THAT(reference.Intersect(SO2dInterval{0.1, 0.3}),
              ApproxInterval(SO2dInterval{0.1, 0.3}, kTolerance));
  EXPECT_THAT(reference.Intersect(SO2dInterval{0.5, 1.5}),
              IntervalEquals(SO2dInterval{0.5, 1.0}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.1, 1.5}),
              IntervalEquals(SO2dInterval::Empty()));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.5, 2.5}),
              IntervalEquals(SO2dInterval{2.0, 2.5}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.1, 2.5}),
              IntervalEquals(SO2dInterval{2.1, 2.5}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{2.5, 0.5}),
              IntervalEquals(SO2dInterval{2.5, 0.5}));
  EXPECT_THAT(reference.Intersect(SO2dInterval{1.8, 1.2}),
              IntervalEquals(reference));
  EXPECT_THAT(reference.Intersect(SO2dInterval{0.8, 2.2}),
              IntervalEquals(SO2dInterval{2.0, 2.2}));
}

void BM_ContainsOrientation(benchmark::State& state) {
  TestGenerator generator(kGeneratorTestSeed);
  std::vector<SO2dInterval> intervals;
  intervals.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    intervals.emplace_back(absl::Uniform(generator, -10, 10),
                           absl::Uniform(generator, -10, 10));
  }
  std::vector<SO2d> points(kSamples);
  for (auto& pt : points) {
    pt = SO2d(absl::Uniform(generator, -10, 10));
  }
  int counter = 0;
  for (auto _ : state) {
    int index = ++counter % kSamples;
    benchmark::DoNotOptimize(
        intervals[index].Contains(points[index], kTolerance));
  }
}
BENCHMARK(BM_ContainsOrientation);

void BM_ContainsAngle(benchmark::State& state) {
  TestGenerator generator(kGeneratorTestSeed);
  std::vector<SO2dInterval> intervals;
  intervals.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    intervals.emplace_back(absl::Uniform(generator, -10, 10),
                           absl::Uniform(generator, -10, 10));
  }
  std::vector<double> points(kSamples);
  for (auto& pt : points) {
    pt = absl::Uniform(generator, -10, 10);
  }
  int counter = 0;
  for (auto _ : state) {
    int index = ++counter % kSamples;
    benchmark::DoNotOptimize(
        intervals[index].Contains(points[index], kTolerance));
  }
}
BENCHMARK(BM_ContainsAngle);

void BM_ContainsInterval(benchmark::State& state) {
  TestGenerator generator(kGeneratorTestSeed);
  std::vector<SO2dInterval> intervals;
  intervals.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    intervals.emplace_back(absl::Uniform(generator, -10, 10),
                           absl::Uniform(generator, -10, 10));
  }
  int counter = 0;
  for (auto _ : state) {
    int first = counter % kSamples;
    int second = ++counter % kSamples;
    benchmark::DoNotOptimize(
        intervals[first].Contains(intervals[second], kTolerance));
  }
}
BENCHMARK(BM_ContainsInterval);

void BM_Intersection(benchmark::State& state) {
  TestGenerator generator(kGeneratorTestSeed);
  std::vector<SO2dInterval> intervals;
  intervals.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    intervals.emplace_back(absl::Uniform(generator, -10, 10),
                           absl::Uniform(generator, -10, 10));
  }
  int counter = 0;
  for (auto _ : state) {
    int first = counter % kSamples;
    int second = ++counter % kSamples;
    benchmark::DoNotOptimize(intervals[first].Intersect(intervals[second]));
  }
}
BENCHMARK(BM_Intersection);

}  // namespace
}  // namespace eigenmath
