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

#include "incremental_stats.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <valarray>

#include "absl/time/clock.h"
#include "gtest/gtest.h"

namespace eigenmath {
namespace {

template <typename Traits>
class IncrementalStatsTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<double, float, uint8_t, int8_t, uint16_t, int16_t, int32_t,
                     uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(IncrementalStatsTest, ScalarTypes);

TYPED_TEST(IncrementalStatsTest, StatisticsCorrect) {
  IncrementalStats<TypeParam> stats;

  // Test three cases: positive and negative values, only positive values, only
  // negative values.  Allow narrowing for integer types.
  const std::vector<std::vector<TypeParam>> datasets = {
      {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0),
       static_cast<TypeParam>(-1.3), static_cast<TypeParam>(5.5)},
      {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0),
       static_cast<TypeParam>(1.3), static_cast<TypeParam>(5.5)},
      {static_cast<TypeParam>(-1.0), static_cast<TypeParam>(-2.0),
       static_cast<TypeParam>(-1.3), static_cast<TypeParam>(-5.5)}};

  for (const auto& values : datasets) {
    // Verify statistics are computed correctly.
    std::valarray<double> double_values(values.size());
    std::copy(values.begin(), values.end(), std::begin(double_values));
    auto expected_average = double_values.sum() / double_values.size();
    auto expected_min = double_values.min();
    auto expected_max = double_values.max();
    auto expected_std_deviation =
        std::sqrt((std::pow(double_values - expected_average, 2.0).sum() -
                   std::pow((double_values - expected_average).sum(), 2.0) /
                       double_values.size()) /
                  (double_values.size() - 1.0));
    for (TypeParam v : values) {
      stats.Update(v);
    }
    EXPECT_DOUBLE_EQ(stats.Average(), expected_average);
    EXPECT_DOUBLE_EQ(stats.StdDeviation(), expected_std_deviation);
    EXPECT_DOUBLE_EQ(stats.LastValue(), values[values.size() - 1]);
    EXPECT_DOUBLE_EQ(stats.Max(), expected_max);
    EXPECT_DOUBLE_EQ(stats.Min(), expected_min);
    EXPECT_DOUBLE_EQ(stats.NumUpdates(), values.size());

    // Verify Reset works.
    stats.Reset();
    EXPECT_DOUBLE_EQ(stats.Average(), 0);
    EXPECT_DOUBLE_EQ(stats.LastValue(), 0);
    EXPECT_EQ(stats.Max(), std::numeric_limits<TypeParam>::lowest());
    EXPECT_EQ(stats.Min(), std::numeric_limits<TypeParam>::max());
    EXPECT_DOUBLE_EQ(stats.NumUpdates(), 0);
  }
}

TEST(IncrementalStatsTimeTest, StatsTimingCorrect) {
  IncrementalStats<double> stats;
  // Check init to invalid time.
  EXPECT_EQ(stats.FirstTimestamp(), absl::InfinitePast());
  EXPECT_EQ(stats.LastTimestamp(), absl::InfinitePast());
  // Check first time = last time on first update.
  absl::SleepFor(absl::Seconds(1));
  stats.Update(1, absl::Now());
  EXPECT_EQ(stats.FirstTimestamp(), stats.LastTimestamp());
  EXPECT_GE(stats.FirstTimestamp(), absl::InfinitePast());

  // Check last time != first time on subsequent update..
  absl::SleepFor(absl::Seconds(1));
  stats.Update(1, absl::Now());
  EXPECT_GT(stats.LastTimestamp(), stats.FirstTimestamp());

  // Check reset.
  stats.Reset();
  EXPECT_EQ(stats.FirstTimestamp(), absl::InfinitePast());
  EXPECT_EQ(stats.LastTimestamp(), absl::InfinitePast());
}

}  // namespace
}  // namespace eigenmath
