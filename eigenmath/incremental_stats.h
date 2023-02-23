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

#ifndef EIGENMATH_EIGENMATH_INCREMENTAL_STATS_H_
#define EIGENMATH_EIGENMATH_INCREMENTAL_STATS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/time/time.h"

namespace eigenmath {

// Computes basic statistics of a sequence of ValueType scalars incrementally.
// ValueType must be an arithmetic type.
// Average and standard deviation are always computed using doubles to avoid
// large errors when incrementally computing the average of integral values.
template <typename ValueType>
class IncrementalStats {
 public:
  IncrementalStats() {
    static_assert(std::is_arithmetic<ValueType>::value == true,
                  "IncrementalStats only works with arithmetic types.");
    Reset();
  }
  // Resets to same state as if newly constructed.
  void Reset();
  // Updates maximum, minimum and average values.
  void Update(const ValueType& value,
              absl::Time timestamp = absl::InfinitePast());
  const ValueType& LastValue() const { return last_; }
  const double& Average() const { return average_; }
  const double StdDeviation() const { return std_deviation_; }
  const ValueType& Min() const { return min_; }
  const ValueType& Max() const { return max_; }
  const uint64_t& NumUpdates() const { return num_updates_; }
  const absl::Time& FirstTimestamp() const { return first_timestamp_; }
  const absl::Time& LastTimestamp() const { return last_timestamp_; }

 private:
  ValueType last_;
  double average_;
  double std_deviation_;
  double sum_diffs_squared_;
  ValueType max_;
  ValueType min_;
  uint64_t num_updates_;
  absl::Time first_timestamp_;
  absl::Time last_timestamp_;
};

template <typename ValueType>
void IncrementalStats<ValueType>::Reset() {
  last_ = 0;
  average_ = 0.0;
  std_deviation_ = 0.0;
  sum_diffs_squared_ = 0.0;
  max_ = std::numeric_limits<ValueType>::lowest();
  min_ = std::numeric_limits<ValueType>::max();
  num_updates_ = 0;
  first_timestamp_ = absl::InfinitePast();
  last_timestamp_ = absl::InfinitePast();
}

template <typename ValueType>
void IncrementalStats<ValueType>::Update(const ValueType& value,
                                         absl::Time timestamp) {
  // We don't really need to save this value, but having it is often convenient.
  last_ = value;
  max_ = std::max(max_, last_);
  min_ = std::min(min_, last_);
  last_timestamp_ = timestamp;
  if (num_updates_ == 0) first_timestamp_ = last_timestamp_;
  num_updates_++;
  const double diff_from_average = static_cast<double>(value) - average_;

  average_ += diff_from_average / static_cast<double>(num_updates_);
  // Incremental update of standard deviation using Welford's algorithm.
  const double diff_from_new_average = static_cast<double>(value) - average_;

  sum_diffs_squared_ += diff_from_average * diff_from_new_average;
  std_deviation_ = sum_diffs_squared_;
  if (num_updates_ > 1) {
    std_deviation_ /= (num_updates_ - 1);
  }
  std_deviation_ = std::sqrt(std_deviation_);
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_INCREMENTAL_STATS_H_
