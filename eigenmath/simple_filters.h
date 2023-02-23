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

#ifndef EIGENMATH_EIGENMATH_SIMPLE_FILTERS_H_
#define EIGENMATH_EIGENMATH_SIMPLE_FILTERS_H_

#include <cmath>

#include "types.h"

namespace eigenmath {

// This filter implements a simple infinite impulse response filter. This is
// the simplest form of low-pass filtering which simply adds new samples in a
// weighted average with its current state:
//   s(n+1) = (1 - f) * s(n) + f * x
// where s is the state, f is a constant factor, and x is a new sample.
template <typename T>
class FirstOrderLowPass {
 public:
  // Construct the filter from a desired initial state, a time-step (time-steps
  // must be uniform for this filter to make sense), and a desired cutoff
  // frequency, the smoothing factor will be computed accordingly.
  template <typename U>
  FirstOrderLowPass(U&& initial_state, double time_step,
                    double cutoff_frequency_hz)
      : state_(std::forward<U>(initial_state)),
        smoothing_(cutoff_frequency_hz * time_step * 2.0 * M_PI) {}

  // Resets the filter to a given initial state.
  template <typename U>
  void Reset(U&& initial_state) {
    state_ = std::forward<U>(initial_state);
  }

  // Register a new sample.
  template <typename U>
  void AddSample(U&& new_sample) {
    state_ =
        (1.0 - smoothing_) * state_ + smoothing_ * std::forward<U>(new_sample);
  }

  // Get the current state of the filter.
  const T& GetCurrentState() const { return state_; }

 private:
  T state_;
  double smoothing_;
};

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SIMPLE_FILTERS_H_
