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

#include "simple_filters.h"

#include <cmath>

#include "gtest/gtest.h"

namespace eigenmath {
namespace {

const double kDefaultTolerance = Eigen::NumTraits<double>::dummy_precision();

TEST(TestFirstOrderLowPass, BasicOperationsScalar) {
  constexpr double kTimeStep = 0.001;
  constexpr double kCutoffFreqHz = 10.0;
  FirstOrderLowPass<double> low_pass_filter(0.0, kTimeStep, kCutoffFreqHz);
  EXPECT_NEAR(low_pass_filter.GetCurrentState(), 0.0, kDefaultTolerance);

  constexpr double kAmplitude = 5.0;

  // Check gain at 1 Hz.
  double test_freq_hz = 1.0;
  low_pass_filter.Reset(0.0);
  EXPECT_NEAR(low_pass_filter.GetCurrentState(), 0.0, kDefaultTolerance);
  for (int i = 0; i < 1.0 / (kTimeStep * test_freq_hz * 4.0); ++i) {
    low_pass_filter.AddSample(
        kAmplitude * std::sin(i * 2.0 * M_PI * kTimeStep * test_freq_hz));
  }
  // Expect less than 0.25dB rejection.
  EXPECT_NEAR(low_pass_filter.GetCurrentState(), kAmplitude, 0.1 * kAmplitude);

  // Check gain at 10 Hz.
  test_freq_hz = 10.0;
  low_pass_filter.Reset(0.0);
  EXPECT_NEAR(low_pass_filter.GetCurrentState(), 0.0, kDefaultTolerance);
  for (int i = 0; i < 1.0 / (kTimeStep * test_freq_hz * 4.0); ++i) {
    low_pass_filter.AddSample(
        kAmplitude * std::sin(i * 2.0 * M_PI * kTimeStep * test_freq_hz));
  }
  // Expect a 2dB to 3dB rejection at the cutoff frequency.
  EXPECT_LT(low_pass_filter.GetCurrentState(), kAmplitude * 0.63);
  EXPECT_GT(low_pass_filter.GetCurrentState(), kAmplitude * 0.5);

  // Check that filter rejects highest frequency signal.
  for (int i = 0; i < 100; ++i) {
    if (i % 2 == 0) {
      low_pass_filter.AddSample(kAmplitude);
    } else {
      low_pass_filter.AddSample(-kAmplitude);
    }
  }
  // Expect a 10dB rejection.
  EXPECT_NEAR(low_pass_filter.GetCurrentState(), 0.0, 0.1 * kAmplitude);
}

}  // namespace
}  // namespace eigenmath
