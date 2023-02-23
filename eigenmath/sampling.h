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

// Provide tools for sampling in tests.  These are intended to be used in
// combination with the distributions in distribution.h.  The primary goal is to
// provide repeatable samples to avoid deterministic (reproducible, non-flaky)
// tests.
#ifndef EIGENMATH_EIGENMATH_SAMPLING_H_
#define EIGENMATH_EIGENMATH_SAMPLING_H_

#include <random>

namespace eigenmath {

// An (arbitrarily chosen) pseudo-random number generator for use in tests.
using TestGenerator = std::mt19937;
inline constexpr int kGeneratorTestSeed = 43;

}  // namespace eigenmath
#endif  // EIGENMATH_EIGENMATH_SAMPLING_H_
