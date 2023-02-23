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

#include "quasi_random_vector.h"

#include <algorithm>

#include "absl/log/check.h"
#include "halton_sequence.h"
#include "prime.h"

namespace eigenmath {

eigenmath::VectorXd GetQuasiRandomVector(const eigenmath::VectorXd &lower,
                                         const eigenmath::VectorXd &upper,
                                         int *halton_sequence_index) {
  CHECK_EQ(lower.size(), upper.size());
  CHECK_NE(halton_sequence_index, nullptr);

  // Get a reasonable starting index to avoid obvious linear correlation
  // between the higher dimensional joints in the first few samples (see
  // https://en.wikipedia.org/wiki/Halton_sequence).  This is done
  // heuristically, by dropping the first prime(n) samples for n joints.
  const int min_halton_index = eigenmath::LookupPrime(lower.size());
  const int sequence_index = std::max(min_halton_index, *halton_sequence_index);
  *halton_sequence_index = sequence_index + 1;
  eigenmath::VectorXd random_q(lower.size());
  for (std::size_t i = 0; i < lower.size(); ++i) {
    const double r =
        eigenmath::HaltonSequence(sequence_index, eigenmath::LookupPrime(i));
    random_q[i] = r * (upper[i] - lower[i]) + lower[i];
  }

  return random_q;
}

QuasiRandomVectorGenerator::QuasiRandomVectorGenerator(
    const eigenmath::VectorXd &lower, const eigenmath::VectorXd &upper,
    int *initial_halton_sequence_index)
    : lower_(lower), upper_(upper) {
  CHECK_EQ(lower.size(), upper.size());
  if (initial_halton_sequence_index != nullptr) {
    halton_sequence_index_ = *initial_halton_sequence_index;
  }
}

eigenmath::VectorXd QuasiRandomVectorGenerator::operator()() {
  return GetQuasiRandomVector(lower_, upper_, &halton_sequence_index_);
}

}  // namespace eigenmath
