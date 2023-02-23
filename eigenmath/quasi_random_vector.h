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

#ifndef EIGENMATH_EIGENMATH_QUASI_RANDOM_VECTOR_H_
#define EIGENMATH_EIGENMATH_QUASI_RANDOM_VECTOR_H_

#include "types.h"

namespace eigenmath {

// Gets a quasirandom joint configuration (using a Halton Sequence) within the
// given joint limits.
// See the following paper for details:
// Deterministic Sampling-Based Motion Planning: Optimality, Complexity, and
// Performance (2016) Lucas Janson et. al. https://arxiv.org/abs/1505.00023
// lower: the lower limits for the vector.
// upper: the upper limits for the vector.
// halton_sequence_index: the seed of the random number generator.
// Returns a deterministic quasirandom vector  within the given limits.
eigenmath::VectorXd GetQuasiRandomVector(const eigenmath::VectorXd &lower,
                                         const eigenmath::VectorXd &upper,
                                         int *halton_sequence_index);

// A wrapper for GetQuasiRandomVector that stores limits and the
// halton-sequence-index.
class QuasiRandomVectorGenerator {
 public:
  // Constructs a `QuasiRandomVectorGenerator`. Vectors returned by the call
  // operator are within the range [lower, upper].
  // If provided, `initial_halton_sequence_index` is used as the first index
  // Halton sequence index. if not provided, zero is used.
  QuasiRandomVectorGenerator(const eigenmath::VectorXd &lower,
                             const eigenmath::VectorXd &upper,
                             int *initial_halton_sequence_index = nullptr);
  QuasiRandomVectorGenerator() = delete;

  // Returns the next quasi-random vector in the sequence.
  eigenmath::VectorXd operator()();

 private:
  eigenmath::VectorXd lower_;
  eigenmath::VectorXd upper_;
  int halton_sequence_index_ = 0;
};

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_QUASI_RANDOM_VECTOR_H_
