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

#ifndef EIGENMATH_EIGENMATH_COMPARE_VECTORS_H_
#define EIGENMATH_EIGENMATH_COMPARE_VECTORS_H_

#include <algorithm>

#include "absl/types/span.h"

namespace eigenmath {

// Returns true if all vectors in vectors_a and vectors_b are equal.
template <typename Vector>
bool VectorsAreEqual(absl::Span<const Vector> vectors_a,
                     absl::Span<const Vector> vectors_b) {
  if (vectors_a.size() != vectors_b.size()) {
    return false;
  }
  auto mismatch_pair =
      std::mismatch(vectors_a.begin(), vectors_a.end(), vectors_b.begin(),
                    [](const Vector& v, const Vector& w) {
                      return (v.rows() == w.rows()) && (v.cols() == v.cols()) &&
                             ((v - w).norm() == 0.0);
                    });
  return mismatch_pair.first == vectors_a.end();
}

// Returns true if all vectors have the expected size.
template <typename Vector>
bool VectorsHaveExpectedSize(absl::Span<const Vector> vectors,
                             int expected_size) {
  for (const auto& p : vectors) {
    if (p.rows() != expected_size) {
      return false;
    }
  }
  return true;
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_COMPARE_VECTORS_H_
