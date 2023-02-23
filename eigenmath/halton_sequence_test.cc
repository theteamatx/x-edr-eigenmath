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

#include "halton_sequence.h"

#include "gtest/gtest.h"

namespace eigenmath {
namespace {

TEST(HaltonSequenceEngine, Base2) {
  HaltonSequenceEngine<> h;
  double seq[] = {1.0 / 2.0, 1.0 / 4.0, 3.0 / 4.0, 1.0 / 8.0,
                  5.0 / 8.0, 3.0 / 8.0, 7.0 / 8.0};
  const int seq_len = sizeof(seq) / sizeof(*seq);
  for (int i = 0; i < seq_len; ++i) {
    EXPECT_DOUBLE_EQ(h(i), seq[i]);
  }
}

TEST(HaltonSequenceEngine, Base3) {
  HaltonSequenceEngine<> h(HaltonSequenceEngine<>::param_type(1));
  double seq[] = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0, 4.0 / 9.0,
                  7.0 / 9.0, 2.0 / 9.0, 5.0 / 9.0, 8.0 / 9.0};
  const int seq_len = sizeof(seq) / sizeof(*seq);
  for (int i = 0; i < seq_len; ++i) {
    EXPECT_DOUBLE_EQ(h(), seq[i]);
  }
}

TEST(HaltonSequenceEngine, Range) {
  for (int base = 0; base <= 14; ++base) {
    HaltonSequenceEngine<> h;
    h.SeedWithNthPrime(base);
    for (int i = 0; i < 100; ++i) {
      double x = h();
      // If this fails then the value generated with this
      // base is out of the range 0 to 1.
      EXPECT_TRUE(x > 0 && x < 1);
    }
  }
}

TEST(HaltonPointEngine, Range) {
  HaltonPointEngine<VectorXd> hp(2);

  const double seq0[] = {1.0 / 2.0, 1.0 / 4.0, 3.0 / 4.0, 1.0 / 8.0,
                         5.0 / 8.0, 3.0 / 8.0, 7.0 / 8.0};
  const double seq1[] = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0, 4.0 / 9.0,
                         7.0 / 9.0, 2.0 / 9.0, 5.0 / 9.0, 8.0 / 9.0};
  const int seq_len = sizeof(seq0) / sizeof(*seq0);
  for (int i = 0; i < seq_len; ++i) {
    VectorXd pt = hp();
    EXPECT_DOUBLE_EQ(pt[0], seq0[i]);
    EXPECT_DOUBLE_EQ(pt[1], seq1[i]);
  }
}

}  // namespace
}  // namespace eigenmath
