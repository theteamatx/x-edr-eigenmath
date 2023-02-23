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

#include "quadratic_optimization.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "sampling.h"
#include "types.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

// Converts simple substitution constraints to matrix format.
// Returns A, B.
std::tuple<MatrixXd, MatrixXd> SimpleToMatrixConstraints(
    const std::vector<VariableSubstitution>& substitutions, int num_vars) {
  CHECK(!substitutions.empty()) << "Need at least one substitution";
  int num_solves = substitutions[0].value.size();
  int num_rows = substitutions.size();

  MatrixXd A = MatrixXd::Zero(num_rows, num_vars);
  MatrixXd B = MatrixXd::Zero(num_rows, num_solves);

  int row = 0;
  for (const auto& sub : substitutions) {
    CHECK_GE(sub.index, 0) << absl::StrFormat("Invalid variable index %d",
                                              sub.index);
    CHECK_LT(sub.index, num_vars)
        << absl::StrFormat("Invalid variable index %d", sub.index);
    CHECK_EQ(sub.value.size(), num_solves)
        << "Incorrect number of values in substitution";
    A(row, sub.index) = 1.0;
    B.row(row) = sub.value.transpose();
    ++row;
  }
  return std::make_tuple(A, B);
}

// Basic test of unconstrained minimization for some known problems.
TEST(TestQuadratic, QuadMinimizeSmoke) {
  MatrixXd G(2, 2);
  MatrixXd C(2, 1);
  MatrixXd X_expected(2, 1);
  MatrixXd X;

  G << 1.0, 0.0, 0.0, 1.0;
  C << 0.0, 0.0;
  X_expected << 0.0, 0.0;
  X = QuadMinimize(G, C);
  EXPECT_TRUE(X.isApprox(X_expected));

  G << 1.0, 0.0, 0.0, 1.0;
  C << 1.0, 0.0;
  X_expected << -1.0, 0.0;
  X = QuadMinimize(G, C);
  EXPECT_TRUE(X.isApprox(X_expected));

  G << 1.0, 0.0, 0.0, 2.0;
  C << 0.0, -1.0;
  X_expected << 0.0, 0.5;
  X = QuadMinimize(G, C);
  EXPECT_TRUE(X.isApprox(X_expected));
}

// Basic test of constrained minimization for a known problem.
TEST(TestQuadratic, QuadMinimizeConstrainedSmoke) {
  MatrixXd G(2, 2);
  MatrixXd C(2, 1);
  MatrixXd X_expected(2, 1);
  MatrixXd XSchur;
  MatrixXd X_subst;
  MatrixXd A(1, 2);
  MatrixXd B(1, 1);
  std::vector<VariableSubstitution> subst;

  G << 1.0, 0.0, 0.0, 2.0;
  C << 0.0, -1.0;
  B << 1.0;
  subst.push_back({0, B});
  X_expected << 1.0, 0.5;
  std::tie(A, B) = SimpleToMatrixConstraints(subst, 2);
  XSchur = QuadMinimizeConstrainedSchur(G, C, A, B);
  X_subst = QuadMinimizeConstrainedElimination(G, C, subst);
  EXPECT_TRUE(XSchur.isApprox(X_expected));
  EXPECT_TRUE(X_subst.isApprox(X_expected));
}

class MatrixDistribution {
 public:
  using result_type = MatrixXd;

  MatrixDistribution() = default;

  template <typename Generator>
  result_type operator()(int rows, int cols, Generator& generator) const {
    result_type sample(rows, cols);
    constexpr double kZero{0};
    constexpr double kOne{1};
    std::uniform_real_distribution<double> interval(kZero, kOne);
    std::generate_n(sample.data(), rows * cols,
                    [&]() { return interval(generator); });
    return sample;  // NRVO
  }
};

// Generates random constrained quadratic problems and compares the output of
// both algorithms.
TEST(TestQuadratic, CompareQuadMinimizeConstrainedMethodsOnSamples) {
  TestGenerator generator(kGeneratorTestSeed);
  MatrixDistribution matrix_dist;
  const int num_tests = 100;
  const int num_solves = 2;
  for (int num_vars : {2, 5, 10, 25, 50, 100}) {
    int num_constraints = num_vars / 2;
    for (int i = 0; i < num_tests; ++i) {
      const MatrixXd R = matrix_dist(num_vars, num_vars, generator);
      // Ensures that G is positive semi-definite.
      const MatrixXd G = R.transpose() * R;
      const MatrixXd C = matrix_dist(num_vars, num_solves, generator);
      std::vector<VariableSubstitution> subst;
      std::vector<int> var_indices(num_vars);
      std::iota(var_indices.begin(), var_indices.end(), 0);
      std::shuffle(var_indices.begin(), var_indices.end(), generator);
      subst.reserve(num_constraints);
      for (int j = 0; j < num_constraints; ++j) {
        subst.push_back(
            {var_indices[j], matrix_dist(num_solves, 1, generator)});
      }
      const auto [A, B] = SimpleToMatrixConstraints(subst, num_vars);

      const MatrixXd schur_solution = QuadMinimizeConstrainedSchur(G, C, A, B);
      const MatrixXd elimination_solution =
          QuadMinimizeConstrainedElimination(G, C, subst);
      EXPECT_THAT(elimination_solution, IsApprox(schur_solution, 1e-6));
    }
  }
}

void BM_SolveWithSchurComplement(benchmark::State& state) {
  TestGenerator generator(kGeneratorTestSeed);
  MatrixDistribution matrix_dist;
  const int num_tests = 100;
  const int num_solves = 2;
  const int num_vars = state.range(0);

  // First construct all the tests before profiling.
  std::vector<MatrixXd> Gs;
  std::vector<MatrixXd> Cs;
  std::vector<MatrixXd> As;
  std::vector<MatrixXd> Bs;
  std::vector<std::vector<VariableSubstitution>> substs;
  int num_constraints = num_vars / 2;
  for (int i = 0; i < num_tests; ++i) {
    MatrixXd R = matrix_dist(num_vars, num_vars, generator);
    // Ensures that G is positive semi-definite.
    MatrixXd G = R.transpose() * R;
    MatrixXd C = matrix_dist(num_vars, num_solves, generator);
    std::vector<VariableSubstitution> subst;
    std::vector<int> var_indices(num_vars);
    std::iota(var_indices.begin(), var_indices.end(), 0);
    std::shuffle(var_indices.begin(), var_indices.end(), generator);
    subst.reserve(num_constraints);
    for (int j = 0; j < num_constraints; ++j) {
      subst.push_back({var_indices[j], matrix_dist(num_solves, 1, generator)});
    }
    const auto [A, B] = SimpleToMatrixConstraints(subst, num_vars);
    Gs.push_back(G);
    Cs.push_back(C);
    As.push_back(A);
    Bs.push_back(B);
    substs.push_back(subst);
  }

  int i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        QuadMinimizeConstrainedSchur(Gs[i], Cs[i], As[i], Bs[i]));
    ++i;
    if (i >= num_tests) {
      i = 0;
    }
  }
}
BENCHMARK(BM_SolveWithSchurComplement)->Range(2, 100);

void BM_SolveWithElimination(benchmark::State& state) {
  TestGenerator generator(kGeneratorTestSeed);
  MatrixDistribution matrix_dist;
  const int num_tests = 100;
  const int num_solves = 2;
  const int num_vars = state.range(0);

  // First construct all the tests before profiling.
  std::vector<MatrixXd> Gs;
  std::vector<MatrixXd> Cs;
  std::vector<MatrixXd> As;
  std::vector<MatrixXd> Bs;
  std::vector<std::vector<VariableSubstitution>> substs;
  int num_constraints = num_vars / 2;
  for (int i = 0; i < num_tests; ++i) {
    MatrixXd R = matrix_dist(num_vars, num_vars, generator);
    // Ensures that G is positive semi-definite.
    MatrixXd G = R.transpose() * R;
    MatrixXd C = matrix_dist(num_vars, num_solves, generator);
    std::vector<VariableSubstitution> subst;
    std::vector<int> var_indices(num_vars);
    std::iota(var_indices.begin(), var_indices.end(), 0);
    std::shuffle(var_indices.begin(), var_indices.end(), generator);
    subst.reserve(num_constraints);
    for (int j = 0; j < num_constraints; ++j) {
      subst.push_back({var_indices[j], matrix_dist(num_solves, 1, generator)});
    }
    const auto [A, B] = SimpleToMatrixConstraints(subst, num_vars);
    Gs.push_back(G);
    Cs.push_back(C);
    As.push_back(A);
    Bs.push_back(B);
    substs.push_back(subst);
  }

  int i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        QuadMinimizeConstrainedElimination(Gs[i], Cs[i], substs[i]));
    ++i;
    if (i >= num_tests) {
      i = 0;
    }
  }
}
BENCHMARK(BM_SolveWithElimination)->Range(2, 100);

}  // namespace
}  // namespace eigenmath
