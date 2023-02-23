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

#include <vector>

#include "Eigen/Cholesky"
#include "Eigen/Core"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"

namespace eigenmath {

MatrixXd QuadMinimize(const MatrixXd& G, const MatrixXd& C) {
  const int num_vars = G.rows();
  CHECK_EQ(G.cols(), num_vars);
  CHECK_EQ(C.rows(), num_vars);

  Eigen::LDLT<MatrixXd> G_solver(G);
  return G_solver.solve(-C);
}

MatrixXd QuadMinimizeConstrainedSchur(const MatrixXd& G, const MatrixXd& C,
                                      const MatrixXd& A, const MatrixXd& B) {
  const int num_vars = G.rows();
  CHECK_EQ(G.cols(), num_vars);
  CHECK_EQ(C.rows(), num_vars);
  const int num_constraints = A.rows();
  if (num_constraints == 0) {
    return QuadMinimize(G, C);
  }
  CHECK_EQ(A.cols(), num_vars);
  CHECK_EQ(B.rows(), num_constraints);
  CHECK_EQ(B.cols(), C.cols());

  Eigen::LDLT<MatrixXd> G_solver(G);
  const MatrixXd Ginv_At = G_solver.solve(A.transpose());
  const MatrixXd Ginv_g = G_solver.solve(C);
  Eigen::LDLT<MatrixXd> schur_solver(A * Ginv_At);
  // Eq (16.13), Nocedal & Wright:
  const MatrixXd lambda = schur_solver.solve(A * Ginv_g + B);
  return G_solver.solve(A.transpose() * lambda - C);
}

MatrixXd QuadMinimizeConstrainedElimination(
    const MatrixXd& G, const MatrixXd& C,
    const std::vector<VariableSubstitution>& substitutions) {
  const int num_orig_vars = G.rows();
  CHECK_EQ(G.cols(), G.rows());
  CHECK_EQ(C.rows(), G.rows());
  const int num_solves = C.cols();

  CHECK_GT(num_orig_vars, static_cast<int>(substitutions.size()))
      << "Need more variables than substitutions.";
  const int num_vars = num_orig_vars - substitutions.size();

  MatrixXd G_new = MatrixXd::Zero(num_vars, num_vars);
  MatrixXd C_new = MatrixXd::Zero(num_vars, num_solves);
  MatrixXd X = MatrixXd::Zero(num_orig_vars, num_solves);
  // Get the set of variables in the new problem by removing each substitution
  // variable from the set.
  std::vector<bool> new_var_set(num_orig_vars, true);
  for (const VariableSubstitution& sub : substitutions) {
    CHECK_LT(sub.index, num_orig_vars)
        << absl::StrFormat("Invalid variable index %d", sub.index);
    X.row(sub.index) = sub.value.transpose();
    new_var_set[sub.index] = false;
  }

  // Aggregate the active variables as a vector of indices.
  std::vector<int> new_vars;
  new_vars.reserve(num_vars);
  for (int var_index = 0; var_index < num_orig_vars; ++var_index) {
    if (new_var_set[var_index]) {
      new_vars.push_back(var_index);
    }
  }
  CHECK_EQ(static_cast<int>(new_vars.size()), num_vars);

  // Construct the new quadratic cost G.
  for (int row = 0; row < num_vars; ++row) {
    for (int col = 0; col < num_vars; ++col) {
      G_new(row, col) = G(new_vars[row], new_vars[col]);
    }
  }

  // Construct the new linear cost C.
  for (int row = 0; row < num_vars; ++row) {
    C_new.row(row) += C.row(new_vars[row]);
    // The terms from the original quad G which have now become linear due to
    // substitution.
    for (const auto& sub : substitutions) {
      C_new.row(row) += sub.value.transpose() * G(new_vars[row], sub.index);
    }
  }

  const MatrixXd X_new = QuadMinimize(G_new, C_new);

  // Fill out the rest of the result rows.
  for (int row = 0; row < num_vars; ++row) {
    X.row(new_vars[row]) = X_new.row(row);
  }
  return X;
}

}  // namespace eigenmath
