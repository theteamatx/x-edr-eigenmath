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

#ifndef EIGENMATH_EIGENMATH_QUADRATIC_OPTIMIZATION_H_
#define EIGENMATH_EIGENMATH_QUADRATIC_OPTIMIZATION_H_

#include <vector>

#include "types.h"

namespace eigenmath {

// Solves a quadratic minimization problem.
//
// Minimizes  q(x) = 0.5 x'.G.x + x'.c_i  for each column c_i of C separately.
//
// G must be symmetric and factorizable using LDLT Cholesky.
//
// Returns a matrix with columns x_i which minimize q(x) for each column c_i of
// C.
MatrixXd QuadMinimize(const MatrixXd& G, const MatrixXd& C);

// Solves an equality-constrained quadratic minimization problem.
//
// Minimizes  q(x) = 0.5 x'.G.x + x'.c_i  with
//             A.x = b_i
// for each pair of columns c_i in C and b_i in B.
//
// This corresponds to Eq (16.3a) of Nocedal and Wright.
// Solution is obtained using Schur-Complement method. This is useful when:
//   - G is well conditioned and easy to invert
//   - The number of equality constraints (A.rows()) is small
//
// G must be symmetric and factorizable using LDLT Cholesky.
//
// Returns a matrix with columns x_i which minimize q(x) subject to the
// constraints for each pair of columns in C and B.
MatrixXd QuadMinimizeConstrainedSchur(const MatrixXd& G, const MatrixXd& C,
                                      const MatrixXd& A, const MatrixXd& B);

// Representation of a variable substitution, for use in
// QuadMinimizeConstrainedElimination.
struct VariableSubstitution {
  // The index of the variable being substituted.
  int index;

  // The value of the variable being substituted.  Each element corresponds to
  // the substitution for an independent optimization problem.
  VectorXd value;
};

// Solves an equality-constrained quadratic minimization problem.
//
// Minimizes  q(x) = 0.5 x'.G.x + x'.c_i  with
//             A.x = b_i
// for each pair of columns c_i in C and b_i in B.  Here, the constraint is
// specified as a set of variable substitutions.
//
// G must be symmetric and factorizable using LDLT Cholesky.
//
// This function is a more specific form of QuadMinimizeConstrainedSchur.
// It is applicable if each row of A has only a single non-zero coefficient.
// Each substitution is represented as a variable index number and its value.
//
// Returns a matrix which minimizes q(x) subject to the constraints, where each
// column corresponds to a column of C.
MatrixXd QuadMinimizeConstrainedElimination(
    const MatrixXd& G, const MatrixXd& C,
    const std::vector<VariableSubstitution>& substitutions);

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_QUADRATIC_OPTIMIZATION_H_
