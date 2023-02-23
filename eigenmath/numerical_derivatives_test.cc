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

#include "numerical_derivatives.h"

#include <cmath>
#include <random>

#include "distribution.h"
#include "gtest/gtest.h"
#include "matchers.h"
#include "sampling.h"
#include "so2.h"
#include "utils.h"

namespace eigenmath {
namespace {

using testing::IsApprox;

Vector2d Transform(const Matrix<double, 2, 2>& S, const Vector3d& x) {
  return S * Project(x);
}

Matrix2d TransformDerivativeS(const Matrix2d& S, const Vector3d& x) {
  Matrix2d G = Matrix2d::Zero();
  G << 0, -1, 1, 0;
  Vector2d proj_x = Project(x);
  Matrix2d result = Matrix2d::Zero();
  result << S * G * proj_x, S * proj_x;
  return result;
}

Matrix<double, 2, 3> TransformDerivativeX(const Matrix2d& S,
                                          const Vector3d& x) {
  return S * ProjectDerivative(x);
}

TEST(NumericalDerivativesTest, VectorFieldOnEuclideanSpace) {
  constexpr double kEpsilon = 1e-5;

  // Consider the mapping  S * x, where S is a 2d rotation and scaling matrix
  // and x a vector.  View this as a vector field on x (for fixed S).
  const Matrix2d S = 1.2 * SO2d{0.3}.matrix();
  auto vector_field = [S](const Vector3d& x) { return Transform(S, x); };

  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionVector3d dist;
  for (int i = 0; i < 5000; ++i) {
    const Vector3d x = dist(generator);
    const Matrix<double, 2, 3> analytical_jacobian_at_x =
        TransformDerivativeX(S, x);
    const Matrix<double, 2, 3> numerical_jacobian_at_x =
        VectorFieldNumericalDerivative<3, 2>(vector_field, x);
    EXPECT_THAT(numerical_jacobian_at_x,
                IsApprox(analytical_jacobian_at_x, kEpsilon));
  }
}

TEST(NumericalDerivativesTest, VectorFieldOnManifold) {
  constexpr double kEpsilon = 1e-5;

  // Consider the mapping  S * x, where S is a 2d rotation and scaling matrix
  // and x a 2-vector.  View this as a vector field on S (for fixed x).
  const Vector3d x(1, 2, 7);
  auto vector_field = [x](const Matrix2d& S) { return Transform(S, x); };

  // For the manifold structure on the matrices (which have 2 dimensions --
  // rotation angle and scaling factor), define the following PlusOperator. This
  // one includes an overloaded plus operator.
  //
  // The derivative is taken in this two-dimensional space, thus:
  //  "S g(R)" is calculated as "de g(plus(S,e)) at e=0"
  auto plus_in_b = [](const Matrix2d& a_S_b,
                      const Vector2d& delta) -> Matrix2d {
    Matrix2d delta_in_b = Matrix2d::Zero();
    delta_in_b << std::exp(delta[1]) * cos(delta[0]), -sin(delta[0]),
        sin(delta[0]), std::exp(delta[1]) * cos(delta[0]);

    return a_S_b * delta_in_b;
  };

  TestGenerator generator(kGeneratorTestSeed);
  UniformDistributionSO2d rotation_dist;
  std::uniform_real_distribution<double> scaling_dist(0.1, 10);
  for (int i = 0; i < 5000; ++i) {
    const double scaling = scaling_dist(generator);
    const Matrix2d S = scaling * rotation_dist(generator).matrix();
    const Matrix2d analytical_jacobian_at_S = TransformDerivativeS(S, x);
    const Matrix2d numerical_jacobian_at_S =
        VectorFieldNumericalDerivative<Matrix2d, 2, 2>(
            vector_field, S, PlusOperator<Matrix2d, 2>(plus_in_b));
    EXPECT_THAT(numerical_jacobian_at_S,
                IsApprox(analytical_jacobian_at_S, kEpsilon));
  }
}

}  // namespace
}  // namespace eigenmath
