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

#ifndef EIGENMATH_EIGENMATH_NUMERICAL_DERIVATIVES_H_
#define EIGENMATH_EIGENMATH_NUMERICAL_DERIVATIVES_H_

#include <functional>

#include "Eigen/Core"
#include "absl/log/check.h"
#include "types.h"

namespace eigenmath {

namespace numerical_derivatives_details {

inline constexpr double kFirstDerivativeStepSize = 1e-10;

}  // namespace numerical_derivatives_details

template <typename Manifold, int TangentDim>
class PlusOperator {
 public:
  using TangentVector = Vector<double, TangentDim>;
  using PlusOperatorFun =
      std::function<Manifold(const Manifold&, const TangentVector&)>;

  explicit PlusOperator(PlusOperatorFun fun, int tangent_dim = TangentDim)
      : fun_(std::move(fun)), tangent_dim_(tangent_dim) {
    CHECK_GT(tangent_dim, 0) << "Run-time tangent_dim must be greater 0";
    CHECK(TangentDim == Eigen::Dynamic || tangent_dim == TangentDim)
        << "If compile-time TangentDim is not Eigen::Dynamic, then it must be "
           "equal to runtime target_dim";
  }

  // Maps point on manifold and tangent vector to another point on manifold
  Manifold operator()(const Manifold& point,
                      const TangentVector& tangent_vector) const {
    return fun_(point, tangent_vector);
  }

  //  Runtime dimension of tangent space
  int TangentDimension() const { return tangent_dim_; }

 private:
  //  plus operator function
  PlusOperatorFun fun_;
  //  runtime dimension of tangent space
  int tangent_dim_;
};

// Returns the numerical Jacobian of `vector_field` at `point` on a manifold.
template <typename Manifold, int TangentDim, int OutputDim>
Matrix<double, OutputDim, TangentDim> VectorFieldNumericalDerivative(
    const std::function<Vector<double, OutputDim>(const Manifold&)>&
        vector_field,
    const Manifold& point, const PlusOperator<Manifold, TangentDim>& plus,
    double step = numerical_derivatives_details::kFirstDerivativeStepSize) {
  const int tangent_dimension = plus.TangentDimension();
  const Vector<double, OutputDim> vector_field_at_point = vector_field(point);
  Matrix<double, OutputDim, TangentDim> jacobian(OutputDim, tangent_dimension);
  jacobian.setZero();
  Vector<double, TangentDim> v(tangent_dimension);
  v.setZero();
  for (int i = 0; i < tangent_dimension; ++i) {
    v[i] = step;
    jacobian.col(i) =
        (vector_field(plus(point, v)) - vector_field_at_point) / step;
    v[i] = 0.;
  }
  return jacobian;
}

// As above, for Euclidean space.
template <int InputDim, int OutputDim>
Matrix<double, OutputDim, InputDim> VectorFieldNumericalDerivative(
    const std::function<Vector<double, OutputDim>(
        const Vector<double, InputDim>&)>& vector_field,
    const Vector<double, InputDim>& point,
    double step = numerical_derivatives_details::kFirstDerivativeStepSize) {
  using InputVector = Vector<double, InputDim>;
  return VectorFieldNumericalDerivative(
      vector_field, point,
      PlusOperator<InputVector, InputDim>(std::plus<InputVector>(),
                                          point.size()),
      step);
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_NUMERICAL_DERIVATIVES_H_
