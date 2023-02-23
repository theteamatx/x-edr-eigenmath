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

#include "quintic_spline.h"

#include <algorithm>
#include <vector>

#include "absl/log/check.h"
#include "spline_coefficients.h"
#include "types.h"

namespace eigenmath {

QuinticSpline::QuinticSpline(const Eigen::MatrixXd& control_points)
    : control_points_(control_points) {
  CHECK_EQ(control_points_.rows() % 3, 0)
      << "Control points must have rows %% 3 == 0";
}

QuinticSpline::LocalIndex QuinticSpline::GetLocalIndex(
    double spline_index) const {
  CHECK_GE(spline_index, 0);
  CHECK_LE(spline_index, MaxSplineIndex() + 1.0e-6);
  int segment = static_cast<int>(spline_index);  // floor
  if (segment == NumSegments()) {
    --segment;
  }
  double local_index = spline_index - static_cast<double>(segment);
  return {segment, local_index};
}

template <int degree>
VectorXd QuinticSpline::Sample(double spline_index) const {
  VectorXd value(NumDof());
  Sample<degree>(spline_index, value);
  return value;
}

template <int degree>
void QuinticSpline::Sample(double spline_index, VectorXd& value) const {
  auto index = GetLocalIndex(spline_index);
  int param_offset = index.segment * 3;
  const auto& coeffs = SplineCoefficients<degree>(index.local_index);
  value = control_points_.middleRows(param_offset, 6).transpose() * coeffs;
}

template <int degree>
std::vector<VectorXd> QuinticSpline::Sample(
    const std::vector<double>& spline_indices) const {
  std::vector<VectorXd> values(spline_indices.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = Sample<degree>(spline_indices[i]);
  }
  return values;
}

template <int degree>
std::vector<VectorXd> QuinticSpline::Discretize(int num_samples) const {
  CHECK_GE(num_samples, 2);
  std::vector<VectorXd> samples(num_samples, VectorXd(NumDof()));
  double interval = MaxSplineIndex() / static_cast<double>(num_samples - 1);
  for (int i = 0; i < num_samples; ++i) {
    samples[i] = Sample<degree>(std::min(MaxSplineIndex(), i * interval));
  }
  return samples;
}

template <int degree>
VectorXd QuinticSpline::IntegralSquaredDerivative() const {
  VectorXd sq_deriv(NumDof());
  sq_deriv.setZero();
  for (int i = 0; i < NumSegments(); ++i) {
    const auto& params = control_points_.middleRows<6>(i * 3);
    sq_deriv += ((SplineIntegralSquaredDerivative<degree>() * params).array() *
                 params.array())
                    .colwise()
                    .sum()
                    .matrix();
  }
  return sq_deriv;
}

double QuinticSpline::MaxSplineIndex() const { return NumSegments(); }

int QuinticSpline::NumDof() const { return control_points_.cols(); }

int QuinticSpline::NumSegments() const {
  return control_points_.rows() / 3 - 1;
}

// instantiate for 0 through 3rd derivatives
template VectorXd QuinticSpline::Sample<0>(double) const;
template VectorXd QuinticSpline::Sample<1>(double) const;
template VectorXd QuinticSpline::Sample<2>(double) const;
template VectorXd QuinticSpline::Sample<3>(double) const;
template void QuinticSpline::Sample<0>(double, VectorXd&) const;
template void QuinticSpline::Sample<1>(double, VectorXd&) const;
template void QuinticSpline::Sample<2>(double, VectorXd&) const;
template void QuinticSpline::Sample<3>(double, VectorXd&) const;
template std::vector<VectorXd> QuinticSpline::Discretize<0>(int) const;
template std::vector<VectorXd> QuinticSpline::Discretize<1>(int) const;
template std::vector<VectorXd> QuinticSpline::Discretize<2>(int) const;
template std::vector<VectorXd> QuinticSpline::Discretize<3>(int) const;
template std::vector<VectorXd> QuinticSpline::Sample<0>(
    const std::vector<double>&) const;
template std::vector<VectorXd> QuinticSpline::Sample<1>(
    const std::vector<double>&) const;
template std::vector<VectorXd> QuinticSpline::Sample<2>(
    const std::vector<double>&) const;
template std::vector<VectorXd> QuinticSpline::Sample<3>(
    const std::vector<double>&) const;
template VectorXd QuinticSpline::IntegralSquaredDerivative<1>() const;
template VectorXd QuinticSpline::IntegralSquaredDerivative<2>() const;
template VectorXd QuinticSpline::IntegralSquaredDerivative<3>() const;

}  // namespace eigenmath
