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

#ifndef EIGENMATH_EIGENMATH_QUADRATURE_H_
#define EIGENMATH_EIGENMATH_QUADRATURE_H_

#include <array>
#include <limits>

#include "absl/log/check.h"

namespace eigenmath {

// Class template to store the knot points and weights for the Gauss-Legendre
// quadrature rules for a given order. This class template is specialized for
// each supported order.
template <int Order>
struct GaussLegendreCoeffs;

template <>
struct GaussLegendreCoeffs<2> {
  static const std::array<double, 2> knots;
  static const std::array<double, 2> weights;
};

template <>
struct GaussLegendreCoeffs<3> {
  static const std::array<double, 3> knots;
  static const std::array<double, 3> weights;
};

template <>
struct GaussLegendreCoeffs<4> {
  static const std::array<double, 4> knots;
  static const std::array<double, 4> weights;
};

template <>
struct GaussLegendreCoeffs<5> {
  static const std::array<double, 5> knots;
  static const std::array<double, 5> weights;
};

template <>
struct GaussLegendreCoeffs<6> {
  static const std::array<double, 6> knots;
  static const std::array<double, 6> weights;
};

template <>
struct GaussLegendreCoeffs<7> {
  static const std::array<double, 7> knots;
  static const std::array<double, 7> weights;
};

template <>
struct GaussLegendreCoeffs<8> {
  static const std::array<double, 8> knots;
  static const std::array<double, 8> weights;
};

template <>
struct GaussLegendreCoeffs<9> {
  static const std::array<double, 9> knots;
  static const std::array<double, 9> weights;
};

template <>
struct GaussLegendreCoeffs<10> {
  static const std::array<double, 10> knots;
  static const std::array<double, 10> weights;
};

template <>
struct GaussLegendreCoeffs<12> {
  static const std::array<double, 12> knots;
  static const std::array<double, 12> weights;
};

// Computes the integral for a given function over the interval [a, b] using the
// Gauss-Legendre quadrature rule of the given order.
template <int Order, typename Func>
double ComputeGaussLegendreIntegral(double a, double b, Func f) {
  const double trans_base = (a + b) / 2;
  const double trans_fact = (b - a) / 2;
  const std::array<double, Order>& x = GaussLegendreCoeffs<Order>::knots;
  const std::array<double, Order>& w = GaussLegendreCoeffs<Order>::weights;
  double result = 0.0;
  for (int i = 0; i < Order; ++i) {
    result += w[i] * f(trans_fact * x[i] + trans_base);
  }
  return trans_fact * result;
}

// Computes the integral for a given function over the interval [a, b], by
// evaluating the function incrementally, starting from the lower-bound a.
//
// The integral is computed with a Gauss-Legendre quadrature rule of the
// given order.
template <int Order, typename Func>
double ComputeGaussLegendreIntegralIncrementally(double a, double b, Func f) {
  const double trans_base = (a + b) / 2;
  const double trans_fact = (b - a) / 2;
  const std::array<double, Order>& x = GaussLegendreCoeffs<Order>::knots;
  const std::array<double, Order>& w = GaussLegendreCoeffs<Order>::weights;
  double result = 0.0;
  double last_input = a;
  for (int i = 0; i < Order; ++i) {
    double current_input = trans_fact * x[i] + trans_base;
    result += w[i] * f(current_input - last_input);
    last_input = current_input;
  }
  return trans_fact * result;
}

// Computes the integral for a given function over the interval [a, b]
// using Gauss-Legendre quadrature rule of the given order.
template <typename Func>
double ComputeGaussLegendreIntegral(int order, double a, double b, Func f) {
  switch (order) {
    case 2:
      return ComputeGaussLegendreIntegral<2>(a, b, f);
    case 3:
      return ComputeGaussLegendreIntegral<3>(a, b, f);
    case 4:
      return ComputeGaussLegendreIntegral<4>(a, b, f);
    case 5:
      return ComputeGaussLegendreIntegral<5>(a, b, f);
    case 6:
      return ComputeGaussLegendreIntegral<6>(a, b, f);
    case 7:
      return ComputeGaussLegendreIntegral<7>(a, b, f);
    case 8:
      return ComputeGaussLegendreIntegral<8>(a, b, f);
    case 9:
      return ComputeGaussLegendreIntegral<9>(a, b, f);
    case 10:
      return ComputeGaussLegendreIntegral<10>(a, b, f);
    case 12:
      return ComputeGaussLegendreIntegral<12>(a, b, f);
    default:
      CHECK(false) << "order not supported.";
  }
}

// Computes the integral for a given function over the interval [a, b], by
// evaluating the function incrementally, starting from the lower-bound a.
// The integral is computed with a Gauss-Legendre quadrature rule of the given
// order.
template <typename Func>
double ComputeGaussLegendreIntegralIncrementally(int order, double a, double b,
                                                 Func f) {
  switch (order) {
    case 2:
      return ComputeGaussLegendreIntegralIncrementally<2>(a, b, f);
    case 3:
      return ComputeGaussLegendreIntegralIncrementally<3>(a, b, f);
    case 4:
      return ComputeGaussLegendreIntegralIncrementally<4>(a, b, f);
    case 5:
      return ComputeGaussLegendreIntegralIncrementally<5>(a, b, f);
    case 6:
      return ComputeGaussLegendreIntegralIncrementally<6>(a, b, f);
    case 7:
      return ComputeGaussLegendreIntegralIncrementally<7>(a, b, f);
    case 8:
      return ComputeGaussLegendreIntegralIncrementally<8>(a, b, f);
    case 9:
      return ComputeGaussLegendreIntegralIncrementally<9>(a, b, f);
    case 10:
      return ComputeGaussLegendreIntegralIncrementally<10>(a, b, f);
    case 12:
      return ComputeGaussLegendreIntegralIncrementally<12>(a, b, f);
    default:
      CHECK(false) << "order not supported.";
  }
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_QUADRATURE_H_
