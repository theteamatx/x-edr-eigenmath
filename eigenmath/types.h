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

#ifndef EIGENMATH_EIGENMATH_TYPES_H_
#define EIGENMATH_EIGENMATH_TYPES_H_

#include <ostream>

#include "Eigen/Core"      // IWYU pragma: export
#include "Eigen/Eigenvalues"  // IWYU pragma: export
#include "Eigen/Geometry"  // IWYU pragma: export
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace eigenmath {

constexpr int kDefaultOptions =
    Eigen::AutoAlign | EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION;

constexpr int kMaxEigenVectorCapacity = 16;
constexpr int kMaxEigenMatrixCapacity = 16;

// N-vector using Scalar.
template <class Scalar, int N, int Options = kDefaultOptions>
using Vector = Eigen::Matrix<Scalar, N, 1, Options>;

template <int N, int Options = kDefaultOptions>
using Vectord = Vector<double, N, Options>;
template <int N, int Options = kDefaultOptions>
using Vectorf = Vector<float, N, Options>;

// 2-vector using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Vector2 = Vector<Scalar, 2, Options>;

using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

// 3-vector using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Vector3 = Vector<Scalar, 3, Options>;

using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;
using Vector3b = Vector3<bool>;
using Vector3i = Vector3<int>;

// 4-vector using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Vector4 = Vector<Scalar, 4, Options>;

using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

// 6-vector using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Vector6 = Vector<Scalar, 6, Options>;

using Vector6d = Vector6<double>;
using Vector6f = Vector6<float>;
using Vector6b = Vector6<bool>;

// Fixed capacity, but dynamic sized vectors.
template <typename Scalar>
using VectorN = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, kDefaultOptions,
                              kMaxEigenVectorCapacity, 1>;
using VectorNd = VectorN<double>;
using VectorNf = VectorN<float>;
using VectorNb = VectorN<bool>;

// N-vector (runtime) using Scalar.
template <typename Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using VectorXd = VectorX<double>;
using VectorXf = VectorX<float>;
using VectorXi = VectorX<int>;
using VectorXb = VectorX<bool>;

// Either a fixed-size vector, a dynamic vector with a runtime size, or a
// dynamic vector with a compile-time maximum size.
// This will match any Vector2, Vector3, etc., VectorX and VectorN.
template <typename Scalar, int N, int MaxSize>
using VectorFixedOrDynamic =
    Eigen::Matrix<Scalar, N, 1, kDefaultOptions, MaxSize, 1>;

// Quaternion.
template <class Scalar, int Options = kDefaultOptions>
using Quaternion = Eigen::Quaternion<Scalar, Options>;

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

// Matrix.
template <class Scalar, int Rows, int Cols,
          int Options = Eigen::AutoAlign |
                        ((Rows == 1 && Cols != 1) ? Eigen::RowMajor
                         : (Cols == 1 && Rows != 1)
                             ? Eigen::ColMajor
                             : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION)>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options>;

// (2x2) matrix using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Matrix2 = Matrix<Scalar, 2, 2, Options>;

using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;
using Eigen::Matrix2Xd;
using Eigen::Matrix2Xf;
using Eigen::MatrixX2d;
using Eigen::MatrixX2f;

// Variable size matrices.
template <class Scalar, int Options = kDefaultOptions>
using MatrixX = Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;
using MatrixXf = MatrixX<float>;
using MatrixXd = MatrixX<double>;

template <int N, int M = N, int Options = kDefaultOptions>
using Matrixd = Matrix<double, N, M, Options>;
template <int N, int M = N, int Options = kDefaultOptions>
using Matrixf = Matrix<float, N, M, Options>;

// (3x3) matrix using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Matrix3 = Matrix<Scalar, 3, 3, Options>;

using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;
using Eigen::Matrix3Xd;
using Eigen::Matrix3Xf;
using Eigen::MatrixX3d;
using Eigen::MatrixX3f;

// (4x4) matrix using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Matrix4 = Matrix<Scalar, 4, 4, Options>;

using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;
using Eigen::Matrix4Xd;
using Eigen::Matrix4Xf;
using Eigen::MatrixX4d;
using Eigen::MatrixX4f;

// 6 x 6 matrix using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Matrix6 = Matrix<Scalar, 6, 6, Options>;

using Matrix6f = Matrix6<float>;
using Matrix6d = Matrix6<double>;
using Matrix6Xf = Eigen::Matrix<float, 6, Eigen::Dynamic>;
using Matrix6Xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;
using MatrixX6f = Eigen::Matrix<float, Eigen::Dynamic, 6>;
using MatrixX6d = Eigen::Matrix<double, Eigen::Dynamic, 6>;

// Fixed capacity, but dynamic sized matrices.
using MatrixNd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, kDefaultOptions,
                  kMaxEigenMatrixCapacity, kMaxEigenMatrixCapacity>;
using MatrixNf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, kDefaultOptions,
                  kMaxEigenMatrixCapacity, kMaxEigenMatrixCapacity>;

using Matrix6Nd = Eigen::Matrix<double, 6, Eigen::Dynamic, kDefaultOptions, 6,
                                kMaxEigenMatrixCapacity>;
using Matrix6Nf = Eigen::Matrix<float, 6, Eigen::Dynamic, kDefaultOptions, 6,
                                kMaxEigenMatrixCapacity>;
using MatrixN6d = Eigen::Matrix<double, Eigen::Dynamic, 6, kDefaultOptions,
                                kMaxEigenMatrixCapacity, 6>;
using MatrixN6f = Eigen::Matrix<float, Eigen::Dynamic, 6, kDefaultOptions,
                                kMaxEigenMatrixCapacity, 6>;

// 3d plane using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Plane3 = Eigen::Hyperplane<Scalar, 3, Options>;

// 3d plane using doubles.
using Plane3d = Plane3<double>;  // Dim coefficients is 3 + 1 = 4

// 3d plane using floats.
using Plane3f = Plane3<float>;  // Dim coefficients is 3 + 1 = 4

// 2d line using Scalar.
template <class Scalar, int Options = kDefaultOptions>
using Line2 = Eigen::Hyperplane<Scalar, 2, Options>;

// 2d line using doubles.
using Line2d = Line2<double>;  // Dim coefficients is 2 + 1 = 3

// 2d line using floats.
using Line2f = Line2<float>;  // Dim coefficients is 2 + 1 = 3

// Policy whether to normalize quaternions on construction.
enum NormalizationPolicy { kNormalize, kDoNotNormalize };

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
absl::Span<const Scalar> MakeSpan(
    const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& mat) {
  return absl::Span<const Scalar>(mat.data(), mat.rows() * mat.cols());
}
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
absl::Span<Scalar> MakeSpan(
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& mat) {
  return absl::Span<Scalar>(mat.data(), mat.rows() * mat.cols());
}
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
absl::Span<const Scalar> MakeConstSpan(
    const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& mat) {
  return absl::Span<const Scalar>(mat.data(), mat.rows() * mat.cols());
}

template <typename AnyEigenType>
struct AbslStringified {
  const AnyEigenType* m;
  explicit AbslStringified(const AnyEigenType& m_) : m(&m_) {}

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const AbslStringified& self) {
    // Print compact one-liner matrix.
    absl::Format(&sink, "%s",
                 (std::stringstream() << self.m->format(
                      Eigen::IOFormat(6, 0, ",", ";", "", "", "[", "]")))
                     .str());
  }
};

template <typename AnyEigenType>
AbslStringified(const AnyEigenType&) -> AbslStringified<AnyEigenType>;

}  // namespace eigenmath

// Open the Eigen namespace for better printers for matrix and vector types.
namespace Eigen {

inline IOFormat GTestFormat() {
  const int precision = 3;
  const int flags = 0;
  const char coeff_sep[] = ", ";
  const char row_sep[] = "\n";
  const char row_pre[] = "  [";
  const char row_post[] = "]";
  const char mat_pre[] = "\n";
  const char mat_post[] = "\n";
  return IOFormat(precision, flags, coeff_sep, row_sep, row_pre, row_post,
                  mat_pre, mat_post);
}

template <typename Scalar, int... Args>
void PrintTo(const Matrix<Scalar, Args...>& m, ::std::ostream* os) {
  (*os) << m.format(GTestFormat());
}

template <typename XprType, int Rows, int Cols, bool InnerPanel>
void PrintTo(const Block<XprType, Rows, Cols, InnerPanel>& b,
             ::std::ostream* os) {
  (*os) << b.format(GTestFormat());
}

template <typename UnaryOp, typename XprType>
void PrintTo(const CwiseUnaryOp<UnaryOp, XprType>& b, ::std::ostream* os) {
  (*os) << b.format(GTestFormat());
}

}  // namespace Eigen

#endif  // EIGENMATH_EIGENMATH_TYPES_H_
