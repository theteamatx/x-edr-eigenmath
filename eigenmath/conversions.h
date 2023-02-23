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

#ifndef EIGENMATH_EIGENMATH_CONVERSIONS_H_
#define EIGENMATH_EIGENMATH_CONVERSIONS_H_

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "eigenmath/eigenmath.pb.h"
#include "eigenmath/pose2.h"
#include "eigenmath/pose3.h"
#include "eigenmath/types.h"
#include "google/protobuf/repeated_field.h"

namespace eigenmath {
namespace conversions {

////////////////////////////////////////////////////////////////////////////////
// Functions to convert between eigemath and proto types.

// Convert from a proto repeated field to an eigenmath vector.
template <class Scalar, int N, int MaxSize>
VectorFixedOrDynamic<Scalar, N, MaxSize> EigenVectorFromProto(
    const ::google::protobuf::RepeatedField<Scalar>& proto_vector);

// As above, but with expected size, returns false if the size doesn't match.
template <class Scalar, int N, int MaxSize>
bool EigenVectorFromProto(
    const ::google::protobuf::RepeatedField<Scalar>& proto_vector,
    int expected_size, VectorFixedOrDynamic<Scalar, N, MaxSize>* output);

// Gets an eigenmath vector from a Vector proto.
// Asserts that the proto is valid, i.e. correct size.
Vector2d EigenVectorFromProto(const Vector2dProto& proto);
Vector3d EigenVectorFromProto(const Vector3dProto& proto);
Vector4d EigenVectorFromProto(const Vector4dProto& proto);
Vector6d EigenVectorFromProto(const Vector6dProto& proto);
VectorXd EigenVectorFromProto(const VectorNdProto& proto);
VectorXd EigenVectorFromProto(const VectorXdProto& proto);

// Gets an eigenmath matrix from a Matrix proto.
// Asserts that the protos is valid, i.e. correct size.
Matrix2d EigenMatrixFromProto(const Matrix2dProto& proto);
Matrix3d EigenMatrixFromProto(const Matrix3dProto& proto);
Matrix4d EigenMatrixFromProto(const Matrix4dProto& proto);
Matrix6d EigenMatrixFromProto(const Matrix6dProto& proto);

// Gets an eigenmath Pose from a Pose proto.
// Asserts that the pose contained in the proto is valid.
Pose2d PoseFromProto(const Pose2dProto& proto);
Pose3d PoseFromProto(const Pose3dProto& proto);
// As above, but returns false if the conversion is not possible, e.g. invalid
// pose.
// If all elements in the quaternion proto are zero/missing, an identity
// quaternion is returned.
bool PoseFromProto(const Pose2dProto& proto, Pose2d* pose);
bool PoseFromProto(const Pose3dProto& proto, Pose3d* pose);

// Convert eigenmath quaternions to/from protos.
Quaterniond QuaternionFromProto(const QuaterniondProto& proto);
QuaterniondProto ProtoFromQuaternion(const eigenmath::Quaterniond& quaternion);

// Convert from an eigenmath vector to a proto repeated field.
template <class Scalar, int N, int MaxSize>
void ProtoFromEigenVector(
    const VectorFixedOrDynamic<Scalar, N, MaxSize>& input,
    ::google::protobuf::RepeatedField<Scalar>* proto_vector);

// Convert an eigenmath vector to the corresponding proto vector type.
Vector2dProto ProtoFromVector2d(const Vector2d& vec2);
Vector3dProto ProtoFromVector3d(const Vector3d& vec3);
Vector4dProto ProtoFromVector4d(const Vector4d& vec4);
Vector6dProto ProtoFromVector6d(const Vector6d& vec6);
VectorNdProto ProtoFromVectorNd(const VectorXd& vec);
VectorXdProto ProtoFromVectorXd(const VectorXd& vec);

// Convert an eigenmath matrix to the corresponding proto matrix type.
Matrix2dProto ProtoFromMatrix2d(const Matrix2d& mat2);
Matrix3dProto ProtoFromMatrix3d(const Matrix3d& mat3);
Matrix4dProto ProtoFromMatrix4d(const Matrix4d& mat4);
Matrix6dProto ProtoFromMatrix6d(const Matrix6d& mat6);

// Converts an eigenmath pose to the corresponding proto pose type.
Pose2dProto ProtoFromPose(const Pose2d& pose);
Pose3dProto ProtoFromPose(const Pose3d& pose);

////////////////////////////////////////////////////////////////////////////////
// Functions to convert eigenmath types to strings.

// Converts any iterable type to a string of the form "0, 1, 2".
template <typename SequenceType>
std::string SequenceToString(const SequenceType& sequence);

// Converts any quaternion type to a string of the form "w, x, y, z".
template <typename RealType>
std::string QuaternionToString(const Quaternion<RealType>& quaternion);

// Converts a Pose3 into "[<px>, <py>, <pz>, <qw>, <qx>, <qy>, <qz>]".
template <typename RealType>
std::string Pose3ToString(const Pose3<RealType, Eigen::AutoAlign>& pose);

// Converts a Pose3d into "[<px>, <py>, <pz>, <qw>, <qx>, <qy>, <qz>]".
std::string Pose3dToString(const Pose3d& pose);

// Converts a VectorXd to a string of the form "[0, 1, 2]".
std::string VectorXdToString(const VectorXd& vector);

// Converts a MatrixXd to a string of the form "[[0, 1]\n[2, 3]]".
std::string MatrixXdToString(const MatrixXd& matrix);

// Converts a Quaterniond into "[<qw>, <qx>, <qy>, <qz>]".
std::string QuaterniondToString(const Quaterniond& quaternion);

////////////////////////////////////////////////////////////////////////////////
// Implementation of templated functions.

// Convert from a proto repeated field to an Eigen::Vector<Scalar> without
// expected size.
template <class Scalar, int N, int MaxSize>
VectorFixedOrDynamic<Scalar, N, MaxSize> EigenVectorFromProto(
    const ::google::protobuf::RepeatedField<Scalar>& proto_vector) {
  VectorFixedOrDynamic<Scalar, N, MaxSize> output(proto_vector.size());
  for (std::size_t i = 0; i < proto_vector.size(); ++i) {
    output[i] = proto_vector.Get(i);
  }
  return output;
}

// Convert to a VectorX from any repeated proto field.
template <class Scalar>
VectorX<Scalar> EigenVectorFromProto(
    const ::google::protobuf::RepeatedField<Scalar>& proto_vector) {
  return EigenVectorFromProto<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
      proto_vector);
}

// Convert from a proto repeated field to an Eigen::Vector<Scalar> with
// expected size.
template <class Scalar, int N, int MaxSize>
bool EigenVectorFromProto(
    const ::google::protobuf::RepeatedField<Scalar>& proto_vector,
    int expected_size, VectorFixedOrDynamic<Scalar, N, MaxSize>* output) {
  CHECK_NE(output, nullptr);

  if (proto_vector.size() != expected_size) {
    return false;
  }
  *output = EigenVectorFromProto<Scalar, N, MaxSize>(proto_vector);
  return true;
}

template <class Scalar, int N, int MaxSize>
void ProtoFromEigenVector(
    const VectorFixedOrDynamic<Scalar, N, MaxSize>& input,
    ::google::protobuf::RepeatedField<Scalar>* proto_vector) {
  CHECK_NE(proto_vector, nullptr);
  proto_vector->Clear();
  proto_vector->Reserve(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    proto_vector->Add(input[i]);
  }
}

template <typename Scalar, typename SequenceType>
bool ProtoFromSequence(
    const SequenceType& input,
    ::google::protobuf::RepeatedField<Scalar>* proto_vector) {
  if (proto_vector == nullptr) {
    return false;
  }
  proto_vector->Clear();
  proto_vector->Reserve(input.size());
  for (const auto& v : input) {  // NOLINT: Not to be used for eigenvectors.
    proto_vector->Add(static_cast<Scalar>(v));
  }
  return true;
}

template <typename SequenceType>
std::string SequenceToString(const SequenceType& sequence) {
  return absl::StrJoin(sequence, ", ");
}

// Converts any quaternion type to a string of the form "w, x, y, z".
template <typename RealType>
std::string QuaternionToString(const Quaternion<RealType>& quaternion) {
  return absl::StrCat(quaternion.w(), ", ", quaternion.x(), ", ",
                      quaternion.y(), ", ", quaternion.z());
}

// Converts a Pose3 into "[<px>, <py>, <pz>, <qw>, <qx>, <qy>, <qz>]".
template <typename RealType>
std::string Pose3ToString(const Pose3<RealType, Eigen::AutoAlign>& pose) {
  static_assert(std::is_floating_point<RealType>::value,
                "RealType must be a floating point number.");
  const auto& translation = pose.translation();
  const auto& quaternion = pose.quaternion();
  return absl::StrCat("[", SequenceToString(translation), ", ",
                      QuaternionToString(quaternion), "]");
}

}  // namespace conversions
}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_CONVERSIONS_H_
