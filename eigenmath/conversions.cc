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

#include "eigenmath/conversions.h"

#include <cmath>
#include <limits>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "eigenmath/eigenmath.pb.h"
#include "eigenmath/pose3_utils.h"
#include "eigenmath/rotation_utils.h"
#include "eigenmath/types.h"

namespace eigenmath::conversions {
namespace {

bool IsQuaternionValid(const Quaterniond& quaternion) {
  static constexpr double TINY = 100 * std::numeric_limits<double>::epsilon();
  if (std::isnan(quaternion.squaredNorm()) || quaternion.squaredNorm() < TINY) {
    return false;
  }
  return true;
}

// Functor to help format rows/cols of a matrix as strings.
struct VectorFormatter {
  template <typename VectorType>
  void operator()(std::string* out, const VectorType& v) const {
    absl::StrAppend(out, "[", SequenceToString(v), "]");
  }
};

template <typename Pose2dProtoType>
void ProtoFromPoseImpl(const Pose2d& pose2d, Pose2dProtoType& proto) {
  proto.set_tx(pose2d.translation().x());
  proto.set_ty(pose2d.translation().y());
  proto.set_rotation(pose2d.angle());
}

template <typename Pose2dProtoType>
void PoseFromProtoImpl(const Pose2dProtoType& proto, Pose2d& pose2d) {
  pose2d.translation().x() = proto.tx();
  pose2d.translation().y() = proto.ty();
  pose2d.setAngle(proto.rotation());
}

}  // namespace.

Vector2d EigenVectorFromProto(const Vector2dProto& proto) {
  CHECK(proto.vec_size() == 0 || proto.vec_size() == 2);
  if (proto.vec().empty()) {
    return Vector2d::Zero();
  }
  return Vector2d(proto.vec(0), proto.vec(1));
}

Vector3d EigenVectorFromProto(const Vector3dProto& proto) {
  CHECK(proto.vec_size() == 0 || proto.vec_size() == 3);
  if (proto.vec().empty()) {
    return Vector3d::Zero();
  }
  return Vector3d(proto.vec(0), proto.vec(1), proto.vec(2));
}

Vector4d EigenVectorFromProto(const Vector4dProto& proto) {
  CHECK(proto.vec_size() == 0 || proto.vec_size() == 4);
  if (proto.vec().empty()) {
    return Vector4d::Zero();
  }
  return Vector4d(proto.vec(0), proto.vec(1), proto.vec(2), proto.vec(3));
}

Vector6d EigenVectorFromProto(const Vector6dProto& proto) {
  CHECK(proto.vec_size() == 0 || proto.vec_size() == 6);
  if (proto.vec().empty()) {
    return Vector6d::Zero();
  }
  return Vector6d(proto.vec(0), proto.vec(1), proto.vec(2), proto.vec(3),
                  proto.vec(4), proto.vec(5));
}

VectorXd EigenVectorFromProto(const VectorNdProto& proto) {
  return EigenVectorFromProto(proto.vec());
}

Matrix2d EigenMatrixFromProto(const Matrix2dProto& proto) {
  CHECK(proto.mat_size() == 0 || proto.mat_size() == 4);
  if (proto.mat().empty()) {
    return Matrix2d::Zero();
  }

  Matrix2d mat = Matrix2d::Zero();
  // Matrices are packed in row-major for the proto view.
  mat(0, 0) = proto.mat(0);
  mat(0, 1) = proto.mat(1);
  mat(1, 0) = proto.mat(2);
  mat(1, 1) = proto.mat(3);
  return mat;
}

Matrix3d EigenMatrixFromProto(const Matrix3dProto& proto) {
  CHECK(proto.mat_size() == 0 || proto.mat_size() == 9);
  if (proto.mat().empty()) {
    return Matrix3d::Zero();
  }

  Matrix3d mat = Matrix3d::Zero();
  // Matrices are packed in row-major for the proto view.
  mat(0, 0) = proto.mat(0);
  mat(0, 1) = proto.mat(1);
  mat(0, 2) = proto.mat(2);
  mat(1, 0) = proto.mat(3);
  mat(1, 1) = proto.mat(4);
  mat(1, 2) = proto.mat(5);
  mat(2, 0) = proto.mat(6);
  mat(2, 1) = proto.mat(7);
  mat(2, 2) = proto.mat(8);
  return mat;
}

Matrix4d EigenMatrixFromProto(const Matrix4dProto& proto) {
  CHECK(proto.mat_size() == 0 || proto.mat_size() == 16);
  if (proto.mat().empty()) {
    return Matrix4d::Zero();
  }

  Matrix4d mat = Matrix4d::Zero();
  // Matrices are packed in row-major for the proto view.
  mat(0, 0) = proto.mat(0);
  mat(0, 1) = proto.mat(1);
  mat(0, 2) = proto.mat(2);
  mat(0, 3) = proto.mat(3);
  mat(1, 0) = proto.mat(4);
  mat(1, 1) = proto.mat(5);
  mat(1, 2) = proto.mat(6);
  mat(1, 3) = proto.mat(7);
  mat(2, 0) = proto.mat(8);
  mat(2, 1) = proto.mat(9);
  mat(2, 2) = proto.mat(10);
  mat(2, 3) = proto.mat(11);
  mat(3, 0) = proto.mat(12);
  mat(3, 1) = proto.mat(13);
  mat(3, 2) = proto.mat(14);
  mat(3, 3) = proto.mat(15);
  return mat;
}

Matrix6d EigenMatrixFromProto(const Matrix6dProto& proto) {
  CHECK(proto.mat_size() == 0 || proto.mat_size() == 36);
  if (proto.mat().empty()) {
    return Matrix6d::Zero();
  }

  Matrix6d mat = Matrix6d::Zero();
  // Matrices are packed in row-major for the proto view.
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      mat(i, j) = proto.mat(6 * i + j);
    }
  }
  return mat;
}

bool PoseFromProto(const Pose2dProto& proto, Pose2d* pose) {
  pose->translation().x() = proto.tx();
  pose->translation().y() = proto.ty();
  pose->so2() = SO2d(proto.rotation());
  return true;
}

Pose2d PoseFromProto(const Pose2dProto& proto) {
  Pose2d pose;
  CHECK(PoseFromProto(proto, &pose)) << absl::StrFormat(
      "Couldn't convert Pose2dProto to Pose2d. Invalid proto:/n%s",
      proto.DebugString().c_str());
  return pose;
}

bool PoseFromProto(const Pose3dProto& proto, Pose3d* pose) {
  Quaterniond quat = Quaterniond::Identity();
  // If the quaternion is null, assume this means it is supposed to be an
  // identity quaternion.
  if (proto.rw() != 0 || proto.rx() != 0 || proto.ry() != 0 ||
      proto.rz() != 0) {
    quat.w() = proto.rw();
    quat.x() = proto.rx();
    quat.y() = proto.ry();
    quat.z() = proto.rz();
    if (!IsQuaternionValid(quat)) {
      return false;
    }

    quat.normalize();
  }
  *pose = Pose3d(quat, Vector3d(proto.tx(), proto.ty(), proto.tz()));
  return true;
}

Pose3d PoseFromProto(const Pose3dProto& proto) {
  Pose3d pose;
  CHECK(PoseFromProto(proto, &pose)) << absl::StrFormat(
      "Couldn't convert Pose3dProto to Pose3d. Invalid proto:\n%s",
      proto.DebugString().c_str());
  return pose;
}

Quaterniond QuaternionFromProto(const QuaterniondProto& proto) {
  return Quaterniond(proto.w(), proto.x(), proto.y(), proto.z());
}

QuaterniondProto ProtoFromQuaternion(const Quaterniond& quaternion) {
  QuaterniondProto proto;
  proto.set_w(quaternion.w());
  proto.set_x(quaternion.x());
  proto.set_y(quaternion.y());
  proto.set_z(quaternion.z());
  return proto;
}

Vector2dProto ProtoFromVector2d(const Vector2d& vec2) {
  Vector2dProto proto;
  proto.add_vec(vec2.x());
  proto.add_vec(vec2.y());
  return proto;
}

Vector3dProto ProtoFromVector3d(const Vector3d& vec3) {
  Vector3dProto proto;
  proto.add_vec(vec3.x());
  proto.add_vec(vec3.y());
  proto.add_vec(vec3.z());
  return proto;
}

Vector4dProto ProtoFromVector4d(const Vector4d& vec4) {
  Vector4dProto proto;
  proto.add_vec(vec4(0));
  proto.add_vec(vec4(1));
  proto.add_vec(vec4(2));
  proto.add_vec(vec4(3));
  return proto;
}

Vector6dProto ProtoFromVector6d(const Vector6d& vec6) {
  Vector6dProto proto;
  proto.add_vec(vec6(0));
  proto.add_vec(vec6(1));
  proto.add_vec(vec6(2));
  proto.add_vec(vec6(3));
  proto.add_vec(vec6(4));
  proto.add_vec(vec6(5));
  return proto;
}

VectorNdProto ProtoFromVectorNd(const VectorXd& vec) {
  VectorNdProto proto;
  ProtoFromEigenVector(vec, proto.mutable_vec());
  return proto;
}

VectorXdProto ProtoFromVectorXd(const VectorXd& vector) {
  VectorXdProto proto;
  proto.mutable_vec()->Reserve(vector.rows());
  for (int i = 0; i < vector.rows(); ++i) {
    proto.mutable_vec()->Add(vector(i));
  }
  return proto;
}

Matrix2dProto ProtoFromMatrix2d(const Matrix2d& mat2) {
  Matrix2dProto proto;
  // Matrices are packed in row-major for the proto view.
  proto.add_mat(mat2(0, 0));
  proto.add_mat(mat2(0, 1));
  proto.add_mat(mat2(1, 0));
  proto.add_mat(mat2(1, 1));
  return proto;
}

Matrix3dProto ProtoFromMatrix3d(const Matrix3d& mat3) {
  Matrix3dProto proto;
  // Matrices are packed in row-major for the proto view.
  proto.add_mat(mat3(0, 0));
  proto.add_mat(mat3(0, 1));
  proto.add_mat(mat3(0, 2));
  proto.add_mat(mat3(1, 0));
  proto.add_mat(mat3(1, 1));
  proto.add_mat(mat3(1, 2));
  proto.add_mat(mat3(2, 0));
  proto.add_mat(mat3(2, 1));
  proto.add_mat(mat3(2, 2));
  return proto;
}

Matrix4dProto ProtoFromMatrix4d(const Matrix4d& mat4) {
  Matrix4dProto proto;
  // Matrices are packed in row-major for the proto view.
  proto.add_mat(mat4(0, 0));
  proto.add_mat(mat4(0, 1));
  proto.add_mat(mat4(0, 2));
  proto.add_mat(mat4(0, 3));
  proto.add_mat(mat4(1, 0));
  proto.add_mat(mat4(1, 1));
  proto.add_mat(mat4(1, 2));
  proto.add_mat(mat4(1, 3));
  proto.add_mat(mat4(2, 0));
  proto.add_mat(mat4(2, 1));
  proto.add_mat(mat4(2, 2));
  proto.add_mat(mat4(2, 3));
  proto.add_mat(mat4(3, 0));
  proto.add_mat(mat4(3, 1));
  proto.add_mat(mat4(3, 2));
  proto.add_mat(mat4(3, 3));
  return proto;
}

Matrix6dProto ProtoFromMatrix6d(const Matrix6d& mat6) {
  Matrix6dProto proto;
  // Matrices are packed in row-major for the proto view.
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      proto.add_mat(mat6(i, j));
    }
  }
  return proto;
}

Pose2dProto ProtoFromPose(const Pose2d& pose) {
  Pose2dProto proto;
  proto.set_tx(pose.translation().x());
  proto.set_ty(pose.translation().y());
  proto.set_rotation(pose.angle());
  return proto;
}

Pose3dProto ProtoFromPose(const Pose3d& pose) {
  Pose3dProto proto;
  proto.set_tx(pose.translation().x());
  proto.set_ty(pose.translation().y());
  proto.set_tz(pose.translation().z());

  proto.set_rw(pose.quaternion().w());
  proto.set_rx(pose.quaternion().x());
  proto.set_ry(pose.quaternion().y());
  proto.set_rz(pose.quaternion().z());
  return proto;
}

std::string Pose3dToString(const Pose3d& pose) { return Pose3ToString(pose); }

std::string VectorXdToString(const VectorXd& vector) {
  return absl::StrCat("[", SequenceToString(vector), "]");
}

std::string MatrixXdToString(const MatrixXd& matrix) {
  return absl::StrCat(
      "[", absl::StrJoin(matrix.rowwise(), "\n", VectorFormatter{}), "]");
}

std::string QuaterniondToString(const Quaterniond& quaternion) {
  return absl::StrCat("[", QuaternionToString(quaternion), "]");
}

VectorXd EigenVectorFromProto(const VectorXdProto& proto) {
  VectorXd output(proto.vec_size());
  for (int i = 0; i < proto.vec_size(); ++i) {
    output(i) = proto.vec(i);
  }
  return output;
}

}  // namespace eigenmath::conversions
