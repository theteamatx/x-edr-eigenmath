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
#include <random>
#include <vector>

#include "eigenmath/distribution.h"
#include "eigenmath/eigenmath.pb.h"
#include "eigenmath/matchers.h"
#include "eigenmath/pose2.h"
#include "eigenmath/sampling.h"
#include "eigenmath/types.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_field.h"
#include "gtest/gtest.h"

namespace eigenmath::conversions {
namespace {
using testing::IsApprox;
using testing::IsApproxEigenVector;

constexpr double kTolerance = 1e-06;

class ConversionsTest : public ::testing::Test {
 public:
  ConversionsTest() : random_engine_(TestGenerator(kGeneratorTestSeed)) {}

 protected:
  // Tests that roundtrip conversion has same size and value.
  template <class Scalar, int N, int MaxSize>
  void EigenRoundTripConversion(
      VectorFixedOrDynamic<Scalar, N, MaxSize> vector) {
    VectorFixedOrDynamic<Scalar, N, MaxSize> roundtrip_vector = vector;
    vector.setZero();
    google::protobuf::RepeatedField<Scalar> proto_vector;
    ProtoFromEigenVector(vector, &proto_vector);

    // conversions with expected size
    bool conversion_ok =
        EigenVectorFromProto(proto_vector, vector.size(), &roundtrip_vector);
    ASSERT_TRUE(conversion_ok);
    EXPECT_THAT(roundtrip_vector, IsApproxEigenVector(vector, kTolerance));

    // conversions without expected size
    roundtrip_vector = EigenVectorFromProto(proto_vector);
    EXPECT_THAT(roundtrip_vector, IsApproxEigenVector(vector, kTolerance));
  }

  // Check that an empty proto converts to a zero vector.
  template <class ProtoType, int N>
  void EmptyVectorProtoConversion() {
    Vector<double, N> vector = EigenVectorFromProto(ProtoType());
    EXPECT_THAT(vector, IsApprox(Vector<double, N>::Zero(), kTolerance));
  }

  // Check that an empty proto converts to a zero matrix.
  template <class ProtoType, int N>
  void EmptyMatrixProtoConversion() {
    Matrix<double, N, N> matrix = EigenMatrixFromProto(ProtoType());
    EXPECT_THAT(matrix, IsApprox(Matrix<double, N, N>::Zero(), kTolerance));
  }

  // Helper to generate a seeded-random matrix of templated type and given size.
  template <typename MatrixType>
  MatrixType MakeRandom(int row_size, int col_size) {
    using ScalarType = typename MatrixType::Scalar;
    std::uniform_real_distribution<double> uniform_scalar(-1.0, 1.0);
    MatrixType matrix(row_size, col_size);
    for (int i = 0; i < matrix.size(); ++i) {
      matrix.reshaped()(i) =
          static_cast<ScalarType>(uniform_scalar(random_engine_));
    }
    return matrix;
  }
  // Helper to generate a seeded-random matrix of templated fixed-size type.
  template <typename MatrixType>
  MatrixType MakeRandom() {
    return MakeRandom<MatrixType>(MatrixType::RowsAtCompileTime,
                                  MatrixType::ColsAtCompileTime);
  }


  // Helper to generate a seeded-random vector of templated type and given size.
  template <typename VectorType>
  VectorType MakeRandom(int size) {
    return MakeRandom<VectorType>(size, VectorType::ColsAtCompileTime);
  }
  // Random engine used with a seed to generate number sequences.
  TestGenerator random_engine_;
};

template <>
Pose2d ConversionsTest::MakeRandom<Pose2d>() {
  std::uniform_real_distribution<double> uniform_scalar(-1.0, 1.0);
  return Pose2d(MakeRandom<Vector2d>(),
                uniform_scalar(random_engine_) * M_PI);
}

TEST_F(ConversionsTest, EigenVectorToFromProto) {
  // Round trip on different dynamic vector sizes and types.
  EigenRoundTripConversion(VectorXd());
  EigenRoundTripConversion(MakeRandom<VectorXd>(6));
  EigenRoundTripConversion(MakeRandom<VectorXd>(20));
  EigenRoundTripConversion(MakeRandom<VectorXf>(6));
  EigenRoundTripConversion(MakeRandom<VectorXb>(6));
  EigenRoundTripConversion(MakeRandom<VectorNd>(6));

  // Round trip on different static vector sizes and types.
  EigenRoundTripConversion(MakeRandom<Vector6d>());
  EigenRoundTripConversion(MakeRandom<Vector6f>());
  EigenRoundTripConversion(MakeRandom<Vector4d>());
  EigenRoundTripConversion(MakeRandom<Vector4f>());
  EigenRoundTripConversion(MakeRandom<Vector3d>());
  EigenRoundTripConversion(MakeRandom<Vector3f>());
  EigenRoundTripConversion(MakeRandom<Vector2d>());
  EigenRoundTripConversion(MakeRandom<Vector2f>());
  EigenRoundTripConversion(MakeRandom<Vector6b>());
}

TEST_F(ConversionsTest, Vector2dProtoRoundTrip) {
  Vector2d vec_in = MakeRandom<Vector2d>();
  Vector2dProto proto = ProtoFromVector2d(vec_in);
  Vector2d vec_out = EigenVectorFromProto(proto);
  EXPECT_THAT(vec_out, IsApproxEigenVector(vec_in, kTolerance));
}

TEST_F(ConversionsTest, EmptyVectors) {
  EmptyVectorProtoConversion<Vector2dProto, 2>();
  EmptyVectorProtoConversion<Vector3dProto, 3>();
  EmptyVectorProtoConversion<Vector4dProto, 4>();
  EmptyVectorProtoConversion<Vector6dProto, 6>();
}

TEST_F(ConversionsTest, EmptyMatrices) {
  EmptyMatrixProtoConversion<Matrix2dProto, 2>();
  EmptyMatrixProtoConversion<Matrix3dProto, 3>();
  EmptyMatrixProtoConversion<Matrix4dProto, 4>();
  EmptyMatrixProtoConversion<Matrix6dProto, 6>();
}

TEST_F(ConversionsTest, Vector3dProtoRoundTrip) {
  Vector3d vec_in = MakeRandom<Vector3d>();
  Vector3dProto proto = ProtoFromVector3d(vec_in);
  Vector3d vec_out = EigenVectorFromProto(proto);
  EXPECT_THAT(vec_out, IsApproxEigenVector(vec_in, kTolerance));
}

TEST_F(ConversionsTest, Vector4dProtoRoundTrip) {
  Vector4d vec_in = MakeRandom<Vector4d>();
  Vector4dProto proto = ProtoFromVector4d(vec_in);
  Vector4d vec_out = EigenVectorFromProto(proto);
  EXPECT_THAT(vec_out, IsApproxEigenVector(vec_in, kTolerance));
}

TEST_F(ConversionsTest, Vector6dProtoRoundTrip) {
  Vector6d vec_in = MakeRandom<Vector6d>();
  Vector6dProto proto = ProtoFromVector6d(vec_in);
  Vector6d vec_out = EigenVectorFromProto(proto);
  EXPECT_THAT(vec_out, IsApproxEigenVector(vec_in, kTolerance));
}

TEST_F(ConversionsTest, VectorNdProtoRoundTrip) {
  VectorNd vec_in = MakeRandom<VectorNd>(5);
  VectorNdProto proto = ProtoFromVectorNd(vec_in);
  VectorNd vec_out = EigenVectorFromProto(proto);
  EXPECT_THAT(vec_out, IsApproxEigenVector(vec_in, kTolerance));
}

TEST_F(ConversionsTest, VectorNdProtoOversize) {
  VectorXd vec_in = MakeRandom<VectorXd>(kMaxEigenVectorCapacity + 1);
  VectorNdProto proto = ProtoFromVectorNd(vec_in);
  EXPECT_DEATH(
      {
        // Assignment of too big proto to fixed size vector should assert.
        VectorNd vec_out = EigenVectorFromProto(proto);
        // Next line should never occur, just to use the output.
        std::cout << vec_out;
      },
      ".*");
}

TEST_F(ConversionsTest, SequenceToProto) {
  std::vector<double> vec_in = {6.00613, 1.5, 64.347, 4.7, 40.8075};
  VectorNdProto proto;
  ASSERT_TRUE(ProtoFromSequence(vec_in, proto.mutable_vec()));
  ASSERT_EQ(proto.vec_size(), vec_in.size());
  for (int c = 0; c < vec_in.size(); ++c) {
    EXPECT_NEAR(proto.vec(c), vec_in[c], kTolerance) << "Index: " << c;
  }
}

TEST_F(ConversionsTest, Matrix2dProtoRoundTrip) {
  Matrix2d mat_in = MakeRandom<Matrix2d>();
  Matrix2dProto proto = ProtoFromMatrix2d(mat_in);
  Matrix2d mat_out = EigenMatrixFromProto(proto);
  EXPECT_THAT(mat_out, IsApprox(mat_in, kTolerance));
}

TEST_F(ConversionsTest, Matrix3dProtoRoundTrip) {
  Matrix3d mat_in = MakeRandom<Matrix3d>();
  Matrix3dProto proto = ProtoFromMatrix3d(mat_in);
  Matrix3d mat_out = EigenMatrixFromProto(proto);
  EXPECT_THAT(mat_out, IsApprox(mat_in, kTolerance));
}

TEST_F(ConversionsTest, Matrix4dProtoRoundTrip) {
  Matrix4d mat_in = MakeRandom<Matrix4d>();
  Matrix4dProto proto = ProtoFromMatrix4d(mat_in);
  Matrix4d mat_out = EigenMatrixFromProto(proto);
  EXPECT_THAT(mat_out, IsApprox(mat_in, kTolerance));
}

TEST_F(ConversionsTest, Matrix6dProtoRoundTrip) {
  Matrix6d mat_in = MakeRandom<Matrix6d>();
  Matrix6dProto proto = ProtoFromMatrix6d(mat_in);
  // Confirm row-major packing.
  EXPECT_EQ(proto.mat(1), mat_in(0, 1));
  Matrix6d mat_out = EigenMatrixFromProto(proto);
  EXPECT_THAT(mat_out, IsApprox(mat_in, kTolerance));
}

TEST_F(ConversionsTest, PoseToFromProto) {
  UniformDistributionPose3<> pose_dist;
  Pose3d eigen_pose = pose_dist(random_engine_);
  Pose3dProto proto_pose = ProtoFromPose(eigen_pose);
  Pose3d roundtrip_pose;
  bool conversion_ok = PoseFromProto(proto_pose, &roundtrip_pose);
  EXPECT_TRUE(conversion_ok);
  EXPECT_NEAR(eigen_pose.translation().x(), roundtrip_pose.translation().x(),
              kTolerance);
  EXPECT_NEAR(eigen_pose.translation().y(), roundtrip_pose.translation().y(),
              kTolerance);
  EXPECT_NEAR(eigen_pose.translation().z(), roundtrip_pose.translation().z(),
              kTolerance);
  EXPECT_NEAR(eigen_pose.quaternion().w(), roundtrip_pose.quaternion().w(),
              kTolerance);
  EXPECT_NEAR(eigen_pose.quaternion().x(), roundtrip_pose.quaternion().x(),
              kTolerance);
  EXPECT_NEAR(eigen_pose.quaternion().y(), roundtrip_pose.quaternion().y(),
              kTolerance);
  EXPECT_NEAR(eigen_pose.quaternion().z(), roundtrip_pose.quaternion().z(),
              kTolerance);
}

TEST_F(ConversionsTest, PoseFromProtoNullQuaternion) {
  Pose3dProto proto_pose;
  proto_pose.set_tx(1);
  proto_pose.set_ty(2);
  proto_pose.set_tz(3);
  // Quaternion ctor uses (w, x, y, z).
  Pose3d pose(Quaterniond(0, 1, 0, 0), Vector3d(0, 0, 0));

  ASSERT_EQ(pose.quaternion().w(), 0);
  ASSERT_EQ(pose.quaternion().x(), 1);
  ASSERT_EQ(pose.quaternion().y(), 0);
  ASSERT_EQ(pose.quaternion().z(), 0);

  EXPECT_TRUE(PoseFromProto(proto_pose, &pose));
  EXPECT_EQ(pose.translation().x(), 1.0);
  EXPECT_EQ(pose.translation().y(), 2.0);
  EXPECT_EQ(pose.translation().z(), 3.0);
  EXPECT_EQ(pose.quaternion().x(), 0.0);
  EXPECT_EQ(pose.quaternion().y(), 0.0);
  EXPECT_EQ(pose.quaternion().z(), 0.0);
  EXPECT_EQ(pose.quaternion().w(), 1.0);
}

TEST_F(ConversionsTest, QuaternionToFromProto) {
  const Quaterniond input = Quaterniond(MakeRandom<Vector4d>());
  QuaterniondProto proto = ProtoFromQuaternion(input);
  const Quaterniond output = QuaternionFromProto(proto);
  EXPECT_NEAR(input.w(), output.w(), kTolerance);
  EXPECT_NEAR(input.x(), output.x(), kTolerance);
  EXPECT_NEAR(input.y(), output.y(), kTolerance);
  EXPECT_NEAR(input.z(), output.z(), kTolerance);
}

TEST_F(ConversionsTest, Pose3dToString) {
  Pose3d eigen_pose(Quaterniond(1, 0, 0, 0), Vector3d(1, 2, 3));
  EXPECT_EQ(Pose3dToString(eigen_pose), "[1, 2, 3, 1, 0, 0, 0]");
}

TEST_F(ConversionsTest, Pose3ToString) {
  Pose3d eigen_posed(Quaterniond(1, 0, 0, 0), Vector3d(1, 2, 3));
  EXPECT_EQ(Pose3ToString(eigen_posed), "[1, 2, 3, 1, 0, 0, 0]");

  Pose3f eigen_posef(Quaternionf(1, 0, 0, 0), Vector3f(1, 2, 3));
  EXPECT_EQ(Pose3ToString(eigen_posef), "[1, 2, 3, 1, 0, 0, 0]");
}

// Test and verify that quaternions are normalized after parsing.
TEST_F(ConversionsTest, Pose3dQuaternionNormalized) {
  Pose3dProto proto;
  proto.set_rx(0);
  proto.set_rw(0);
  proto.set_rz(0);
  proto.set_rw(0.999);
  Pose3d pose = PoseFromProto(proto);
  EXPECT_NEAR(pose.quaternion().norm(), 1.0,
              std::numeric_limits<double>::epsilon());
  Pose3d pose2;
  EXPECT_TRUE(PoseFromProto(proto, &pose2));
  EXPECT_NEAR(pose2.quaternion().norm(), 1.0,
              std::numeric_limits<double>::epsilon());
}

TEST_F(ConversionsTest, VectorXdToString) {
  Vector3d vec(1, 2, 3);
  EXPECT_EQ(VectorXdToString(vec), "[1, 2, 3]");
}

TEST_F(ConversionsTest, QuaterniondToString) {
  Quaterniond quaternion(0.5, 0.5, 0.5, 0.5);
  EXPECT_EQ(QuaterniondToString(quaternion), "[0.5, 0.5, 0.5, 0.5]");
}

TEST_F(ConversionsTest, MatrixXdToString) {
  Matrix2d matrix = Matrix2d::Zero();
  matrix << 1, 2, 3, 4;
  EXPECT_EQ(MatrixXdToString(matrix), "[[1, 2]\n[3, 4]]");
}

TEST_F(ConversionsTest, VectorXd) {
  VectorXd vector_x_d(20);
  for (int i = 0; i < vector_x_d.size(); ++i) {
    vector_x_d(i) = i;
  }
  VectorXdProto vector_proto = ProtoFromVectorXd(vector_x_d);
  VectorXd converted_vector = EigenVectorFromProto(vector_proto);
  EXPECT_TRUE(converted_vector.isApprox(vector_x_d));
}

TEST_F(ConversionsTest, Pose2dProtoRoundTrip) {
  Pose2d vec_in = MakeRandom<Pose2d>();
  Pose2dProto proto = ProtoFromPose(vec_in);
  const Pose2d vec_out = PoseFromProto(proto);
  EXPECT_THAT(vec_out, IsApprox(vec_in, kTolerance));
}

}  // namespace
}  // namespace eigenmath::conversions
