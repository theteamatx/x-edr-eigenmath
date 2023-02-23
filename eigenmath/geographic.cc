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

#include "geographic.h"

#include <vector>

#include "Eigen/Core"
#include "pose3.h"
#include "third_party/geographiclib/include/GeographicLib/Geocentric.hpp"
#include "types.h"

namespace {

inline Quaterniond GeographicLibRotationMatrixToQuaternion(
    const std::vector<double>& M) {
  // GeographicLib rotation matrices are row-major. Eigen is col-major by
  // default.
  return Quaterniond(Eigen::Map<const Matrix3d>(M.data()).transpose());
}

}  // namespace

GeoPoint::GeoPoint(const double latitude, const double longitude,
                   const double elevation)
    : latitude_(latitude), longitude_(longitude), elevation_(elevation) {}

GeoPoint::GeoPoint(const Vector3d& ecef_point) {
  GeographicLib::Geocentric::WGS84().Reverse(ecef_point.x(), ecef_point.y(),
                                             ecef_point.z(), latitude_,
                                             longitude_, elevation_);
}

Vector3d GeoPoint::ToEcef() const {
  Vector3d ecef;
  GeographicLib::Geocentric::WGS84().Forward(latitude_, longitude_, elevation_,
                                             ecef.x(), ecef.y(), ecef.z());
  return ecef;
}

GeoPose::GeoPose(const GeoPoint& geo_point, const Quaterniond& enu_orientation)
    : point_(geo_point), enu_orientation_(enu_orientation) {}

GeoPose::GeoPose(const Pose3d& ecef_pose) {
  std::vector<double> M(9);
  GeographicLib::Geocentric::WGS84().Reverse(
      ecef_pose.translation().x(), ecef_pose.translation().y(),
      ecef_pose.translation().z(), point_.latitude(), point_.longitude(),
      point_.elevation(), M);
  enu_orientation_ = GeographicLibRotationMatrixToQuaternion(M).inverse() *
                     ecef_pose.quaternion();
}

Pose3d GeoPose::ToEcef() const {
  Pose3d ecef;
  std::vector<double> M(9);
  GeographicLib::Geocentric::WGS84().Forward(
      point_.latitude(), point_.longitude(), point_.elevation(),
      ecef.translation().x(), ecef.translation().y(), ecef.translation().z(),
      M);
  ecef.setQuaternion(GeographicLibRotationMatrixToQuaternion(M) *
                     enu_orientation_);
  return ecef;
}
