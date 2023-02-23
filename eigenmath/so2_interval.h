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

#ifndef EIGENMATH_EIGENMATH_SO2_INTERVAL_H_
#define EIGENMATH_EIGENMATH_SO2_INTERVAL_H_

#include <cmath>
#include <type_traits>

#include "scalar_utils.h"
#include "so2.h"

namespace eigenmath {
// A class expressing an interval of planar orientations as angles [from_angle,
// to_angle] in radians. The interval includes both boundary points, and
// includes the points between from_angle and to_angle by following a positive
// rotation (aka, counter-clockwise).
//
// The data structure is intended to be mainly used for containment tests, and
// intersections of interval. Calls to From() and To() are not recommended,
// since the internal representation does not guarantee that these points are
// identical to the values passed to the interval.
template <typename T>
class SO2Interval {
  static_assert(std::is_floating_point_v<T>,
                "SO2Interval only supports floating point types.");

 public:
  // Constructs an empty interval.
  SO2Interval() : SO2Interval(StateEnum::kEmpty) {}

  // Constructs an interval to include a single point.
  explicit SO2Interval(const SO2<T>& orientation)
      : SO2Interval(orientation.angle()) {}
  // Same as above, avoids conversion cost.
  explicit SO2Interval(T angle)
      : min_angle_(angle), max_angle_(angle), state_(StateEnum::kRegular) {}

  // Constructs the interval [from, to] going in positive (aka,
  // counter-clockwise) direction.  To construct an interval representing a full
  // circle, use the factory function FullCircle().  To get a single point
  // interval, prefer the single point overload.
  SO2Interval(const SO2<T>& from, const SO2<T>& to)
      : SO2Interval(from.angle(), to.angle()) {}
  // Same as above, avoids conversion cost.
  SO2Interval(T from, T to)
      : min_angle_(from),
        max_angle_(from + AngleBetween(from, to)),
        state_(StateEnum::kRegular) {}

  // Constructs an interval between `from` and `to`, using the orientation
  // specified by the sign of `direction`.  A negative sign corresponds to a
  // negative orientation (so `direction` could be an angular velocity, or
  // to curvature).
  SO2Interval(const SO2<T>& from, const SO2<T>& to, T direction)
      : SO2Interval(std::signbit(direction) ? to : from,
                    std::signbit(direction) ? from : to) {}
  SO2Interval(T from, T to, T direction)
      : SO2Interval(std::signbit(direction) ? to : from,
                    std::signbit(direction) ? from : to) {}

  // Special intervals.
  static SO2Interval Empty() { return SO2Interval(StateEnum::kEmpty); }
  static SO2Interval FullCircle() {
    return SO2Interval(StateEnum::kFullCircle);
  }

  SO2Interval(const SO2Interval&) = default;
  SO2Interval& operator=(const SO2Interval&) = default;
  SO2Interval(SO2Interval&&) = default;
  SO2Interval& operator=(SO2Interval&&) = default;

  // Boundary points if interval is not empty or full circle.
  SO2<T> From() const { return SO2<T>(min_angle_); }
  SO2<T> To() const { return SO2<T>(max_angle_); }
  T FromAngle() const { return Wrap(min_angle_, -kPi, kPi); }
  T ToAngle() const { return Wrap(max_angle_, -kPi, kPi); }

  // The circular segment covered by the interval, in radians.
  T Length() const;
  bool IsEmpty() const { return state_ == StateEnum::kEmpty; }
  bool IsFullCircle() const { return state_ == StateEnum::kFullCircle; }

  // Returns true if `orientation` is included in the interval.
  bool Contains(const SO2<T>& orientation, T tolerance) const;
  // Returns true if the (unnormalized) `angle` is included in the interval.
  bool Contains(T angle, T tolerance) const;
  // Returns true if `other` is fully contained in this interval.
  bool Contains(const SO2Interval& other, T tolerance) const;
  // Returns true if the angular segment starting at `start_angle` of signed
  // length `delta_angle` is fully contained in this interval.
  bool Contains(T start_angle, T delta_angle, T tolerance) const;

  // Returns the first interval in the (possibly non-connected) intersection
  // with other (measured inside this interval).
  SO2Interval Intersect(const SO2Interval& other) const;

 private:
  static constexpr T kTwoPi{2 * M_PI};
  static constexpr T kPi{M_PI};

  enum class StateEnum {
    kEmpty,
    kFullCircle,
    kRegular,
  };

  explicit SO2Interval(StateEnum state) : state_(state) {}

  static T Normalize(T angle) { return Wrap(angle, kTwoPi); }

  T LengthUnchecked() const { return max_angle_ - min_angle_; }

  // Accepts the interval as a full circle if the interval covers the whole
  // circle if the tolerance is applied to each side.  This is equivalent to
  // every point being contained in the interval, up to the tolerance.
  bool IsFullCircle(T tolerance) const {
    return IsFullCircle() || (kTwoPi - LengthUnchecked() <= 2 * tolerance);
  }
  static T AngleBetween(T from, T to) {
    const T delta = Normalize(to - from);
    return delta;
  }

  // Data respresentation minimizing atan2 calls.
  T min_angle_;
  T max_angle_;
  // Disambiguates empty interval, single point, and full circle.
  StateEnum state_ = StateEnum::kEmpty;
};

using SO2dInterval = SO2Interval<double>;

template <typename T>
T SO2Interval<T>::Length() const {
  if (IsEmpty()) {
    return T{0};
  }
  if (IsFullCircle()) {
    return kTwoPi;
  }
  return LengthUnchecked();
}

template <typename T>
bool SO2Interval<T>::Contains(const SO2<T>& orientation, T tolerance) const {
  return Contains(orientation.angle(), tolerance);
}

template <typename T>
bool SO2Interval<T>::Contains(T angle, T tolerance) const {
  if (IsEmpty()) {
    return false;
  }
  if (IsFullCircle(tolerance)) {
    return true;
  }
  // Align with min_angle_ and check against max_angle_.
  const T alignment = min_angle_ - tolerance;
  angle = Wrap(angle, alignment, alignment + kTwoPi);
  return angle <= max_angle_ + tolerance;
}

template <typename T>
bool SO2Interval<T>::Contains(const SO2Interval<T>& other, T tolerance) const {
  if (IsEmpty()) {
    return false;
  }
  if (IsFullCircle(tolerance)) {
    return true;
  }
  if (other.IsEmpty()) {
    return true;
  }
  if (other.IsFullCircle(tolerance)) {
    return false;
  }
  // Align with min_angle_ and check against max_angle_.
  const T alignment = min_angle_ - tolerance;
  const T aligned_min_angle =
      Wrap(other.min_angle_, alignment, alignment + kTwoPi);
  const T max = aligned_min_angle + other.LengthUnchecked();
  return max <= max_angle_ + tolerance;
}

template <typename T>
bool SO2Interval<T>::Contains(T start_angle, T delta_angle, T tolerance) const {
  if (IsEmpty()) {
    return false;
  }
  if (IsFullCircle(tolerance)) {
    return true;
  }
  // Align with min_angle_ and check against max_angle_.
  const T unaligned_min_angle =
      std::min(start_angle, start_angle + delta_angle);
  const T alignment = min_angle_ - tolerance;
  using std::abs;
  const T aligned_min_angle =
      Wrap(unaligned_min_angle, alignment, alignment + kTwoPi);
  const T aligned_max_angle = aligned_min_angle + abs(delta_angle);
  return aligned_max_angle <= max_angle_ + tolerance;
}

template <typename T>
SO2Interval<T> SO2Interval<T>::Intersect(const SO2Interval<T>& other) const {
  if (IsEmpty()) {
    return Empty();
  }
  if (IsFullCircle()) {
    return other;
  }
  if (other.IsFullCircle()) {
    return *this;
  }
  if (other.IsEmpty()) {
    return Empty();
  }

  // Align each endpoint with this interval.
  const T aligned_min_angle =
      Wrap(other.min_angle_, min_angle_, min_angle_ + kTwoPi);
  const T aligned_max_angle =
      Wrap(other.max_angle_, min_angle_, min_angle_ + kTwoPi);

  // Check for non-connected intersection.
  if (aligned_max_angle < aligned_min_angle) {
    return SO2Interval<T>{min_angle_, std::min(max_angle_, aligned_max_angle)};
  }
  if (max_angle_ < std::min(aligned_min_angle, aligned_max_angle)) {
    return Empty();
  }
  return SO2Interval<T>{std::max(min_angle_, aligned_min_angle),
                        std::min(max_angle_, aligned_max_angle)};
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_SO2_INTERVAL_H_
