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

// A Halton Sequence is a Low Discrepancey Sequence of Quasirandom numbers.
// https://en.wikipedia.org/wiki/Halton_sequence
//
// This sampler is deterministic and does not actually generate random numbers.
// It has the advantage that it improves both predictability and
// worst case performance in robotic path planners according to:
//
// Deterministic Sampling-Based Motion
// Planning: Optimality, Complexity, and
// Performance (2016) Lucas Janson et. al.
// https://arxiv.org/abs/1505.00023
//
// Since halton sequences are designed to generate points with a nice
// distribution, it is important to know that for extremely high dimension
// points this function must be modified. Structure appears in generated
// points after ~14 dimensions or higher according to lecture
// slides from the university of waterloo:
//
// http://sas.uwaterloo.ca/~dlmcleis/s906/lds.pdf
//
// Mitigation methods have been published if more dimensions are required:
//
// "Good permutations for deterministic scrambled Halton sequences
// in terms of L2-discrepancy"
// Journal of Computational and Applied Mathematics
// Volume 189, Issues 1–2, 1 May 2006, Pages 341–361
#ifndef EIGENMATH_EIGENMATH_HALTON_SEQUENCE_H_
#define EIGENMATH_EIGENMATH_HALTON_SEQUENCE_H_

#include "prime.h"
#include "type_checks.h"

namespace eigenmath {

// REQUIRES: i >= 0.
// INVARIANT: Get(0) != 0
// INVARIANT: 0 < HaltonSequence(i,j) < 1 for all i >= 0 and all 0 <= j <= 10000
// INVARIANT: prime_number a prime number used as the base of the HaltonSequence
//
// To get prime_number it is suggested the user employ LookupPrime(PrimeIndex).
//
// Returns the i'th value in the Halton sequence.
// Time complexity is O(log(i)).
template <typename FloatType = double>
FloatType HaltonSequence(int i, int prime_number = 0) {
  // ensure get(0) != 0
  i++;
  // ib = inverted base
  const FloatType ib = static_cast<FloatType>(1) / prime_number;
  FloatType cdb = ib;  // cdb = current digit base = ib ^ position
  FloatType h = 0;
  // Iterate through the base 'prime_number' digits of 'i'.
  for (; i > 0; i /= prime_number) {
    h += (i % prime_number) * cdb;
    cdb *= ib;
  }
  return h;
}

// Generates a Halton Sequence at a single specified prime index
// seed. For use in place of Random Number Generators from the
// C++11 Standard Library.
//
// C++ Concepts Implemented:
//
//        - UniformRandomBitGenerator from C++11
//          http://en.cppreference.com/w/cpp/concept/UniformRandomBitGenerator
//
//       -  Partial implementation of RandomNumberDistribution from C++11
//          http://en.cppreference.com/w/cpp/concept/RandomNumberDistribution
//
// let i be the number of times increment() was called, which is the index
// into the sequence for a given prime number input.
//
// Time complexity is O(log(i))).
// REQUIRES: i >= 0.
// INVARIANT: Get(0) != 0
// INVARIANT: 0 < Get() < 1 for all i >= 0
// INVARIANT: 0 <= DimIndex <= 10000
template <typename FloatType = double>
class HaltonSequenceEngine {
 public:
  // Partial compliance with RandomNumberDistribution::param_type
  // accesses the list of first 10k prime numbers in an array,
  // starting at index 0 with value 2
  class Seed {
   public:
    typedef int result_type;
    Seed() : prime_(LookupPrime(0)) {}

    // index of the prime number to be used as the seed
    explicit Seed(result_type value) : prime_(LookupPrime(value)) {}

   private:
    friend class HaltonSequenceEngine;
    int prime_;
  };

  typedef FloatType result_type;
  typedef Seed param_type;

  explicit HaltonSequenceEngine() {}

  explicit HaltonSequenceEngine(param_type param) : param_(param) {}

  // Returns the i'th value in the Halton sequence at the
  // specified prime dimension index, where
  //
  //       index 0 is the prime number 2
  //       index 1 is the prime number 3
  //       etc.
  //
  // Note that this function differs because it is indexed
  // on the prime sequence dimension and uses the internal
  // halton sequence index at runtime.
  //
  // Time complexity is O(log(i)).
  // REQUIRES: i >= 0.
  // INVARIANT: Get(0) != 0
  // INVARIANT: 0 < Get(i) < 1 for all i >= 0
  // INVARIANT: 0 <= dim <= 14
  result_type GetDim(int dim) {
    return HaltonSequence<result_type>(sequence_index_, LookupPrime(dim));
  }

  // Returns the i'th value in the Halton sequence at the
  // specified prime dimension index, where
  //
  //       index 0 is the prime number 2
  //       index 1 is the prime number 3
  //       etc.
  result_type Get(int i, int dim) {
    return HaltonSequence<result_type>(i, LookupPrime(dim));
  }

  // Returns the i'th value in the Halton sequence at the
  // specified prime dimension index, where
  //
  //       index 0 is the prime number 2
  //       index 1 is the prime number 3
  //       etc.
  template <int DimIndex = 0>
  result_type Get(int i) {
    return HaltonSequence<result_type>(i, LookupPrime(DimIndex));
  }

  // Returns the current value in the Halton sequence
  // based on the number of calles to increment() at the
  // specified prime dimension index, where
  //
  //       index 0 is the prime number 2
  //       index 1 is the prime number 3
  //       etc.
  template <int DimIndex = 0>
  result_type Get() {
    return HaltonSequence<result_type>(sequence_index_, LookupPrime(DimIndex));
  }

  // Generates the next value in the Halton sequence
  result_type operator()() {
    result_type result =
        HaltonSequence<result_type>(sequence_index_, param_.prime_);
    sequence_index_++;
    return result;
  }

  // Generates the value at the specified Halton sequence index i
  // does not modify the current index
  result_type operator()(int i) {
    result_type result = HaltonSequence<result_type>(i, param_.prime_);
    return result;
  }

  // primeIndex is an index into an array of the first 10,000 prime numbers.
  void SeedWithNthPrime(int primeIndex) { param_ = param_type(primeIndex); }

  void Seed(param_type param) { param_ = param; }

  void Increment() { sequence_index_++; }

  void Advance(int n) { sequence_index_ += n; }

  void Reset() { sequence_index_ = 0; }

  result_type Min() { return 0; }
  result_type Max() { return 1; }

 private:
  int sequence_index_ = 0;
  param_type param_;
};

// Halton Sequence Generator of multi-dimension Eigen points using
// the C++ Standard Library Concept "Generator"
//
// See the following function for algorithm details and constraints:
//
//          template<typename FloatType = double>
//          FloatType HaltonSequence(int i, int primeIndex = 0)
//
// C++ Concepts Implemented:
//
//        - UniformRandomBitGenerator from C++11
//          http://en.cppreference.com/w/cpp/concept/UniformRandomBitGenerator
//
// For example, it can be used with:
//
// std::generate()
// http://en.cppreference.com/w/cpp/algorithm/generate
//
// INVARIANT: PointType must be DefaultConstructible.
template <typename PointType>
class HaltonPointEngine {
 public:
  typedef PointType result_type;

  explicit HaltonPointEngine(int point_dimensions)
      : point_dimensions_(point_dimensions) {}

  result_type operator()() {
    result_type point(point_dimensions_);
    for (int i = 0; i < point_dimensions_; ++i) {
      point(i) = sampler_.GetDim(i);
    }
    sampler_.Increment();
    return point;
  }

  result_type Min() {
    result_type point(PointType::Zero(point_dimensions_));
    return point;
  }

  result_type Max() {
    result_type point(PointType::Ones(point_dimensions_));
    return point;
  }

  void Increment() { sampler_.Increment(); }

  void Advance(int n) { sampler_.Advance(n); }

  void Reset() { sampler_.Reset(); }

 private:
  int point_dimensions_;
  // HaltonSequenceEngine of the type stored for each coordinate (eg double,
  // float, int etc).
  HaltonSequenceEngine<ScalarTypeOf<PointType>> sampler_;
};

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_HALTON_SEQUENCE_H_
