# eigenmath

[TOC]

This package contains common math functionality useful in higher level robotics
algorithms, implemented using Eigen.

Commonly used datatypes include
 - vectors: see eigenmath/types.h
 - rotations: see eigenmath::SO2|3
 - poses: see eigenmath::Pose2|3

## Random values

### Tests

To test functions on eigenmath data types, use the
provided tools (see eigenmath/\(distribution.h|sampling.h|matchers.h\))
for unit tests. Tests should be deterministic. Other tests
exploring a range of inputs should be excluded from test suites, and should be
run manually. Using random values in tests should only be
seen as a basic check of some functionality. If failing inputs are discovered
in production or fuzzing, add those inputs as separate test cases.

The recommended use of sampling in tests is shown in
eigenmath/vector\_utils\_test.cc at ExtendToOrthonormalBasisRandomValues

Although the eigenmath library is developed on top of Eigen, avoid using Eigen's
sampling methods, including `Random()` and `setRandom()`. In particular, use a
named pseudo random number generator and a fixed seed in your test. For example,
if you have a test

```c++ {.bad}
TEST(SomeFunction, BasicTest) {
  for (int i = 0; i < 100; ++i) {
    ...
    Vector3d x = Vector3d::Random().normalized();
    ...
  }
}
```

you can change it to

```c++ {.good}
TEST(SomeFunction, BasicTestRandomValues) {
  auto generator = TestGenerator(kGeneratorTestSeed);
  for (int i = 0; i < 100; ++i) {
    ...
    UniformDistributionUnitVector3d unit_vector_dist;
    Vector3d x = unit_vector_dist(generator);
    ...
  }
}
```

and add an optional test for edge cases

```c++ {.good}
TEST(SomeFunction, BasicTestEdgeCases) {
  const Vector3d edge_cases[] = {{0.0, 1.0, -1.0}};
  for (const Vector3d& x : edge_cases) {
    ...
  }
}
```

### Algorithms

If you need randomness in an algorithm, consider whether a sequence such as the
Halton sequence (see eigenmath::HaltonSequence) would be
appropriate for your use case. Try to handle the sequence's state so that the
behavior in production can be reproducable. This is especially
important for debugging. Also, when using multiple sequences (or
using multiple samples from a sequence, for example to build a vector or pose),
ensure that you do not suffer from dependencies between the samples.

If you _must_ use a random number generator in an algorithm, consult
go/cpp-random-numbers. The random number tools provided here are not checked for
suitability outside tests.

## Scalar types and implementation details

Types in eigenmath are templated on the scalar type, e.g.
`eigenmath::Pose3<Scalar>`. Most commonly, `Scalar` is instantiated with float
or double. However, this library is designed in a way so that most types and
functions can be used inside Ceres' optimization loop and especially with Ceres
autodifferentiation. In this case, `Scalar` is instantiated with the templated
type `ceres::Jet<T, N>`. See include/ceres/jet.h for details.

Therefore, it is a common pattern in this library to use using-declarations for
common math functions (`abs`, `cos`, `sin`, ...), since the correspsonding
function for `ceres::Jet<T, N>` are not defined in the `std` namespace. Instead,
they are defined in the `ceres` namespace so they can be found using
argument-dependent lookup (ADL).
