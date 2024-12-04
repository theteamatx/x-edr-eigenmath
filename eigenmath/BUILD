# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")

licenses(["notice"])

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "eigenmath",
    srcs = [
        "covariance.cc",
        "quadratic_optimization.cc",
        "quadrature.cc",
        "quintic_spline.cc",
        "spline_factories.cc",
        "spline_fit.cc",
        "vector_utils.cc",
    ],
    hdrs = [
        "constants.h",
        "covariance.h",
        "distribution.h",
        "halton_sequence.h",
        "interpolation.h",
        "line_search.h",
        "line_utils.h",
        "manifolds.h",
        "mean_and_covariance.h",
        "normal_distribution.h",
        "numerical_derivatives.h",
        "plane_conversions.h",
        "pose2.h",
        "pose3.h",
        "pose3_utils.h",
        "pose_and_covariance3_utils.h",
        "prime.h",
        "quadratic_optimization.h",
        "quadrature.h",
        "quintic_spline.h",
        "quintic_spline_segment.h",
        "rotation_utils.h",
        "scalar_utils.h",
        "similarity3.h",
        "simple_filters.h",
        "so2.h",
        "so2_interval.h",
        "so3.h",
        "spline_coefficients.h",
        "spline_factories.h",
        "spline_fit.h",
        "type_checks.h",
        "types.h",
        "utils.h",
        "vector_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@eigen",
        "@x_edr_genit//genit:iterators",
    ],
)

proto_library(
    name = "eigenmath_proto",
    srcs = [
        "eigenmath.proto",
    ],
)

cc_proto_library(
    name = "eigenmath_cc_proto",
    deps = [":eigenmath_proto"],
)

cc_library(
    name = "conversions",
    srcs = [
        "conversions.cc",
    ],
    hdrs = [
        "conversions.h",
    ],
    deps = [
        ":eigenmath",
        ":eigenmath_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "conversions_test",
    srcs = [
        "conversions_test.cc",
    ],
    deps = [
        ":conversions",
        ":eigenmath_cc_proto",
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    hdrs = [
        "pose3_test_utils.h",
        "test_utils.h",
    ],
    deps = [":eigenmath"],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        ":eigenmath",
        "@eigen",
        "@com_google_googletest//:gtest",
        "@x_edr_genit//genit:iterators",
    ],
)

cc_library(
    name = "incremental_stats",
    hdrs = ["incremental_stats.h"],
    deps = ["@com_google_absl//absl/time"],
)

cc_library(
    name = "sampling",
    hdrs = ["sampling.h"],
)

cc_test(
    name = "sampling_test",
    srcs = ["sampling_test.cc"],
    deps = [
        ":sampling",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "covariance_test",
    srcs = ["covariance_test.cc"],
    deps = [
        ":eigenmath",
        ":matchers",
        ":test_constants",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "alignment_test",
    srcs = [
        "alignment_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "constants_test",
    srcs = [
        "constants_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "halton_sequence_test",
    srcs = [
        "halton_sequence_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "incremental_stats_test",
    srcs = [
        "incremental_stats_test.cc",
    ],
    deps = [
        ":incremental_stats",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "interpolation_test",
    srcs = [
        "interpolation_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "line_search_test",
    srcs = [
        "line_search_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "line_utils_test",
    srcs = ["line_utils_test.cc"],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "manifolds_test",
    srcs = [
        "manifolds_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "matchers_test",
    srcs = ["matchers_test.cc"],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mean_and_covariance_test",
    srcs = [
        "mean_and_covariance_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "normal_distribution_test",
    srcs = [
        "normal_distribution_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "numerical_derivatives_test",
    srcs = [
        "numerical_derivatives_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "plane_conversions_test",
    srcs = [
        "plane_conversions_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "pose2_test",
    srcs = [
        "pose2_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "pose3_test",
    srcs = [
        "pose3_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pose3_utils_test",
    srcs = [
        "pose3_utils_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pose_and_covariance3_utils_test",
    srcs = [
        "pose_and_covariance3_utils_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "prime_test",
    srcs = [
        "prime_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quadratic_optimization_test",
    srcs = [
        "quadratic_optimization_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quadrature_test",
    srcs = [
        "quadrature_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quintic_spline_test",
    srcs = [
        "quintic_spline_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rotation_utils_test",
    srcs = [
        "rotation_utils_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "scalar_utils_test",
    srcs = [
        "scalar_utils_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":sampling",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "similarity3_test",
    srcs = [
        "similarity3_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simple_filters_test",
    srcs = [
        "simple_filters_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "so2_test",
    srcs = [
        "so2_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_absl//absl/random:distributions",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "so2_interval_test",
    srcs = ["so2_interval_test.cc"],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_absl//absl/random:distributions",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "so3_test",
    srcs = [
        "so3_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "spline_fit_test",
    srcs = [
        "spline_fit_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "distribution_test",
    srcs = [
        "distribution_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "type_checks_test",
    srcs = [
        "type_checks_test.cc",
    ],
    deps = [
        ":eigenmath",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "vector_utils_test",
    srcs = [
        "vector_utils_test.cc",
    ],
    deps = [
        ":eigenmath",
        ":matchers",
        ":sampling",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_constants",
    testonly = 1,
    srcs = [
        "test_constants.cc",
    ],
    hdrs = [
        "test_constants.h",
    ],
    deps = [
        ":eigenmath",
    ],
)

cc_library(
    name = "compare_vectors",
    hdrs = ["compare_vectors.h"],
    deps = [
        "@com_google_absl//absl/types:span"
    ],
)

cc_test(
    name = "compare_vectors_test",
    srcs = [
        "compare_vectors_test.cc",
    ],
    deps = [
        ":compare_vectors",
        ":eigenmath",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quasi_random_vector",
    srcs = ["quasi_random_vector.cc"],
    hdrs = ["quasi_random_vector.h"],
    deps = [
        ":eigenmath",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "quasi_random_vector_test",
    srcs = ["quasi_random_vector_test.cc"],
    deps = [
        ":quasi_random_vector",
        "@com_google_googletest//:gtest_main",
    ],
)
