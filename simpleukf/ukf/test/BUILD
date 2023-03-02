load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])


cc_test(
  name = "ukf_component_tests",
  srcs = ["component_tests.cpp"],
  deps = [
    "@gtest//:gtest_main",
    "//simpleukf/models:crtv_models",
    "//simpleukf/ukf:ukf",
  ],
)