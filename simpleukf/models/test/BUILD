load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])


cc_test(
  name = "models_tests",
  srcs = ["models_tests.cpp"],
  deps = [
    "@gtest//:gtest_main",
    "//simpleukf/models:crtv_models",
    "//simpleukf/models:models_strategies",
  ],
)