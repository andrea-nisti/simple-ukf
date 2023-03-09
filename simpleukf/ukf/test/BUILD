load("@rules_cc//cc:defs.bzl", "cc_test", "cc_library")
package(default_visibility = ["//visibility:public"])


cc_test(
  name = "component_tests",
  srcs = ["component_tests.cpp"],
  deps = [
    "@gtest//:gtest_main",
    "//simpleukf/models:ctrv_models",
    "//simpleukf/ukf:ukf",
    ":test_utils"
  ],
)

cc_test(
  name = "strategies_tests",
  srcs = ["strategies_tests.cpp"],
  deps = [
    "@gtest//:gtest_main",
    "//simpleukf/models:ctrv_models",
    "//simpleukf/ukf:ukf",
    ":test_utils"
  ],
)

cc_library(
    name = "test_utils",
    srcs = [
        "test_utils.h"
    ],
    deps = [
       "@gtest//:gtest",
    ]
)