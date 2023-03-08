load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ukf",
    hdrs = [
        "ukf.h",
        "linear_update_strategy.h"
    ],
    deps = [
       ":ukf_utils",
    ]
)

cc_library(
    name = "ukf_utils",
    srcs = [
        "ukf_utils.h"
    ],
    deps = [
       "@eigen_repo//:eigen"
    ]
)