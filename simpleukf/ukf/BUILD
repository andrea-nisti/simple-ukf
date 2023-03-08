load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ukf",
    hdrs = [
        "linear_update_strategy.h",
        "ukf.h",
        "unscented_update_strategy.h",
    ],
    deps = [
        ":ukf_utils",
    ],
)

cc_library(
    name = "ukf_utils",
    srcs = [
        "ukf_utils.h",
    ],
    deps = [
        "@eigen_repo//:eigen",
    ],
)
