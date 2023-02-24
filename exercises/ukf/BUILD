load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])


cc_library(
    name = "ukf",
    srcs = [
        "ukf.h"
    ],
    hdrs = ["ukf.h"],
    deps = [
       ":ukf_utils",
    ]
)

cc_library(
    name = "ukf_utils",
    srcs = [
        "ukf_utils.cpp",
        "ukf_utils.h"
    ],
    hdrs = ["ukf_utils.h"],
    deps = [
       "@eigen_repo//:eigen"
    ]
)

cc_binary(
    name = "ukf_main",
    srcs = [
        "ukf_main.cpp",
    ],
    deps = [
       "//exercises/models",
       ":ukf",
    ],
)