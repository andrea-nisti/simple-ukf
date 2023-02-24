load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])


cc_library(
    name = "utils",
    srcs = [
        "utils.cpp",
        "utils.h"
    ],
    hdrs = ["utils.h"],
    deps = [
       "@eigen_repo//:eigen"
    ]
)

cc_binary(
    name = "jacobian_main",
    srcs = [
        "jacobian_main.cpp",
        "utils.h" 
    ],
    deps = [
       ":utils"
    ],
)

cc_binary(
    name = "rms_main",
    srcs = [
        "rms_main.cpp",
        "utils.h" 
    ],
    deps = [
       ":utils"
    ],
)