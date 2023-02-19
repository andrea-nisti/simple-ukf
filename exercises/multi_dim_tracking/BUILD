load("@rules_cc//cc:defs.bzl", "cc_binary")
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "kalman_multi",
    srcs = [
        "main.cpp",
        "kalman_filter.cpp",
        "tracking.cpp",
        "kalman_filter.h",
        "measurement_package.h",
        "tracking.h",
    ],
    data = [":obj_pose-laser-radar-synthetic-input.txt"],
    copts = ["-Wno-unused-variable"],
    deps = [
       "@eigen_repo//:eigen"
    ],
)