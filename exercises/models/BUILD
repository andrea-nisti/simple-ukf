load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "models",
    srcs = [
        "crtv_model.h",
        "radar_measurement_model.h"
    ],
    hdrs = ["models.h"],
    deps = [
       "@eigen_repo//:eigen"
    ]
)