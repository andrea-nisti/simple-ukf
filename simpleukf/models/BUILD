load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "crtv_models",
    srcs = [
        "crtv/crtv_model.h",
        "crtv/radar_measurement_model.h"
    ],
    hdrs = ["crtv_models.h"],
    deps = [
       "@eigen_repo//:eigen"
    ]
)