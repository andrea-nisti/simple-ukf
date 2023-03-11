load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ctrv_models",
    srcs = [
        "ctrv_model.h",
        "radar_measurement_model.h",
        "lidar_measurement_model.h"
    ],
    hdrs = ["ctrv_models.h"],
    deps = [
       "@eigen_repo//:eigen",
       ":models_utils"
    ]
)

cc_library(
    name = "models_utils",
    srcs = [
        "models_utils.h",
    ]
)
