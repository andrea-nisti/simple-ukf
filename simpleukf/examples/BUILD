load("@rules_cc//cc:defs.bzl", "cc_binary")
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "main_ukf",
    srcs = [
        "main_ukf.cpp",
    ],
    deps = [
       "//simpleukf/models:ctrv_models",
       "//simpleukf/ukf:ukf",
    ],
)