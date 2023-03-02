workspace(name = "KalmanCourse")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

new_git_repository(
    name = "eigen_repo",
    build_file = "//external_builds:eigen.BUILD",
    remote = "https://gitlab.com/libeigen/eigen.git",
    tag = "3.4.0",
)

http_archive(
  name = "gtest",
  urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
  strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
)