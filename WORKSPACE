workspace(name = "KalmanCourse")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

new_git_repository(
    name = "eigen_repo",
    build_file = "//external_builds:eigen.BUILD",
    remote = "https://gitlab.com/libeigen/eigen.git",
    tag = "3.4.0",
)


