cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(ukf)

add_library(simpleukf STATIC simpleukf/ukf/ukf_utils.h simpleukf/ukf/ukf.h )
# target_include_directories(simpleukf PUBLIC simpleukf/ukf)

add_library(models STATIC simpleukf/models/models_utils.h simpleukf/models/crtv/crtv_model.h simpleukf/models/crtv/radar_measurement_model.h)
# target_include_directories(models PUBLIC simpleukf/models)

target_include_directories(models PUBLIC /home/andre/.cache/bazel/_bazel_andre/c04b8793f4717ff643e41619e852db81/external/eigen_repo)
target_link_libraries(simpleukf PUBLIC models)

set_target_properties(models PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(simpleukf PROPERTIES LINKER_LANGUAGE CXX)

include_directories(./) # to maintain bazel include convention (e.g. includes are relative to root folder)
add_executable (ukf_main simpleukf/ukf/ukf_main.cpp)
target_link_libraries(ukf_main PUBLIC simpleukf)