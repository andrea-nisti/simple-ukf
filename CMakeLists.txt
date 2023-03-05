cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(ukf)

## Simple workaround to make Eigen work with Udacity repo
if( NOT DEFINED EIGEN3_INCLUDE_DIR )
    message(STATUS "Checking for eigen in  ${CMAKE_SOURCE_DIR}/src")
    # First check in current folder
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/Eigen)
        set( EIGEN3_INCLUDE_DIR  ${CMAKE_SOURCE_DIR}/src)
    else()
        message( FATAL_ERROR "Please set the CMake variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation. E.g. cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen/dir")
    endif()

endif()
message(STATUS "Eigen Found in ${EIGEN3_INCLUDE_DIR}")
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

add_library(models INTERFACE)
target_include_directories(models INTERFACE simpleukf/models)

add_library(simpleukf INTERFACE)
target_include_directories(simpleukf INTERFACE simpleukf/ukf)
target_link_libraries(simpleukf INTERFACE models eigen)

message(STATUS "T Include dirs for simpleukf: ${PROJECT_SOURCE_DIR}")
target_include_directories(simpleukf INTERFACE ${PROJECT_SOURCE_DIR}) # to maintain bazel include convention (e.g. includes are relative to root folder)

add_executable (ukf_main simpleukf/ukf/ukf_main.cpp)
# target_include_directories(ukf_main)
target_link_libraries(ukf_main PRIVATE simpleukf)