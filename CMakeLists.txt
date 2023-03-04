cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(ukf)

## Simple workaround to make Eigen work with Udacity repo
if( NOT DEFINED EIGEN3_INCLUDE_DIR )

    # First check in current folder
    if(EXISTS ${PROJECT_SOURCE_DIR}/Eigen)
        set( EIGEN3_INCLUDE_DIR Eigen)
    else()
        message( FATAL_ERROR "Please set the CMake variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation. E.g. cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen/dir")
    endif()

endif()

message(STATUS "Eigen Found in ${EIGEN3_INCLUDE_DIR}")

add_library(models STATIC simpleukf/models/models_utils.h simpleukf/models/crtv/crtv_model.h simpleukf/models/crtv/radar_measurement_model.h)
set_target_properties(models PROPERTIES LINKER_LANGUAGE CXX)
include_directories(models PUBLIC ${EIGEN3_INCLUDE_DIR})

add_library(simpleukf STATIC simpleukf/ukf/ukf_utils.h simpleukf/ukf/ukf.h )
set_target_properties(simpleukf PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(simpleukf PUBLIC models)
target_include_directories(simpleukf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/simpleukf)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # to maintain bazel include convention (e.g. includes are relative to root folder)

add_executable (ukf_main simpleukf/ukf/ukf_main.cpp)
target_link_libraries(ukf_main PUBLIC simpleukf)