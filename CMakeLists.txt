cmake_minimum_required(VERSION 2.6)

project(masc_clustering_lib)
set(PACKAGE_VERSION "0.1.0")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 3.1.2 REQUIRED)

include_directories(  
  ${EIGEN3_INCLUDE_DIR}
)

########## set compiler flags ##########
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

########## add subdirectories ############
add_subdirectory(libclustering)
add_subdirectory(examples)
