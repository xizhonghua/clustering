set(LIB_SRC
    Clustering.cpp
    util/EigenIOHelper.cpp
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}  
  "${CMAKE_SOURCE_DIR}/libclustering"
)

add_library(clustering ${LIB_SRC})
