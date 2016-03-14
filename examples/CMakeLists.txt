add_executable(comparison comparison.cpp)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

file(COPY 
      "noisy_moons.txt"
      "comparison.py"      
    DESTINATION 
      ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(comparison clustering)

