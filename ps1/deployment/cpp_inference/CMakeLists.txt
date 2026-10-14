cmake_minimum_required(VERSION 3.16)
project(OpenSetReID_Edge_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime paths (Configure ONNXRUNTIME_ROOT if installed externally)
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_ROOT})
    include_directories(${ONNXRUNTIME_DIR}/include)
    link_directories(${ONNXRUNTIME_DIR}/lib)
endif()

add_library(reid_engine_lib STATIC
    reid_engine.cpp
    reid_engine.hpp
)
target_include_directories(reid_engine_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(reid_engine_lib ${OpenCV_LIBS})

# Build standalone benchmark executable
add_executable(openset_reid_edge
    main.cpp
)
target_link_libraries(openset_reid_edge reid_engine_lib ${OpenCV_LIBS})
