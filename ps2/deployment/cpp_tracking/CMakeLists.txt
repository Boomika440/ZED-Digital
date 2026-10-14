cmake_minimum_required(VERSION 3.16)
project(OAR_Tracker_Edge_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

add_library(oar_tracker_lib STATIC
    kalman_filter.cpp
    kalman_filter.hpp
    hungarian.cpp
    hungarian.hpp
    track.cpp
    track.hpp
    oar_tracker.cpp
    oar_tracker.hpp
    types.hpp
)

target_include_directories(oar_tracker_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Standalone Real-Time Benchmark Executable
add_executable(oar_tracker_edge
    main.cpp
)

target_link_libraries(oar_tracker_edge oar_tracker_lib)
