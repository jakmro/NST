cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nst)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

message(STATUS "OpenCV found at: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Torch found at: ${TORCH_INCLUDE_DIRS}")