cmake_minimum_required(VERSION 3.14)
project(NN_Lab3_4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# Python модуль
add_library(nn_core MODULE bindings.cpp)
target_link_libraries(nn_core PRIVATE pybind11::module)
target_include_directories(nn_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Тестовая программа
add_executable(test_nn main.cpp)
target_include_directories(test_nn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Проект настроен!")
message(STATUS "Собирай командой: cmake --build . --config Release")