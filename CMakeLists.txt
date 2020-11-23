cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)

find_package(Legion REQUIRED)
find_package(Kokkos REQUIRED)
add_executable(LegionSolvers src/main.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-unused-but-set-variable -Wno-unused-parameter")

target_link_libraries(LegionSolvers Legion::Legion)
