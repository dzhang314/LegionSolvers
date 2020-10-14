cmake_minimum_required(VERSION 3.16)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER "/opt/kokkos/bin/nvcc_wrapper")

find_package(Legion REQUIRED)
find_package(Kokkos REQUIRED)
add_executable(LegionSolvers src/main.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -march=native -mtune=native")

target_link_libraries(LegionSolvers Kokkos::kokkoscore)
target_link_libraries(LegionSolvers Legion::Legion)
