cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(Legion_DIR /home/users/dkzhang/lib/legion/share/Legion/cmake)
set(GASNet_INCLUDE_DIR /home/users/dkzhang/lib/gasnet/include)
set(Kokkos_DIR /home/users/dkzhang/lib/kokkos/lib64/cmake/Kokkos)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-unused-but-set-variable -Wno-unused-parameter -Wfatal-errors")

find_package(Legion REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(LegionSolversTestCOO1D src/TestCOO1D.cpp)
add_executable(LegionSolversTestCOO2D src/TestCOO2D.cpp)
add_executable(LegionSolversTestCSR1D src/TestCSR1D.cpp)

target_link_libraries(LegionSolversTestCOO1D Legion::Legion)
target_link_libraries(LegionSolversTestCOO2D Legion::Legion)
target_link_libraries(LegionSolversTestCSR1D Legion::Legion)
