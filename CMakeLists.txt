cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-unused-but-set-variable -Wno-unused-parameter -Wfatal-errors ${CMAKE_CXX_FLAGS}")

find_package(Legion REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(TestExecutable
    src/TestMain.cpp
    src/COOMatrixTasks.cpp
    src/LegionUtilities.cpp
    src/LinearAlgebraTasks.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(TestExecutable Kokkos::kokkoscore Legion::Legion)

# add_executable(LegionSolversTestCOO1D src/TestCOO1D.cpp)
# add_executable(LegionSolversTestCOO2D src/TestCOO2D.cpp)
# add_executable(LegionSolversTestCSR1D src/TestCSR1D.cpp)

# target_link_libraries(LegionSolversTestCOO1D Kokkos::kokkoscore Legion::Legion)
# target_link_libraries(LegionSolversTestCOO2D Kokkos::kokkoscore Legion::Legion)
# target_link_libraries(LegionSolversTestCSR1D Kokkos::kokkoscore Legion::Legion)
