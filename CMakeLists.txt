cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-unused-but-set-variable -Wno-unused-parameter -Wfatal-errors -g ${CMAKE_CXX_FLAGS}")

find_package(Legion REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(Test01VectorArithmetic
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/Test01VectorArithmetic.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(Test01VectorArithmetic Kokkos::kokkoscore Legion::Legion)

add_executable(Test02CGLaplacian1D
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/Test02CGLaplacian1D.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(Test02CGLaplacian1D Kokkos::kokkoscore Legion::Legion)

add_executable(TestExecutable
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/TestMain.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(TestExecutable Kokkos::kokkoscore Legion::Legion)
