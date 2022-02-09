cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-unused-but-set-variable -Wno-unused-parameter -Wfatal-errors -g ${CMAKE_CXX_FLAGS}")

find_package(Legion REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(Test01DenseVectorArithmetic
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LegionUtilities.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/Test01DenseVectorArithmetic.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(Test01DenseVectorArithmetic Kokkos::kokkoscore Legion::Legion)

add_executable(Test02DenseDistributedVectorArithmetic
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LegionUtilities.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/Test02DenseDistributedVectorArithmetic.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(Test02DenseDistributedVectorArithmetic Kokkos::kokkoscore Legion::Legion)

add_executable(Test03COO1DPartitioning
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LegionUtilities.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/Test03COO1DPartitioning.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(Test03COO1DPartitioning Kokkos::kokkoscore Legion::Legion)

add_executable(Test04CSR1DPartitioning
    src/COOMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/LegionSolversMapper.cpp
    src/LegionUtilities.cpp
    src/LinearAlgebraTasks.cpp
    src/TaskRegistration.cpp
    src/Test04CSR1DPartitioning.cpp
    src/UtilityTasks.cpp
)

target_link_libraries(Test04CSR1DPartitioning Kokkos::kokkoscore Legion::Legion)

# add_executable(Test05COO1DSolveCGExact
#     src/COOMatrixTasks.cpp
#     src/ExampleSystems.cpp
#     src/LegionSolversMapper.cpp
#     src/LegionUtilities.cpp
#     src/LinearAlgebraTasks.cpp
#     src/TaskRegistration.cpp
#     src/Test05COO1DSolveCGExact.cpp
#     src/UtilityTasks.cpp
# )

# target_link_libraries(Test05COO1DSolveCGExact Kokkos::kokkoscore Legion::Legion)
