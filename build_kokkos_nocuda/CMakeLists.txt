cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)
set(CMAKE_CXX_STANDARD 17)

find_package(Kokkos REQUIRED)
find_package(Legion REQUIRED)

add_executable(Test00Build
    ../src/LegionSolversMapper.cpp
    ../src/Test00Build.cpp
)

target_link_libraries(Test00Build Kokkos::kokkoscore Legion::Legion)

add_executable(Test01ScalarOperations
    ../src/LegionSolversMapper.cpp
    ../src/Test01ScalarOperations.cpp
)

target_link_libraries(Test01ScalarOperations Kokkos::kokkoscore Legion::Legion)

add_executable(Test02VectorOperations
    ../src/LegionSolversMapper.cpp
    ../src/Test02VectorOperations.cpp
)

target_link_libraries(Test02VectorOperations Kokkos::kokkoscore Legion::Legion)

# add_executable(Test01DenseVectorArithmetic
#     ../src/COOMatrixTasks.cpp
#     ../src/ExampleSystems.cpp
#     ../src/LegionSolversMapper.cpp
#     ../src/LegionUtilities.cpp
#     ../src/LinearAlgebraTasks.cpp
#     ../src/TaskRegistration.cpp
#     ../src/Test01DenseVectorArithmetic.cpp
#     ../src/UtilityTasks.cpp
# )
#
# target_link_libraries(Test01DenseVectorArithmetic Kokkos::kokkoscore Legion::Legion)
#
# add_executable(Test02DenseDistributedVectorArithmetic
#     ../src/COOMatrixTasks.cpp
#     ../src/ExampleSystems.cpp
#     ../src/LegionSolversMapper.cpp
#     ../src/LegionUtilities.cpp
#     ../src/LinearAlgebraTasks.cpp
#     ../src/TaskRegistration.cpp
#     ../src/Test02DenseDistributedVectorArithmetic.cpp
#     ../src/UtilityTasks.cpp
# )
#
# target_link_libraries(Test02DenseDistributedVectorArithmetic Kokkos::kokkoscore Legion::Legion)
#
# add_executable(Test03COO1DPartitioning
#     ../src/COOMatrixTasks.cpp
#     ../src/ExampleSystems.cpp
#     ../src/LegionSolversMapper.cpp
#     ../src/LegionUtilities.cpp
#     ../src/LinearAlgebraTasks.cpp
#     ../src/TaskRegistration.cpp
#     ../src/Test03COO1DPartitioning.cpp
#     ../src/UtilityTasks.cpp
# )
#
# target_link_libraries(Test03COO1DPartitioning Kokkos::kokkoscore Legion::Legion)
#
# add_executable(Test04CSR1DPartitioning
#     ../src/COOMatrixTasks.cpp
#     ../src/ExampleSystems.cpp
#     ../src/LegionSolversMapper.cpp
#     ../src/LegionUtilities.cpp
#     ../src/LinearAlgebraTasks.cpp
#     ../src/TaskRegistration.cpp
#     ../src/Test04CSR1DPartitioning.cpp
#     ../src/UtilityTasks.cpp
# )
#
# target_link_libraries(Test04CSR1DPartitioning Kokkos::kokkoscore Legion::Legion)
#
# add_executable(Test05COO1DSolveCGExact
#     ../src/COOMatrixTasks.cpp
#     ../src/ExampleSystems.cpp
#     ../src/LegionSolversMapper.cpp
#     ../src/LegionUtilities.cpp
#     ../src/LinearAlgebraTasks.cpp
#     ../src/TaskRegistration.cpp
#     ../src/Test05COO1DSolveCGExact.cpp
#     ../src/UtilityTasks.cpp
# )
#
# target_link_libraries(Test05COO1DSolveCGExact Kokkos::kokkoscore Legion::Legion)
#
# add_executable(Test06FillMatrix
#     ../src/COOMatrixTasks.cpp
#     ../src/ExampleSystems.cpp
#     ../src/LegionSolversMapper.cpp
#     ../src/LegionUtilities.cpp
#     ../src/LinearAlgebraTasks.cpp
#     ../src/TaskRegistration.cpp
#     ../src/Test06FillMatrix.cpp
#     ../src/UtilityTasks.cpp
# )
#
# target_link_libraries(Test06FillMatrix Kokkos::kokkoscore Legion::Legion)
