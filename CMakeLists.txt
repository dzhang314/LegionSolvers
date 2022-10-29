cmake_minimum_required(VERSION 3.13)
project(LegionSolvers)

find_package(Legion REQUIRED)

# If we can use CUDA, find some more packages and enable CUDA.
if(Legion_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
endif()

set(LegionSolvers_SOURCES "")

list(APPEND LegionSolvers_SOURCES
    src/AbstractMatrix.cpp
    src/COOMatrixTasks.cpp
    src/CSRMatrixTasks.cpp
    src/ExampleSystems.cpp
    src/Initialize.cpp
    src/LegionSolversMapper.cpp
    src/LegionUtilities.cpp
    src/LinearAlgebraTasks.cpp
    src/Scalar.cpp
    src/UtilityTasks.cpp
)

if(Legion_USE_OPENMP AND NOT Legion_USE_KOKKOS)
    # Placeholder if we ever add OpenMP tasks.
endif()

if(Legion_USE_CUDA AND NOT Legion_USE_KOKKOS)
    list(APPEND LegionSolvers_SOURCES
        src/COOMatrixTasks.cu
        src/CSRMatrixTasks.cu
        src/CudaLibs.cu
        src/LinearAlgebraTasks.cu
    )
endif()

# Register the LegionSolvers library.
add_library(LegionSolvers ${LegionSolvers_SOURCES})
# Link in Legion.
target_link_libraries(LegionSolvers Legion::Legion)
# If Kokkos was enabled in Legion, link in Kokkos.
if(Legion_USE_KOKKOS)
    target_link_libraries(Kokkos::kokkoscore)
endif()
# If Legion has CUDA support, link in the appopriate CUDA libraries.
if(Legion_USE_CUDA)
    if(NOT Legion_USE_KOKKOS)
        target_link_libraries(LegionSolvers
            CUDA::cudart CUDA::cublas CUDA::cusparse
        )
    endif()
endif()

# TODO (rohany): Add a debug configuration and debug defines for LegionSolvers.
set_target_properties(LegionSolvers
    PROPERTIES CXX_STANDARD             17
    CXX_STANDARD_REQUIRED               ON
    POSITION_INDEPENDENT_CODE           ON
    INTERFACE_POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD                       17
    CUDA_STANDARD_REQUIRED              ON
    LIBRARY_OUTPUT_DIRECTORY            lib
)

# TODO (rohany): Export all of the hpp files to a location to include. This
# needs to be done once LegionSolvers is a library that other people are
# expected to import, as right now it's just us writing applications in it.

add_executable(Test00Build src/Test00Build.cpp)
target_link_libraries(Test00Build LegionSolvers)
set_target_properties(Test00Build PROPERTIES CXX_STANDARD 17)

add_executable(Test01ScalarOperations src/Test01ScalarOperations.cpp)
target_link_libraries(Test01ScalarOperations LegionSolvers)
set_target_properties(Test01ScalarOperations PROPERTIES CXX_STANDARD 17)

add_executable(Test02VectorOperations src/Test02VectorOperations.cpp)
target_link_libraries(Test02VectorOperations LegionSolvers)
set_target_properties(Test02VectorOperations PROPERTIES CXX_STANDARD 17)

add_executable(Test03COOPartitioning src/Test03COOPartitioning.cpp)
target_link_libraries(Test03COOPartitioning LegionSolvers)
set_target_properties(Test03COOPartitioning PROPERTIES CXX_STANDARD 17)

add_executable(Test04CSRPartitioning src/Test04CSRPartitioning.cpp)
target_link_libraries(Test04CSRPartitioning LegionSolvers)
set_target_properties(Test04CSRPartitioning PROPERTIES CXX_STANDARD 17)

add_executable(Test05COOSolveCG src/Test05COOSolveCG.cpp)
target_link_libraries(Test05COOSolveCG LegionSolvers)
set_target_properties(Test05COOSolveCG PROPERTIES CXX_STANDARD 17)

add_executable(Test06CSRSolveCG src/Test06CSRSolveCG.cpp)
target_link_libraries(Test06CSRSolveCG LegionSolvers)
set_target_properties(Test06CSRSolveCG PROPERTIES CXX_STANDARD 17)
