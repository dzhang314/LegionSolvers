cmake_minimum_required(VERSION 2.7)
project(TrilinosSolverBenchmark)

SET(Trilinos_PREFIX /p/gpfs1/zhang70/lib/trilinos)
SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
FIND_PACKAGE(Trilinos REQUIRED)

# Make sure to use same compilers and flags as Trilinos
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

INCLUDE_DIRECTORIES(./ ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
LINK_DIRECTORIES(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

add_executable(TrilinosSolverBenchmark main.cpp)
target_link_libraries(TrilinosSolverBenchmark ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
