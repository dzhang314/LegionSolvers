#ifndef LEGION_SOLVERS_LIBRARY_OPTIONS_HPP
#define LEGION_SOLVERS_LIBRARY_OPTIONS_HPP

#include <legion.h>

#include "MetaprogrammingUtilities.hpp"


namespace LegionSolvers {


    #ifndef LEGION_SOLVERS_MAPPER_ID
    constexpr Legion::MapperID
    LEGION_SOLVERS_MAPPER_ID = 1'000;
    #endif


    #ifndef LEGION_SOLVERS_TASK_ID_ORIGIN
    constexpr Legion::TaskID
    LEGION_SOLVERS_TASK_ID_ORIGIN = 1'000'000;
    #endif


    #ifndef LEGION_SOLVERS_PROJECTION_ID_ORIGIN
    constexpr Legion::ProjectionID
    LEGION_SOLVERS_PROJECTION_ID_ORIGIN = 10'000;
    #endif


    #ifndef LEGION_SOLVERS_MAX_DIM
    constexpr int
    LEGION_SOLVERS_MAX_DIM = 3;
    #endif


    static_assert(LEGION_SOLVERS_MAX_DIM <= LEGION_MAX_DIM,
                  "Legion was not compiled with LEGION_MAX_DIM large enough "
                  "to support the specified value for LEGION_SOLVERS_MAX_DIM");


    #ifndef LEGION_SOLVERS_DEFAULT_VECTOR_FID
    constexpr Legion::FieldID
    LEGION_SOLVERS_DEFAULT_VECTOR_FID = 101;
    #endif


    #ifndef LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_I
    constexpr Legion::FieldID
    LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_I = 102;
    #endif


    #ifndef LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_J
    constexpr Legion::FieldID
    LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_J = 103;
    #endif


    #ifndef LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_ENTRY
    constexpr Legion::FieldID
    LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_ENTRY = 104;
    #endif


    #ifndef LEGION_SOLVERS_DEFAULT_TILE_PARTITION_COLOR
    constexpr Legion::Color
    LEGION_SOLVERS_DEFAULT_TILE_PARTITION_COLOR = 777;
    #endif


    using LEGION_SOLVERS_SUPPORTED_TYPES = TypeList<float, double>;


    // TODO: preprocessor flag for bounds checking
    constexpr bool LEGION_SOLVERS_CHECK_BOUNDS = false;


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LIBRARY_OPTIONS_HPP
