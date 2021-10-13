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
                  "to support specified value for LEGION_SOLVERS_MAX_DIM");


    #ifndef LEGION_SOLVERS_DEFAULT_VECTOR_FID
    constexpr Legion::FieldID
    LEGION_SOLVERS_DEFAULT_VECTOR_FID = 101;
    #endif


    using LEGION_SOLVERS_SUPPORTED_TYPES = TypeList<float, double>;


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LIBRARY_OPTIONS_HPP
