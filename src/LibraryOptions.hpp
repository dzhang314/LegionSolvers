#ifndef LEGION_SOLVERS_LIBRARY_OPTIONS_HPP_INCLUDED
#define LEGION_SOLVERS_LIBRARY_OPTIONS_HPP_INCLUDED

#include <legion.h> // for Legion::*

namespace LegionSolvers {


// TODO: provide ways to turn these off
#define LEGION_SOLVERS_USE_FLOAT true
#define LEGION_SOLVERS_USE_DOUBLE true
#define LEGION_SOLVERS_USE_S32_INDICES true
#define LEGION_SOLVERS_USE_U32_INDICES true
#define LEGION_SOLVERS_USE_S64_INDICES true


#ifdef NDEBUG
constexpr bool LEGION_SOLVERS_CHECK_BOUNDS = false;
#else
constexpr bool LEGION_SOLVERS_CHECK_BOUNDS = true;
#endif // NDEBUG


#ifndef LEGION_SOLVERS_TASK_ID_ORIGIN
constexpr Legion::TaskID LEGION_SOLVERS_TASK_ID_ORIGIN = 500'000;
#endif // LEGION_SOLVERS_TASK_ID_ORIGIN


#ifndef LEGION_SOLVERS_MAPPER_ID
constexpr Legion::MapperID LEGION_SOLVERS_MAPPER_ID = 1'000;
#endif // LEGION_SOLVERS_MAPPER_ID


#ifndef LEGION_SOLVERS_PROJECTION_ID_ORIGIN
constexpr Legion::ProjectionID LEGION_SOLVERS_PROJECTION_ID_ORIGIN = 10'000;
#endif // LEGION_SOLVERS_PROJECTION_ID_ORIGIN


#ifndef LEGION_SOLVERS_MAX_DIM
    #define LEGION_SOLVERS_MAX_DIM 3
#endif // LEGION_SOLVERS_MAX_DIM


static_assert(
    LEGION_MAX_DIM >= 3,
    "LegionSolvers requires Legion to be compiled with the "
    "preprocessor macro LEGION_MAX_DIM set to 3 or higher."
);


static_assert(
    LEGION_SOLVERS_MAX_DIM <= LEGION_MAX_DIM,
    "Legion was not compiled with LEGION_MAX_DIM large enough to "
    "support the specified value for LEGION_SOLVERS_MAX_DIM."
);


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LIBRARY_OPTIONS_HPP_INCLUDED
