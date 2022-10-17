#ifndef LEGION_SOLVERS_TASK_IDS_HPP_INCLUDED
#define LEGION_SOLVERS_TASK_IDS_HPP_INCLUDED

#include <legion.h>

namespace LegionSolvers {


enum TaskBlockID : Legion::TaskID {
    PRINT_SCALAR_TASK_BLOCK_ID,
}; // enum TaskBlockID


// enum ProjectionFunctorID : Legion::ProjectionID {
//     PFID_KDR_TO_K = LEGION_SOLVERS_PROJECTION_ID_ORIGIN,
//     PFID_KDR_TO_D,
//     PFID_KDR_TO_R,
//     PFID_KDR_TO_DR,
// }; // enum ProjectionFunctorID


template <typename T> constexpr Legion::ReductionOpID LEGION_REDOP_SUM = -1;
template <> constexpr Legion::ReductionOpID LEGION_REDOP_SUM<float> =
    LEGION_REDOP_SUM_FLOAT32;
template <> constexpr Legion::ReductionOpID LEGION_REDOP_SUM<double> =
    LEGION_REDOP_SUM_FLOAT64;


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_TASK_IDS_HPP_INCLUDED
