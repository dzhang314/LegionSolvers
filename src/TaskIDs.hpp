#ifndef LEGION_SOLVERS_TASK_IDS_HPP_INCLUDED
#define LEGION_SOLVERS_TASK_IDS_HPP_INCLUDED

#include <legion.h> // for Legion::*

namespace LegionSolvers {


enum TaskBlockID : Legion::TaskID {
    PRINT_SCALAR_TASK_BLOCK_ID,
    NEGATE_SCALAR_TASK_BLOCK_ID,
    ADD_SCALAR_TASK_BLOCK_ID,
    SUBTRACT_SCALAR_TASK_BLOCK_ID,
    MULTIPLY_SCALAR_TASK_BLOCK_ID,
    DIVIDE_SCALAR_TASK_BLOCK_ID,
    SCAL_TASK_BLOCK_ID,
    AXPY_TASK_BLOCK_ID,
    XPAY_TASK_BLOCK_ID,
    DOT_TASK_BLOCK_ID,
    COO_MATVEC_TASK_BLOCK_ID,
    COO_RMATVEC_TASK_BLOCK_ID,
    CSR_MATVEC_TASK_BLOCK_ID,
    CSR_RMATVEC_TASK_BLOCK_ID,
}; // enum TaskBlockID


template <typename T>
constexpr Legion::ReductionOpID LEGION_REDOP_SUM = -1;
template <>
constexpr Legion::ReductionOpID LEGION_REDOP_SUM<float> =
    LEGION_REDOP_SUM_FLOAT32;
template <>
constexpr Legion::ReductionOpID LEGION_REDOP_SUM<double> =
    LEGION_REDOP_SUM_FLOAT64;


// enum ProjectionFunctorID : Legion::ProjectionID {
//     PFID_KDR_TO_K = LEGION_SOLVERS_PROJECTION_ID_ORIGIN,
//     PFID_KDR_TO_D,
//     PFID_KDR_TO_R,
//     PFID_KDR_TO_DR,
// }; // enum ProjectionFunctorID


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_TASK_IDS_HPP_INCLUDED
