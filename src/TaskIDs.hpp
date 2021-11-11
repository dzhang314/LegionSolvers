#ifndef LEGION_SOLVERS_TASK_IDS_HPP
#define LEGION_SOLVERS_TASK_IDS_HPP

#include <legion.h>

#include "LibraryOptions.hpp"


namespace LegionSolvers {


    enum TaskBlockID : Legion::TaskID {
        ADDITION_TASK_BLOCK_ID,
        SUBTRACTION_TASK_BLOCK_ID,
        NEGATION_TASK_BLOCK_ID,
        MULTIPLICATION_TASK_BLOCK_ID,
        DIVISION_TASK_BLOCK_ID,
        DUMMY_TASK_BLOCK_ID,
        RANDOM_FILL_TASK_BLOCK_ID,
        PRINT_VECTOR_TASK_BLOCK_ID,
        SCAL_TASK_BLOCK_ID,
        AXPY_TASK_BLOCK_ID,
        XPAY_TASK_BLOCK_ID,
        DOT_TASK_BLOCK_ID,
        COO_MATVEC_TASK_BLOCK_ID,
        COO_RMATVEC_TASK_BLOCK_ID,
        COO_PRINT_TASK_BLOCK_ID,
        FILL_COO_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID,
        FILL_COO_NEGATIVE_LAPLACIAN_2D_TASK_BLOCK_ID,
    };


    enum ProjectionFunctorID : Legion::ProjectionID {
        PFID_KDR_TO_K = LEGION_SOLVERS_PROJECTION_ID_ORIGIN,
        PFID_KDR_TO_D,
        PFID_KDR_TO_R,
        PFID_KDR_TO_DR,
    };


    template <typename T> constexpr Legion::ReductionOpID LEGION_REDOP_SUM = -1;
    template <> constexpr Legion::ReductionOpID LEGION_REDOP_SUM<float > = LEGION_REDOP_SUM_FLOAT32;
    template <> constexpr Legion::ReductionOpID LEGION_REDOP_SUM<double> = LEGION_REDOP_SUM_FLOAT64;


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_IDS_HPP
