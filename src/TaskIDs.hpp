#ifndef LEGION_SOLVERS_TASK_IDS_HPP
#define LEGION_SOLVERS_TASK_IDS_HPP

#include <legion.h>


namespace LegionSolvers {


    enum TaskBlockID : Legion::TaskID {
        ADDITION_TASK_BLOCK_ID,
        SUBTRACTION_TASK_BLOCK_ID,
        NEGATION_TASK_BLOCK_ID,
        MULTIPLICATION_TASK_BLOCK_ID,
        DIVISION_TASK_BLOCK_ID,
        DUMMY_TASK_BLOCK_ID,
        CONSTANT_FILL_TASK_BLOCK_ID,
        RANDOM_FILL_TASK_BLOCK_ID,
        COPY_TASK_BLOCK_ID,
        AXPY_TASK_BLOCK_ID,
        XPAY_TASK_BLOCK_ID,
        PRINT_VECTOR_TASK_BLOCK_ID,
        DOT_PRODUCT_TASK_BLOCK_ID,
        COO_MATVEC_TASK_BLOCK_ID,
        COO_RMATVEC_TASK_BLOCK_ID,
        COO_PRINT_TASK_BLOCK_ID,
        FILL_COO_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID,
        FILL_COO_NEGATIVE_LAPLACIAN_2D_TASK_BLOCK_ID,
    };


    enum ProjectionFunctorID : Legion::ProjectionID {
        PFID_IJ_TO_I = 10'000,
        PFID_IJ_TO_J,
        PFID_IJ_TO_IJ,
        PFID_IJ_TO_JI,
    };

    template <typename T> constexpr Legion::ReductionOpID LEGION_REDOP_SUM = -1;
    template <> constexpr Legion::ReductionOpID LEGION_REDOP_SUM<float > = LEGION_REDOP_SUM_FLOAT32;
    template <> constexpr Legion::ReductionOpID LEGION_REDOP_SUM<double> = LEGION_REDOP_SUM_FLOAT64;


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_TASK_IDS_HPP
