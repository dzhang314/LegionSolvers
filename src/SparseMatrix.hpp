#ifndef LEGION_SOLVERS_SPARSE_MATRIX_HPP
#define LEGION_SOLVERS_SPARSE_MATRIX_HPP

#include "MaterializedLinearOperator.hpp"


namespace LegionSolvers {


    template <int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM, typename ENTRY_T>
    class SparseMatrix
        : public MaterializedLinearOperator<KERNEL_DIM, DOMAIN_DIM, RANGE_DIM,
                                            ENTRY_T> {


        // TODO


    }; // class SparseMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SPARSE_MATRIX_HPP
