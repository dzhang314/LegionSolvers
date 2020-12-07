#ifndef LEGION_SOLVERS_SPARSE_MATRIX_HPP
#define LEGION_SOLVERS_SPARSE_MATRIX_HPP

#include "MaterializedLinearOperator.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    class SparseMatrix : public MaterializedLinearOperator<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {


        // TODO


    }; // class SparseMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SPARSE_MATRIX_HPP
