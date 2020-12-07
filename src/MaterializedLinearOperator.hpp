#ifndef LEGION_SOLVERS_MATERIALIZED_LINEAR_OPERATOR_HPP
#define LEGION_SOLVERS_MATERIALIZED_LINEAR_OPERATOR_HPP

#include "LinearOperator.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    class MaterializedLinearOperator : public LinearOperator {


      public:
        virtual Legion::IndexPartitionT<KERNEL_DIM> kernel_partition_from_domain_partition(
            Legion::IndexPartitionT<DOMAIN_DIM> domain_partition, Legion::Context ctx, Legion::Runtime *rt) const = 0;


        virtual Legion::IndexPartitionT<KERNEL_DIM> kernel_partition_from_range_partition(
            Legion::IndexPartitionT<RANGE_DIM> range_partition, Legion::Context ctx, Legion::Runtime *rt) const = 0;


    }; // class MaterializedLinearOperator


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_MATERIALIZED_LINEAR_OPERATOR_HPP
