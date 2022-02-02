#ifndef LEGION_SOLVERS_ABSTRACT_LINEAR_OPERATOR_HPP
#define LEGION_SOLVERS_ABSTRACT_LINEAR_OPERATOR_HPP

#include <legion.h>


namespace LegionSolvers {


    template <typename ENTRY_T>
    class AbstractLinearOperator {

    public:

        virtual Legion::IndexPartition domain_partition_from_range_partition(
            Legion::IndexSpace domain_space,
            Legion::IndexPartition range_partition
        ) const = 0;

        virtual Legion::IndexPartition range_partition_from_domain_partition(
            Legion::IndexSpace range_space,
            Legion::IndexPartition domain_partition
        ) const = 0;

        // virtual void matvec_local(
        //     AbstractVector<ENTRY_T> &output_vector,
        //     const AbstractVector<ENTRY_T> &input_vector
        // ) const = 0;

        // virtual void rmatvec_local(
        //     AbstractVector<ENTRY_T> &output_vector,
        //     const AbstractVector<ENTRY_T> &input_vector
        // ) const = 0;

        // virtual void print() const = 0;

    }; // class AbstractLinearOperator


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_ABSTRACT_LINEAR_OPERATOR_HPP
