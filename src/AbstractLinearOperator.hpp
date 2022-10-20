#ifndef LEGION_SOLVERS_ABSTRACT_LINEAR_OPERATOR_HPP_INCLUDED
#define LEGION_SOLVERS_ABSTRACT_LINEAR_OPERATOR_HPP_INCLUDED

#include <legion.h> // for Legion::IndexSpace, Legion::IndexPartition

namespace LegionSolvers {


template <typename ENTRY_T>
class AbstractLinearOperator {

  public:

    virtual Legion::IndexPartition domain_partition_from_range_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition range_partition
    ) const = 0;

    virtual Legion::IndexPartition range_partition_from_domain_partition(
        Legion::IndexSpace range_space, Legion::IndexPartition domain_partition
    ) const = 0;

}; // class AbstractLinearOperator


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_ABSTRACT_LINEAR_OPERATOR_HPP_INCLUDED
