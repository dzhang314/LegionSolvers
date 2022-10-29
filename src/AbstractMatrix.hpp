#ifndef LEGION_SOLVERS_ABSTRACT_MATRIX_HPP_INCLUDED
#define LEGION_SOLVERS_ABSTRACT_MATRIX_HPP_INCLUDED

#include <vector> // for std::vector

#include <legion.h> // for Legion::*

#include "AbstractLinearOperator.hpp" // for AbstractLinearOperator
#include "PartitionedVector.hpp"      // for PartitionedVector

namespace LegionSolvers {


template <typename ENTRY_T>
class AbstractMatrix : public AbstractLinearOperator<ENTRY_T> {

public:

    virtual Legion::IndexSpace get_kernel_space() const = 0;

    virtual Legion::LogicalRegion get_kernel_region() const = 0;

    virtual std::vector<Legion::LogicalRegion>
    get_auxiliary_regions() const = 0;

    virtual Legion::IndexPartition kernel_partition_from_domain_partition(
        Legion::IndexPartition domain_partition
    ) const = 0;

    virtual Legion::IndexPartition
    kernel_partition_from_range_partition(Legion::IndexPartition range_partition
    ) const = 0;

    virtual Legion::IndexPartition domain_partition_from_kernel_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition kernel_partition
    ) const = 0;

    virtual Legion::IndexPartition range_partition_from_kernel_partition(
        Legion::IndexSpace range_space, Legion::IndexPartition kernel_partition
    ) const = 0;

    virtual void matvec(
        PartitionedVector<ENTRY_T> &dst_vector,
        const PartitionedVector<ENTRY_T> &src_vector,
        Legion::LogicalPartition kernel_partition,
        Legion::IndexPartition ghost_partition
    ) const = 0;

    virtual Legion::IndexPartition domain_partition_from_range_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition range_partition
    ) const override;

    virtual Legion::IndexPartition range_partition_from_domain_partition(
        Legion::IndexSpace range_space, Legion::IndexPartition domain_partition
    ) const override;

}; // class AbstractMatrix


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_ABSTRACT_MATRIX_HPP_INCLUDED
