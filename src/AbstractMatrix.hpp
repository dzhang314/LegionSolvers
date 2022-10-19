#ifndef LEGION_SOLVERS_ABSTRACT_MATRIX_HPP_INCLUDED
#define LEGION_SOLVERS_ABSTRACT_MATRIX_HPP_INCLUDED

#include <vector>

#include <legion.h>

#include "AbstractLinearOperator.hpp"

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

    virtual Legion::IndexPartition domain_partition_from_range_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition range_partition
    ) const override {
        return domain_partition_from_kernel_partition(
            domain_space, kernel_partition_from_range_partition(range_partition)
        );
    }

    virtual Legion::IndexPartition range_partition_from_domain_partition(
        Legion::IndexSpace range_space, Legion::IndexPartition domain_partition
    ) const override {
        return range_partition_from_kernel_partition(
            range_space,
            kernel_partition_from_domain_partition(domain_partition)
        );
    }

}; // class AbstractMatrix


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_ABSTRACT_MATRIX_HPP_INCLUDED
