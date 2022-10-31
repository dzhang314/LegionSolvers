#ifndef LEGION_SOLVERS_COO_MATRIX_HPP_INCLUDED
#define LEGION_SOLVERS_COO_MATRIX_HPP_INCLUDED

#include <legion.h> // for Legion::*

#include "AbstractMatrix.hpp"    // for AbstractMatrix
#include "PartitionedVector.hpp" // for PartitionedVector

namespace LegionSolvers {


template <typename ENTRY_T>
class COOMatrix : public AbstractMatrix<ENTRY_T> {

    const Legion::Context ctx;
    Legion::Runtime *const rt;

    const Legion::LogicalRegion kernel_region;
    const Legion::FieldID fid_entry;
    const Legion::FieldID fid_row;
    const Legion::FieldID fid_col;

public:

    explicit COOMatrix(
        Legion::Context ctx,
        Legion::Runtime *rt,
        Legion::LogicalRegion kernel_region,
        Legion::FieldID fid_entry,
        Legion::FieldID fid_row,
        Legion::FieldID fid_col
    )
        : ctx(ctx)
        , rt(rt)
        , kernel_region(kernel_region)
        , fid_entry(fid_entry)
        , fid_row(fid_row)
        , fid_col(fid_col) {}

    virtual Legion::IndexSpace get_kernel_space() const override {
        return kernel_region.get_index_space();
    }

    virtual Legion::LogicalRegion get_kernel_region() const override {
        return kernel_region;
    }

    virtual std::vector<Legion::LogicalRegion>
    get_auxiliary_regions() const override {
        return {};
    }

    virtual Legion::IndexPartition kernel_partition_from_domain_partition(
        Legion::IndexPartition domain_partition
    ) const override;

    virtual Legion::IndexPartition
    kernel_partition_from_range_partition(Legion::IndexPartition range_partition
    ) const override;

    virtual Legion::IndexPartition domain_partition_from_kernel_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition kernel_partition
    ) const override;

    virtual Legion::IndexPartition range_partition_from_kernel_partition(
        Legion::IndexSpace range_space, Legion::IndexPartition kernel_partition
    ) const override;

    virtual void matvec(
        PartitionedVector<ENTRY_T> &dst_vector,
        const PartitionedVector<ENTRY_T> &src_vector,
        Legion::LogicalPartition kernel_partition,
        Legion::IndexPartition ghost_partition
    ) const override;

}; // class COOMatrix


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_COO_MATRIX_HPP_INCLUDED
