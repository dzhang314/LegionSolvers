#ifndef LEGION_SOLVERS_COO_MATRIX_HPP_INCLUDED
#define LEGION_SOLVERS_COO_MATRIX_HPP_INCLUDED

#include <string> // for std::string

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

    COOMatrix() = delete;

    explicit COOMatrix(
        Legion::Context ctx,
        Legion::Runtime *rt,
        Legion::LogicalRegion kernel_region,
        Legion::FieldID fid_entry,
        Legion::FieldID fid_row,
        Legion::FieldID fid_col
    );

    COOMatrix(const COOMatrix &);

    COOMatrix(COOMatrix &&) = delete;

    ~COOMatrix();

    COOMatrix &operator=(const COOMatrix &) = delete;

    COOMatrix &operator=(COOMatrix &&) = delete;

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

    virtual Legion::IndexPartition
    create_kernel_partition_from_domain_partition(
        Legion::IndexPartition domain_partition
    ) const override;

    virtual Legion::IndexPartition create_kernel_partition_from_range_partition(
        Legion::IndexPartition range_partition
    ) const override;

    virtual Legion::IndexPartition
    create_domain_partition_from_kernel_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition kernel_partition
    ) const override;

    virtual Legion::IndexPartition create_range_partition_from_kernel_partition(
        Legion::IndexSpace range_space, Legion::IndexPartition kernel_partition
    ) const override;

    virtual void matvec(
        PartitionedVector<ENTRY_T> &dst_vector,
        const PartitionedVector<ENTRY_T> &src_vector,
        Legion::LogicalPartition kernel_partition,
        Legion::IndexPartition ghost_partition
    ) const override;

    virtual void print(Legion::IndexSpace index_space) const;

}; // class COOMatrix


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_COO_MATRIX_HPP_INCLUDED
