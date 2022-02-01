#ifndef LEGION_SOLVERS_CSR_MATRIX_HPP
#define LEGION_SOLVERS_CSR_MATRIX_HPP

#include <cassert>

#include <legion.h>

#include "AbstractMatrix.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class CSRMatrix : public AbstractMatrix<ENTRY_T> {

        Legion::Context ctx;
        Legion::Runtime *rt;
        Legion::IndexSpace kernel_space;
        Legion::LogicalRegion kernel_region;
        Legion::FieldID fid_col;
        Legion::FieldID fid_entry;
        Legion::IndexSpace range_space;
        Legion::LogicalRegion rowptr_region;
        Legion::FieldID fid_rowptr;

    public:

        explicit CSRMatrix(
            Legion::Context ctx,
            Legion::Runtime *rt,
            Legion::LogicalRegion kernel_region,
            Legion::FieldID fid_col,
            Legion::FieldID fid_entry,
            Legion::LogicalRegion rowptr_region,
            Legion::FieldID fid_rowptr
        ) : ctx(ctx), rt(rt),
            kernel_space(kernel_region.get_index_space()),
            kernel_region(kernel_region),
            fid_col(fid_col),
            fid_entry(fid_entry),
            range_space(rowptr_region.get_index_space()),
            rowptr_region(rowptr_region),
            fid_rowptr(fid_rowptr) {}

        virtual Legion::IndexSpace get_kernel_space() const override {
            return kernel_space;
        }

        virtual Legion::LogicalRegion get_kernel_region() const override {
            return kernel_region;
        }

        virtual std::vector<Legion::LogicalRegion>
        get_auxiliary_regions() const override {
            return {rowptr_region};
        }

        virtual Legion::IndexPartition kernel_partition_from_domain_partition(
            Legion::IndexPartition domain_partition
        ) const override {
            return rt->create_partition_by_preimage(ctx,
                domain_partition,
                kernel_region,
                kernel_region,
                fid_col,
                rt->get_index_partition_color_space_name(domain_partition)
            );
        }

        virtual Legion::IndexPartition kernel_partition_from_range_partition(
            Legion::IndexPartition range_partition
        ) const override {
            return rt->create_partition_by_image_range(ctx,
                kernel_space,
                rt->get_logical_partition(rowptr_region, range_partition),
                rowptr_region,
                fid_rowptr,
                rt->get_index_partition_color_space_name(range_partition)
            );
        }

        virtual Legion::IndexPartition domain_partition_from_kernel_partition(
            Legion::IndexSpace domain_space,
            Legion::IndexPartition kernel_partition
        ) const override {
            return rt->create_partition_by_image(ctx,
                domain_space,
                rt->get_logical_partition(kernel_region, kernel_partition),
                kernel_region,
                fid_col,
                rt->get_index_partition_color_space_name(kernel_partition)
            );
        }

        virtual Legion::IndexPartition range_partition_from_kernel_partition(
            Legion::IndexSpace range_space_,
            Legion::IndexPartition kernel_partition
        ) const override {
            assert(range_space == range_space_);
            return rt->create_partition_by_preimage_range(ctx,
                kernel_partition,
                rowptr_region,
                rowptr_region,
                fid_rowptr,
                rt->get_index_partition_color_space_name(kernel_partition)
            );
        }

    }; // class CSRMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_CSR_MATRIX_HPP
