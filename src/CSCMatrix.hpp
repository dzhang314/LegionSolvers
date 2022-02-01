#ifndef LEGION_SOLVERS_CSC_MATRIX_HPP
#define LEGION_SOLVERS_CSC_MATRIX_HPP

#include <cassert>

#include <legion.h>

#include "AbstractMatrix.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class CSCMatrix : public AbstractMatrix<ENTRY_T> {

        Legion::Context ctx;
        Legion::Runtime *rt;
        Legion::IndexSpace kernel_space;
        Legion::LogicalRegion kernel_region;
        Legion::FieldID fid_row;
        Legion::FieldID fid_entry;
        Legion::IndexSpace domain_space;
        Legion::LogicalRegion colptr_region;
        Legion::FieldID fid_colptr;

    public:

        explicit CSCMatrix(
            Legion::Context ctx,
            Legion::Runtime *rt,
            Legion::LogicalRegion kernel_region,
            Legion::FieldID fid_row,
            Legion::FieldID fid_entry,
            Legion::LogicalRegion colptr_region,
            Legion::FieldID fid_colptr
        ) : ctx(ctx), rt(rt),
            kernel_space(kernel_region.get_index_space()),
            kernel_region(kernel_region),
            fid_row(fid_row),
            fid_entry(fid_entry),
            domain_space(colptr_region.get_index_space()),
            colptr_region(colptr_region),
            fid_colptr(fid_colptr) {}

        virtual Legion::IndexSpace get_kernel_space() const override {
            return kernel_space;
        }

        virtual Legion::LogicalRegion get_kernel_region() const override {
            return kernel_region;
        }

        virtual std::vector<Legion::LogicalRegion>
        get_auxiliary_regions() const override {
            return {colptr_region};
        }

        virtual Legion::IndexPartition kernel_partition_from_domain_partition(
            Legion::IndexPartition domain_partition
        ) const override {
            return rt->create_partition_by_image_range(ctx,
                kernel_space,
                rt->get_logical_partition(colptr_region, domain_partition),
                colptr_region,
                fid_colptr,
                rt->get_index_partition_color_space_name(domain_partition)
            );
        }

        virtual Legion::IndexPartition kernel_partition_from_range_partition(
            Legion::IndexPartition range_partition
        ) const override {
            return rt->create_partition_by_preimage(ctx,
                range_partition,
                kernel_region,
                kernel_region,
                fid_row,
                rt->get_index_partition_color_space_name(range_partition)
            );
        }

        virtual Legion::IndexPartition domain_partition_from_kernel_partition(
            Legion::IndexSpace domain_space_,
            Legion::IndexPartition kernel_partition
        ) const override {
            assert(domain_space == domain_space_);
            return rt->create_partition_by_preimage_range(ctx,
                kernel_partition,
                colptr_region,
                colptr_region,
                fid_colptr,
                rt->get_index_partition_color_space_name(kernel_partition)
            );
        }

        virtual Legion::IndexPartition range_partition_from_kernel_partition(
            Legion::IndexSpace range_space,
            Legion::IndexPartition kernel_partition
        ) const override {
            return rt->create_partition_by_image(ctx,
                range_space,
                rt->get_logical_partition(kernel_region, kernel_partition),
                kernel_region,
                fid_row,
                rt->get_index_partition_color_space_name(kernel_partition)
            );
        }

    }; // class CSCMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_CSC_MATRIX_HPP
