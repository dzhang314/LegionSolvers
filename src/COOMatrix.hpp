#ifndef LEGION_SOLVERS_COO_MATRIX_HPP
#define LEGION_SOLVERS_COO_MATRIX_HPP

#include <legion.h>

#include "AbstractMatrix.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class COOMatrix : public AbstractMatrix<ENTRY_T> {


        Legion::Context ctx;
        Legion::Runtime *rt;
        Legion::IndexSpace kernel_space;
        Legion::LogicalRegion kernel_region;
        Legion::FieldID fid_i;
        Legion::FieldID fid_j;
        Legion::FieldID fid_entry;


    public:


        explicit COOMatrix(
            Legion::Context ctx,
            Legion::Runtime *rt,
            Legion::LogicalRegion kernel_region,
            const Legion::FieldID fid_i,
            const Legion::FieldID fid_j,
            const Legion::FieldID fid_entry
        ) : ctx(ctx), rt(rt),
            kernel_space(kernel_region.get_index_space()),
            kernel_region(kernel_region),
            fid_i(fid_i),
            fid_j(fid_j),
            fid_entry(fid_entry) {}


        virtual Legion::IndexSpace get_kernel_space() const override {
            return kernel_space;
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
        ) const override {
            return rt->create_partition_by_preimage(ctx,
                domain_partition,
                kernel_region,
                kernel_region,
                fid_j,
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
                fid_i,
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
                fid_j,
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
                fid_i,
                rt->get_index_partition_color_space_name(kernel_partition)
            );
        }


    }; // class COOMatrixT


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_HPP
