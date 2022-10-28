#ifndef LEGION_SOLVERS_CSR_MATRIX_HPP_INCLUDED
#define LEGION_SOLVERS_CSR_MATRIX_HPP_INCLUDED

#include <legion.h> // for Legion::*

#include "AbstractMatrix.hpp"    // for AbstractMatrix
#include "CSRMatrixTasks.hpp"    // for CSRMatvecTask, CSRRmatvecTask
#include "LibraryOptions.hpp"    // for LEGION_SOLVERS_MAPPER_ID
#include "PartitionedVector.hpp" // for PartitionedVector

namespace LegionSolvers {


template <typename ENTRY_T>
class CSRMatrix : public AbstractMatrix<ENTRY_T> {

    const Legion::Context ctx;
    Legion::Runtime *const rt;
    const Legion::IndexSpace kernel_space;
    const Legion::LogicalRegion kernel_region;
    const Legion::FieldID fid_col;
    const Legion::FieldID fid_entry;
    const Legion::IndexSpace range_space;
    const Legion::LogicalRegion rowptr_region;
    const Legion::FieldID fid_rowptr;

public:

    explicit CSRMatrix(
        Legion::Context ctx,
        Legion::Runtime *rt,
        Legion::LogicalRegion kernel_region,
        Legion::FieldID fid_col,
        Legion::FieldID fid_entry,
        Legion::LogicalRegion rowptr_region,
        Legion::FieldID fid_rowptr
    )
        : ctx(ctx), rt(rt), kernel_space(kernel_region.get_index_space()),
          kernel_region(kernel_region), fid_col(fid_col), fid_entry(fid_entry),
          range_space(rowptr_region.get_index_space()),
          rowptr_region(rowptr_region), fid_rowptr(fid_rowptr) {}

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
        return rt->create_partition_by_preimage(
            ctx,
            domain_partition,
            kernel_region,
            kernel_region,
            fid_col,
            rt->get_index_partition_color_space_name(domain_partition)
        );
    }

    virtual Legion::IndexPartition
    kernel_partition_from_range_partition(Legion::IndexPartition range_partition
    ) const override {
        return rt->create_partition_by_image_range(
            ctx,
            kernel_space,
            rt->get_logical_partition(rowptr_region, range_partition),
            rowptr_region,
            fid_rowptr,
            rt->get_index_partition_color_space_name(range_partition)
        );
    }

    virtual Legion::IndexPartition domain_partition_from_kernel_partition(
        Legion::IndexSpace domain_space, Legion::IndexPartition kernel_partition
    ) const override {
        return rt->create_partition_by_image(
            ctx,
            domain_space,
            rt->get_logical_partition(kernel_region, kernel_partition),
            kernel_region,
            fid_col,
            rt->get_index_partition_color_space_name(kernel_partition)
        );
    }

    virtual Legion::IndexPartition range_partition_from_kernel_partition(
        Legion::IndexSpace range_space_, Legion::IndexPartition kernel_partition
    ) const override {
        assert(range_space == range_space_);
        return rt->create_partition_by_preimage_range(
            ctx,
            kernel_partition,
            rowptr_region,
            rowptr_region,
            fid_rowptr,
            rt->get_index_partition_color_space_name(kernel_partition)
        );
    }

    virtual void matvec(
        PartitionedVector<ENTRY_T> &dst_vector,
        const PartitionedVector<ENTRY_T> &src_vector,
        Legion::LogicalPartition kernel_partition,
        Legion::IndexPartition ghost_partition
    ) const override {
        const Legion::FieldID fids[3] = {fid_entry, fid_col};

        Legion::IndexLauncher launcher(
            LegionSolvers::CSRMatvecTask<ENTRY_T, 0, 0, 0, void, void, void>::
                task_id(
                    kernel_space,
                    src_vector.get_index_space(),
                    dst_vector.get_index_space()
                ),
            dst_vector.get_color_space(),
            Legion::TaskArgument(&fids, sizeof(Legion::FieldID[3])),
            Legion::ArgumentMap()
        );
        launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;

        launcher.add_region_requirement(Legion::RegionRequirement(
            dst_vector.get_logical_partition(),
            0,
            LEGION_READ_WRITE,
            LEGION_EXCLUSIVE,
            dst_vector.get_logical_region()
        ));
        launcher.add_field(0, dst_vector.get_fid());

        launcher.add_region_requirement(Legion::RegionRequirement(
            kernel_partition,
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            kernel_region
        ));
        launcher.add_field(1, fid_entry);
        launcher.add_field(1, fid_col);

        launcher.add_region_requirement(Legion::RegionRequirement(
            rt->get_logical_partition(
                rowptr_region, dst_vector.get_index_partition()
            ),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            rowptr_region
        ));
        launcher.add_field(2, fid_rowptr);

        launcher.add_region_requirement(Legion::RegionRequirement(
            rt->get_logical_partition(
                src_vector.get_logical_region(), ghost_partition
            ),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            src_vector.get_logical_region()
        ));
        launcher.add_field(3, src_vector.get_fid());

        rt->execute_index_space(ctx, launcher);
    }

}; // class CSRMatrix


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_CSR_MATRIX_HPP_INCLUDED
