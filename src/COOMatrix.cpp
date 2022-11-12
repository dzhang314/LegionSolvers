#include "COOMatrix.hpp"

#include "COOMatrixTasks.hpp" // for COOMatvecTask, COORmatvecTask
#include "LibraryOptions.hpp" // for LEGION_SOLVERS_*

using LegionSolvers::COOMatrix;


template <typename ENTRY_T>
Legion::IndexPartition
COOMatrix<ENTRY_T>::kernel_partition_from_domain_partition(
    Legion::IndexPartition domain_partition
) const {
    const Legion::IndexSpace domain_color_space =
        rt->get_index_partition_color_space_name(domain_partition);
    return rt->create_partition_by_preimage(
        ctx,
        domain_partition,
        kernel_region,
        kernel_region,
        fid_col,
        domain_color_space,
        LEGION_COMPUTE_KIND,
        LEGION_AUTO_GENERATE_ID,
        LEGION_SOLVERS_MAPPER_ID
    );
}


template <typename ENTRY_T>
Legion::IndexPartition
COOMatrix<ENTRY_T>::kernel_partition_from_range_partition(
    Legion::IndexPartition range_partition
) const {
    const Legion::IndexSpace range_color_space =
        rt->get_index_partition_color_space_name(range_partition);
    return rt->create_partition_by_preimage(
        ctx,
        range_partition,
        kernel_region,
        kernel_region,
        fid_row,
        range_color_space,
        LEGION_COMPUTE_KIND,
        LEGION_AUTO_GENERATE_ID,
        LEGION_SOLVERS_MAPPER_ID
    );
}


template <typename ENTRY_T>
Legion::IndexPartition
COOMatrix<ENTRY_T>::domain_partition_from_kernel_partition(
    Legion::IndexSpace domain_space, Legion::IndexPartition kernel_partition
) const {
    const Legion::LogicalPartition kernel_logical_partition =
        rt->get_logical_partition(kernel_region, kernel_partition);
    const Legion::IndexSpace kernel_color_space =
        rt->get_index_partition_color_space_name(kernel_partition);
    return rt->create_partition_by_image(
        ctx,
        domain_space,
        kernel_logical_partition,
        kernel_region,
        fid_col,
        kernel_color_space,
        LEGION_COMPUTE_KIND,
        LEGION_AUTO_GENERATE_ID,
        LEGION_SOLVERS_MAPPER_ID
    );
}


template <typename ENTRY_T>
Legion::IndexPartition
COOMatrix<ENTRY_T>::range_partition_from_kernel_partition(
    Legion::IndexSpace range_space, Legion::IndexPartition kernel_partition
) const {
    const Legion::LogicalPartition kernel_logical_partition =
        rt->get_logical_partition(kernel_region, kernel_partition);
    const Legion::IndexSpace kernel_color_space =
        rt->get_index_partition_color_space_name(kernel_partition);
    return rt->create_partition_by_image(
        ctx,
        range_space,
        kernel_logical_partition,
        kernel_region,
        fid_row,
        kernel_color_space,
        LEGION_COMPUTE_KIND,
        LEGION_AUTO_GENERATE_ID,
        LEGION_SOLVERS_MAPPER_ID
    );
}


template <typename ENTRY_T>
void COOMatrix<ENTRY_T>::matvec(
    PartitionedVector<ENTRY_T> &dst_vector,
    const PartitionedVector<ENTRY_T> &src_vector,
    Legion::LogicalPartition kernel_partition,
    Legion::IndexPartition ghost_partition
) const {

    typename COOMatvecTask<ENTRY_T, 0, 0, 0, void, void, void>::Args args;
    args.fid_entry = fid_entry;
    args.fid_row = fid_row;
    args.fid_col = fid_col;

    Legion::IndexLauncher launcher(
        COOMatvecTask<ENTRY_T, 0, 0, 0, void, void, void>::task_id(
            kernel_region.get_index_space(),
            src_vector.get_index_space(),
            dst_vector.get_index_space()
        ),
        dst_vector.get_color_space(),
        Legion::TaskArgument(&args, sizeof(decltype(args))),
        Legion::ArgumentMap()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;

    launcher.add_region_requirement(Legion::RegionRequirement(
        dst_vector.get_logical_partition(),
        0,
        LEGION_READ_WRITE,
        LEGION_EXCLUSIVE,
        dst_vector.get_logical_region()
    ));
    launcher.add_field(0, dst_vector.get_fid());

    launcher.add_region_requirement(Legion::RegionRequirement(
        kernel_partition, 0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, kernel_region
    ));
    launcher.add_field(1, fid_entry);
    launcher.add_field(1, fid_row);
    launcher.add_field(1, fid_col);

    launcher.add_region_requirement(Legion::RegionRequirement(
        rt->get_logical_partition(
            src_vector.get_logical_region(), ghost_partition
        ),
        0,
        LEGION_READ_ONLY,
        LEGION_EXCLUSIVE,
        src_vector.get_logical_region()
    ));
    launcher.add_field(2, src_vector.get_fid());

    rt->execute_index_space(ctx, launcher);
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template Legion::IndexPartition COOMatrix<float>::kernel_partition_from_domain_partition(Legion::IndexPartition) const;
    template Legion::IndexPartition COOMatrix<float>::kernel_partition_from_range_partition(Legion::IndexPartition) const;
    template Legion::IndexPartition COOMatrix<float>::domain_partition_from_kernel_partition(Legion::IndexSpace, Legion::IndexPartition) const;
    template Legion::IndexPartition COOMatrix<float>::range_partition_from_kernel_partition(Legion::IndexSpace, Legion::IndexPartition) const;
    template void COOMatrix<float>::matvec(PartitionedVector<float> &, const PartitionedVector<float> &, Legion::LogicalPartition, Legion::IndexPartition) const;
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template Legion::IndexPartition COOMatrix<double>::kernel_partition_from_domain_partition(Legion::IndexPartition) const;
    template Legion::IndexPartition COOMatrix<double>::kernel_partition_from_range_partition(Legion::IndexPartition) const;
    template Legion::IndexPartition COOMatrix<double>::domain_partition_from_kernel_partition(Legion::IndexSpace, Legion::IndexPartition) const;
    template Legion::IndexPartition COOMatrix<double>::range_partition_from_kernel_partition(Legion::IndexSpace, Legion::IndexPartition) const;
    template void COOMatrix<double>::matvec(PartitionedVector<double> &, const PartitionedVector<double> &, Legion::LogicalPartition, Legion::IndexPartition) const;
#endif // LEGION_SOLVERS_USE_F64
// clang-format on
