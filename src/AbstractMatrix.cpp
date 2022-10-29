#include "AbstractMatrix.hpp"

#include "LibraryOptions.hpp" // for LEGION_SOLVERS_USE_*

using LegionSolvers::AbstractMatrix;


template <typename ENTRY_T>
Legion::IndexPartition
AbstractMatrix<ENTRY_T>::domain_partition_from_range_partition(
    Legion::IndexSpace domain_space, Legion::IndexPartition range_partition
) const {
    return domain_partition_from_kernel_partition(
        domain_space, kernel_partition_from_range_partition(range_partition)
    );
}


template <typename ENTRY_T>
Legion::IndexPartition
AbstractMatrix<ENTRY_T>::range_partition_from_domain_partition(
    Legion::IndexSpace range_space, Legion::IndexPartition domain_partition
) const {
    return range_partition_from_kernel_partition(
        range_space, kernel_partition_from_domain_partition(domain_partition)
    );
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template Legion::IndexPartition AbstractMatrix<float>::domain_partition_from_range_partition(Legion::IndexSpace, Legion::IndexPartition) const;
    template Legion::IndexPartition AbstractMatrix<float>::range_partition_from_domain_partition(Legion::IndexSpace, Legion::IndexPartition) const;
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template Legion::IndexPartition AbstractMatrix<double>::domain_partition_from_range_partition(Legion::IndexSpace, Legion::IndexPartition) const;
    template Legion::IndexPartition AbstractMatrix<double>::range_partition_from_domain_partition(Legion::IndexSpace, Legion::IndexPartition) const;
#endif // LEGION_SOLVERS_USE_F64
// clang-format on
