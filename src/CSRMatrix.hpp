#ifndef LEGION_SOLVERS_CSR_MATRIX_HPP
#define LEGION_SOLVERS_CSR_MATRIX_HPP

#include <legion.h>


namespace LegionSolvers {


    template <typename ENTRY_T, int DOMAIN_DIM, int RANGE_DIM>
    struct CSRMatrix : public SparseMatrix<ENTRY_T, 1, DOMAIN_DIM, RANGE_DIM> {


        Legion::FieldID fid_j;
        Legion::FieldID fid_entry;
        Legion::LogicalRegionT<RANGE_DIM> rowptr_region;
        Legion::FieldID fid_rowptr;


        explicit CSRMatrix(
            Legion::LogicalRegionT<1> matrix_region,
            Legion::FieldID fid_j,
            Legion::FieldID fid_entry,
            Legion::LogicalRegionT<RANGE_DIM> rowptr_region,
            Legion::FieldID fid_rowptr,
            Legion::IndexPartitionT<1> kernel_partition,
            Legion::IndexPartitionT<DOMAIN_DIM> input_partition,
            Legion::IndexPartitionT<RANGE_DIM> output_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) : SparseMatrix<ENTRY_T, 1, DOMAIN_DIM, RANGE_DIM>(
                matrix_region, kernel_partition,
                input_partition, output_partition
            ),
            fid_j(fid_j),
            fid_entry(fid_entry),
            rowptr_region(rowptr_region),
            fid_rowptr(fid_rowptr) {
            this->compute_nonempty_tiles(fid_entry, ctx, rt);
        }


        virtual Legion::IndexPartitionT<1>
        kernel_partition_from_domain_partition(
            Legion::IndexPartitionT<DOMAIN_DIM> domain_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) const override {
            return Legion::IndexPartitionT<1>{
                rt->create_partition_by_preimage(
                    ctx, domain_partition,
                    this->matrix_region, this->matrix_region, fid_j,
                    rt->get_index_partition_color_space_name(domain_partition)
                )
            };
        }


        virtual Legion::IndexPartitionT<1>
        kernel_partition_from_range_partition(
            Legion::IndexPartitionT<RANGE_DIM> range_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) const override {
            return Legion::IndexPartitionT<1>{
                rt->create_partition_by_image_range(
                    ctx, this->matrix_region.get_index_space(),
                    rt->get_logical_partition(rowptr_region, range_partition),
                    rowptr_region, fid_rowptr,
                    rt->get_index_partition_color_space_name(range_partition)
                )
            };
        }


        virtual void matvec(
            Legion::LogicalRegion output_vector, Legion::FieldID output_fid,
            Legion::LogicalRegion input_vector, Legion::FieldID input_fid,
            Legion::Context ctx, Legion::Runtime *rt
        ) const override {}


        virtual void print(
            Legion::Context ctx, Legion::Runtime *rt
        ) const override {}


    }; // class CSRMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_CSR_MATRIX_HPP
