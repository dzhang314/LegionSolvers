#ifndef LEGION_SOLVERS_COO_MATRIX_HPP
#define LEGION_SOLVERS_COO_MATRIX_HPP

#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

#include <legion.h>

#include "COOMatrixTasks.hpp"
#include "SparseMatrix.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    class COOMatrix : public SparseMatrix<ENTRY_T, KERNEL_DIM,
                                          DOMAIN_DIM, RANGE_DIM> {


      protected:
        Legion::FieldID fid_i;
        Legion::FieldID fid_j;
        Legion::FieldID fid_entry;


      public:
        virtual Legion::IndexPartitionT<KERNEL_DIM>
        kernel_partition_from_domain_partition(
                Legion::IndexPartitionT<DOMAIN_DIM> domain_partition,
                Legion::Context ctx, Legion::Runtime *rt) const override {
            const Legion::IndexSpace color_space =
                rt->get_index_partition_color_space_name(domain_partition);
            const Legion::IndexPartition result =
                rt->create_partition_by_preimage(
                    ctx, domain_partition, this->matrix_region,
                    this->matrix_region, fid_j, color_space,
                    LEGION_DISJOINT_COMPLETE_KIND);
            assert(result.get_dim() == KERNEL_DIM);
            return Legion::IndexPartitionT<KERNEL_DIM>{result};
        }


        virtual Legion::IndexPartitionT<KERNEL_DIM>
        kernel_partition_from_range_partition(
                Legion::IndexPartitionT<RANGE_DIM> range_partition,
                Legion::Context ctx, Legion::Runtime *rt) const override {
            const Legion::IndexSpace color_space =
                rt->get_index_partition_color_space_name(range_partition);
            const Legion::IndexPartition result =
                rt->create_partition_by_preimage(
                    ctx, range_partition, this->matrix_region,
                    this->matrix_region, fid_i, color_space,
                    LEGION_DISJOINT_COMPLETE_KIND);
            assert(result.get_dim() == KERNEL_DIM);
            return Legion::IndexPartitionT<KERNEL_DIM>{result};
        }


        explicit COOMatrix(Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
                           Legion::FieldID fid_i, Legion::FieldID fid_j,
                           Legion::FieldID fid_entry,
                           Legion::IndexPartitionT<DOMAIN_DIM> domain_partition,
                           Legion::IndexPartitionT<RANGE_DIM> range_partition,
                           Legion::Context ctx, Legion::Runtime *rt)

            : SparseMatrix<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>(
                  matrix_region, domain_partition, range_partition),
              fid_i(fid_i), fid_j(fid_j), fid_entry(fid_entry) {

            assert(rt->is_index_partition_complete(ctx, this->domain_partition));
            assert(rt->is_index_partition_disjoint(ctx, this->domain_partition));
            assert(rt->is_index_partition_complete(ctx, this->range_partition));
            assert(rt->is_index_partition_disjoint(ctx, this->range_partition));

            this->compute_nonempty_tiles(fid_entry, ctx, rt);
        }


        virtual void matvec(Legion::LogicalRegion output_vector,
                            Legion::FieldID output_fid,
                            Legion::LogicalRegion input_vector,
                            Legion::FieldID input_fid,
                            Legion::Context ctx,
                            Legion::Runtime *rt) const override {
            zero_fill<ENTRY_T>(output_vector, output_fid,
                               this->range_partition, ctx, rt);
            const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
            Legion::IndexLauncher launcher{
                COOMatvecTask<ENTRY_T, KERNEL_DIM,
                              DOMAIN_DIM, RANGE_DIM>::task_id,
                this->tile_index_space,
                Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])},
                Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;

            launcher.add_region_requirement(Legion::RegionRequirement{
                rt->get_logical_partition(output_vector, this->range_partition),
                PFID_IJ_TO_J, LEGION_REDOP_SUM<ENTRY_T>, LEGION_SIMULTANEOUS,
                output_vector
            });
            launcher.add_field(0, output_fid);

            launcher.add_region_requirement(Legion::RegionRequirement{
                this->column_logical_partition, PFID_IJ_TO_IJ,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, this->matrix_region
            });
            launcher.add_field(1, fid_i);
            launcher.add_field(1, fid_j);
            launcher.add_field(1, fid_entry);

            launcher.add_region_requirement(Legion::RegionRequirement{
                rt->get_logical_partition(input_vector, this->domain_partition),
                PFID_IJ_TO_I, LEGION_READ_ONLY, LEGION_EXCLUSIVE, input_vector
            });
            launcher.add_field(2, input_fid);

            rt->execute_index_space(ctx, launcher);
        }


        virtual void print(Legion::Context ctx,
                           Legion::Runtime *rt) const override {
            // const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
            // Legion::IndexLauncher launcher{
            //     COOPrintTask<ENTRY_T, KERNEL_DIM,
            //                  DOMAIN_DIM, RANGE_DIM>::task_id,
            //     this->tile_index_space,
            //     Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])},
            //     Legion::ArgumentMap{}
            // };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            // launcher.add_region_requirement(Legion::RegionRequirement{
            //     this->column_logical_partition, PFID_IJ_TO_IJ,
            //     LEGION_READ_ONLY, LEGION_EXCLUSIVE, this->matrix_region
            // });
            // launcher.add_field(0, fid_i);
            // launcher.add_field(0, fid_j);
            // launcher.add_field(0, fid_entry);
            // rt->execute_index_space(ctx, launcher);
        }


    }; // class COOMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_HPP
