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
    class COOMatrix : public SparseMatrix<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {


      protected:
        Legion::FieldID fid_i;
        Legion::FieldID fid_j;
        Legion::FieldID fid_entry;


      public:
        virtual Legion::IndexPartitionT<KERNEL_DIM>
        kernel_partition_from_domain_partition(Legion::IndexPartitionT<DOMAIN_DIM> domain_partition,
                                               Legion::Context ctx,
                                               Legion::Runtime *rt) const override {
            const Legion::IndexSpace color_space = rt->get_index_partition_color_space_name(domain_partition);
            const Legion::IndexPartition result = rt->create_partition_by_preimage(
                ctx, domain_partition, this->matrix_region, this->matrix_region, fid_j, color_space);
            assert(result.get_dim() == KERNEL_DIM);
            return Legion::IndexPartitionT<KERNEL_DIM>{result};
        }


        virtual Legion::IndexPartitionT<KERNEL_DIM>
        kernel_partition_from_range_partition(Legion::IndexPartitionT<RANGE_DIM> range_partition,
                                              Legion::Context ctx,
                                              Legion::Runtime *rt) const override {
            const Legion::IndexSpace color_space = rt->get_index_partition_color_space_name(range_partition);
            const Legion::IndexPartition result = rt->create_partition_by_preimage(
                ctx, range_partition, this->matrix_region, this->matrix_region, fid_i, color_space);
            assert(result.get_dim() == KERNEL_DIM);
            return Legion::IndexPartitionT<KERNEL_DIM>{result};
        }


        explicit COOMatrix(Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
                           Legion::FieldID fid_i,
                           Legion::FieldID fid_j,
                           Legion::FieldID fid_entry,
                           Legion::IndexPartitionT<DOMAIN_DIM> input_partition,
                           Legion::IndexPartitionT<RANGE_DIM> output_partition,
                           Legion::Context ctx,
                           Legion::Runtime *rt)

            : SparseMatrix<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>(
                  matrix_region, input_partition, output_partition),
              fid_i(fid_i), fid_j(fid_j), fid_entry(fid_entry) {

            assert(rt->is_index_partition_complete(ctx, this->input_partition));
            assert(rt->is_index_partition_disjoint(ctx, this->input_partition));
            assert(rt->is_index_partition_complete(ctx, this->output_partition));
            assert(rt->is_index_partition_disjoint(ctx, this->output_partition));

            const Legion::Domain input_color_space = rt->get_index_partition_color_space(this->input_partition);
            const Legion::Domain output_color_space = rt->get_index_partition_color_space(this->output_partition);

            std::cout << "Constructing COOMatrix" << std::endl;
            std::cout << "Input color space: " << input_color_space.get_volume() << std::endl;
            std::cout << "Output color space: " << output_color_space.get_volume() << std::endl;
            this->compute_nonempty_tiles(fid_entry, ctx, rt);
            std::cout << "Computed " << this->nonempty_tiles.size() << " nonempty tiles." << std::endl;
        }


        virtual void matvec(Legion::LogicalRegion output_vector,
                            Legion::FieldID output_fid,
                            Legion::LogicalRegion input_vector,
                            Legion::FieldID input_fid,
                            Legion::Context ctx,
                            Legion::Runtime *rt) const override {
            zero_fill<ENTRY_T>(output_vector, output_fid, this->output_partition, ctx, rt);
            for (const auto [input_color, output_color] : this->nonempty_tiles) {
                std::cout << "Launching matvec on tile " << input_color << " " << output_color << std::endl;
                const auto column = rt->get_logical_subregion_by_color(this->column_logical_partition, input_color);
                const auto column_partition = rt->get_logical_partition_by_color(column, this->tile_partition);
                const auto tile = rt->get_logical_subregion_by_color(column_partition, output_color);
                const auto input_subregion = rt->get_logical_subregion_by_color(
                    rt->get_logical_partition(input_vector, this->input_partition), input_color);
                const auto output_subregion = rt->get_logical_subregion_by_color(
                    rt->get_logical_partition(output_vector, this->output_partition), output_color);
                const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
                {
                    Legion::TaskLauncher launcher{COOMatvecTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::task_id,
                                                  Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])}};
                    launcher.add_region_requirement(Legion::RegionRequirement{output_subregion, LEGION_READ_WRITE,
                                                                              LEGION_EXCLUSIVE, output_vector});
                    launcher.add_field(0, output_fid);
                    launcher.add_region_requirement(
                        Legion::RegionRequirement{tile, LEGION_READ_ONLY, LEGION_EXCLUSIVE, this->matrix_region});
                    launcher.add_field(1, fid_i);
                    launcher.add_field(1, fid_j);
                    launcher.add_field(1, fid_entry);
                    launcher.add_region_requirement(
                        Legion::RegionRequirement{input_subregion, LEGION_READ_ONLY, LEGION_EXCLUSIVE, input_vector});
                    launcher.add_field(2, input_fid);
                    rt->execute_task(ctx, launcher);
                }
            }
        }


        virtual void print(Legion::Context ctx, Legion::Runtime *rt) const override {
            const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
            Legion::TaskLauncher launcher{COOPrintTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::task_id,
                                          Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])}};
            launcher.add_region_requirement(Legion::RegionRequirement{this->matrix_region, LEGION_READ_ONLY,
                                                                      LEGION_EXCLUSIVE, this->matrix_region});
            launcher.add_field(0, fid_i);
            launcher.add_field(0, fid_j);
            launcher.add_field(0, fid_entry);
            rt->execute_task(ctx, launcher);
        }


    }; // class COOMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_HPP
