#ifndef LEGION_SOLVERS_COO_MATRIX_HPP
#define LEGION_SOLVERS_COO_MATRIX_HPP

#include <iostream>
#include <utility>
#include <vector>

#include <legion.h>

#include "SparseMatrix.hpp"
#include "Tasks.hpp"


namespace LegionSolvers {


    class COOMatrix : public SparseMatrix {


      private:
        Legion::LogicalRegion matrix_region;
        Legion::FieldID fid_i;
        Legion::FieldID fid_j;
        Legion::FieldID fid_entry;
        Legion::IndexPartition input_partition;
        Legion::IndexPartition output_partition;

        Legion::LogicalPartition column_logical_partition;
        Legion::Color tile_partition;
        std::vector<std::pair<Legion::DomainPoint, Legion::DomainPoint>>
            nonempty_tiles;


      public:
        explicit COOMatrix(Legion::LogicalRegion matrix_region,
                           Legion::FieldID fid_i, Legion::FieldID fid_j,
                           Legion::FieldID fid_entry,
                           Legion::IndexPartition input_partition,
                           Legion::IndexPartition output_partition,
                           Legion::Context ctx, Legion::Runtime *rt)

            : matrix_region(matrix_region), fid_i(fid_i), fid_j(fid_j),
              fid_entry(fid_entry), input_partition(input_partition),
              output_partition(output_partition) {

            const auto matrix_input_partition =
                rt->create_partition_by_preimage(
                    ctx, input_partition, matrix_region, matrix_region, fid_j,
                    rt->get_index_partition_color_space_name(input_partition));

            const auto matrix_output_partition =
                rt->create_partition_by_preimage(
                    ctx, output_partition, matrix_region, matrix_region, fid_i,
                    rt->get_index_partition_color_space_name(output_partition));

            column_logical_partition = rt->get_logical_partition(
                matrix_region, matrix_input_partition);

            std::map<Legion::IndexSpace, Legion::IndexPartition> map{};
            tile_partition = rt->create_cross_product_partitions(
                ctx, matrix_input_partition, matrix_output_partition, map);

            std::vector<std::tuple<Legion::DomainPoint, Legion::DomainPoint,
                                   Legion::Future>>
                nonempty_futures{};

            for (Legion::PointInDomainIterator<1> input_iter{
                     rt->get_index_partition_color_space(input_partition)};
                 input_iter(); ++input_iter) {

                const Legion::DomainPoint input_color{*input_iter};
                const auto column = rt->get_logical_subregion_by_color(
                    column_logical_partition, input_color);
                const auto column_partition =
                    rt->get_logical_partition_by_color(column, tile_partition);

                for (Legion::PointInDomainIterator<1> output_iter{
                         rt->get_index_partition_color_space(output_partition)};
                     output_iter(); ++output_iter) {

                    const Legion::DomainPoint output_color{*output_iter};
                    const auto tile = rt->get_logical_subregion_by_color(
                        column_partition, output_color);

                    Legion::TaskLauncher launcher{
                        IS_NONEMPTY_TASK_ID, Legion::TaskArgument{nullptr, 0}};
                    launcher.add_region_requirement(Legion::RegionRequirement{
                        tile, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                        matrix_region});
                    launcher.add_field(0, fid_entry);
                    nonempty_futures.emplace_back(
                        input_color, output_color,
                        rt->execute_task(ctx, launcher));
                }
            }
            for (const auto [input_color, output_color, is_nonempty] :
                 nonempty_futures) {
                if (is_nonempty.get_result<bool>()) {
                    nonempty_tiles.emplace_back(input_color, output_color);
                    std::cout << "Tile " << output_color << ", " << input_color
                              << " contains nonzero values." << std::endl;
                } else {
                }
            }
        }


        virtual void launch_matvec(Legion::LogicalRegion output_vector,
                                   Legion::FieldID output_fid,
                                   Legion::LogicalRegion input_vector,
                                   Legion::FieldID input_fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *rt) const override {
            {
                Legion::IndexLauncher launcher{
                    ZERO_FILL_TASK_ID,
                    rt->get_index_partition_color_space(ctx, output_partition),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(output_vector, output_partition),
                    0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, output_vector});
                launcher.add_field(0, output_fid);
                rt->execute_index_space(ctx, launcher);
            }
            for (const auto [input_color, output_color] : nonempty_tiles) {
                const auto column = rt->get_logical_subregion_by_color(
                    column_logical_partition, input_color);
                const auto column_partition =
                    rt->get_logical_partition_by_color(column, tile_partition);
                const auto tile = rt->get_logical_subregion_by_color(
                    column_partition, output_color);
                const auto input_subregion = rt->get_logical_subregion_by_color(
                    rt->get_logical_partition(input_vector, input_partition),
                    input_color);
                const auto output_subregion =
                    rt->get_logical_subregion_by_color(
                        rt->get_logical_partition(output_vector,
                                                  output_partition),
                        output_color);
                const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
                {
                    Legion::TaskLauncher launcher{
                        COO_MATVEC_TASK_ID,
                        Legion::TaskArgument{&fids,
                                             sizeof(Legion::FieldID[3])}};
                    launcher.add_region_requirement(Legion::RegionRequirement{
                        output_subregion, LEGION_READ_WRITE, LEGION_EXCLUSIVE,
                        output_vector});
                    launcher.add_field(0, output_fid);
                    launcher.add_region_requirement(Legion::RegionRequirement{
                        tile, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                        matrix_region});
                    launcher.add_field(1, fid_i);
                    launcher.add_field(1, fid_j);
                    launcher.add_field(1, fid_entry);
                    launcher.add_region_requirement(Legion::RegionRequirement{
                        input_subregion, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                        input_vector});
                    launcher.add_field(2, input_fid);
                    rt->execute_task(ctx, launcher);
                }
            }
        }


    }; // class COOMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_COO_MATRIX_HPP
