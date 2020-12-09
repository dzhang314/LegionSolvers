#ifndef LEGION_SOLVERS_SPARSE_MATRIX_HPP
#define LEGION_SOLVERS_SPARSE_MATRIX_HPP

#include "MaterializedLinearOperator.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    class SparseMatrix : public MaterializedLinearOperator<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {


      protected:
        Legion::LogicalRegionT<KERNEL_DIM> matrix_region;
        Legion::IndexPartitionT<DOMAIN_DIM> input_partition;
        Legion::IndexPartitionT<RANGE_DIM> output_partition;

        Legion::LogicalPartition column_logical_partition;
        Legion::Color tile_partition;
        std::vector<std::pair<Legion::DomainPoint, Legion::DomainPoint>> nonempty_tiles;


      public:
        explicit SparseMatrix(Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
                              Legion::IndexPartitionT<DOMAIN_DIM> input_partition,
                              Legion::IndexPartitionT<RANGE_DIM> output_partition)
            : matrix_region(matrix_region), input_partition(input_partition), output_partition(output_partition) {}


        virtual void compute_nonempty_tiles(Legion::FieldID fid, Legion::Context ctx, Legion::Runtime *rt) {

            const auto kernel_domain_partition = this->kernel_partition_from_domain_partition(input_partition, ctx, rt);
            const auto kernel_range_partition = this->kernel_partition_from_range_partition(output_partition, ctx, rt);
            column_logical_partition = rt->get_logical_partition(matrix_region, kernel_domain_partition);

            std::map<Legion::IndexSpace, Legion::IndexPartition> map{};
            tile_partition =
                rt->create_cross_product_partitions(ctx, kernel_domain_partition, kernel_range_partition, map);

            std::vector<std::tuple<Legion::DomainPoint, Legion::DomainPoint, Legion::Future>> nonempty_futures{};

            // TODO: Handle multi-dimensional color spaces
            for (Legion::PointInDomainIterator<1> input_iter{rt->get_index_partition_color_space(input_partition)};
                 input_iter(); ++input_iter) {

                const Legion::DomainPoint input_color{*input_iter};
                const auto column = rt->get_logical_subregion_by_color(column_logical_partition, input_color);
                const auto column_partition = rt->get_logical_partition_by_color(column, tile_partition);

                for (Legion::PointInDomainIterator<1> output_iter{
                         rt->get_index_partition_color_space(output_partition)};
                     output_iter(); ++output_iter) {

                    const Legion::DomainPoint output_color{*output_iter};
                    const auto tile = rt->get_logical_subregion_by_color(column_partition, output_color);
                    nonempty_futures.emplace_back(input_color, output_color,
                                                  is_nonempty<KERNEL_DIM>(tile, fid, ctx, rt));
                }
            }
            for (const auto [input_color, output_color, is_nonempty] : nonempty_futures) {
                if (is_nonempty.template get_result<bool>()) {
                    nonempty_tiles.emplace_back(input_color, output_color);
                    std::cout << "Tile " << output_color << ", " << input_color << " contains nonzero values."
                              << std::endl;
                }
            }
        }


    }; // class SparseMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SPARSE_MATRIX_HPP
