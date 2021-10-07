#ifndef LEGION_SOLVERS_SPARSE_MATRIX_HPP
#define LEGION_SOLVERS_SPARSE_MATRIX_HPP

#include "MaterializedLinearOperator.hpp"
#include "UtilityTasks.hpp"
#include "TaskRegistration.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    struct SparseMatrix : public MaterializedLinearOperator<
        ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM
    > {


        Legion::LogicalRegionT<KERNEL_DIM> matrix_region;
        Legion::IndexPartitionT<KERNEL_DIM> kernel_partition;
        Legion::IndexPartitionT<DOMAIN_DIM> domain_partition;
        Legion::IndexPartitionT<RANGE_DIM> range_partition;

        Legion::LogicalPartition column_logical_partition;
        Legion::Color tile_partition;
        // TODO: Handle multi-dimensional color spaces
        Legion::IndexSpaceT<3> tile_index_space;


        explicit SparseMatrix(
            Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
            Legion::IndexPartitionT<KERNEL_DIM> kernel_partition,
            Legion::IndexPartitionT<DOMAIN_DIM> domain_partition,
            Legion::IndexPartitionT<RANGE_DIM> range_partition
        ) : matrix_region(matrix_region),
            kernel_partition(kernel_partition),
            domain_partition(domain_partition),
            range_partition(range_partition) {}


        virtual void compute_nonempty_tiles(
            Legion::Context ctx, Legion::Runtime *rt
        ) {

            const Legion::Domain kernel_color_space = rt->get_index_partition_color_space(kernel_partition);
            const Legion::Domain domain_color_space = rt->get_index_partition_color_space(domain_partition);
            const Legion::Domain range_color_space = rt->get_index_partition_color_space(range_partition);

            const Legion::IndexPartitionT<KERNEL_DIM> kernel_domain_partition = this->kernel_partition_from_domain_partition(domain_partition, ctx, rt);
            const Legion::IndexPartitionT<KERNEL_DIM> kernel_range_partition = this->kernel_partition_from_range_partition(range_partition, ctx, rt);

            const Legion::LogicalPartition kernel_logical_partition = rt->get_logical_partition(matrix_region, kernel_partition);
            column_logical_partition = rt->get_logical_partition(matrix_region, kernel_domain_partition);

            std::map<Legion::IndexSpace, Legion::IndexPartition> map{};
            tile_partition = rt->create_cross_product_partitions(
                ctx,
                kernel_domain_partition,
                kernel_range_partition,
                map,
                LEGION_DISJOINT_COMPLETE_KIND,
                GLOBAL_TILE_PARTITION_COLOR
            );

            // TODO: Handle multi-dimensional color spaces
            std::vector<Legion::Point<3>> tile_points{};
            for (Legion::PointInDomainIterator<1> domain_iter{domain_color_space}; domain_iter(); ++domain_iter) {

                const Legion::DomainPoint domain_color = *domain_iter;
                const Legion::LogicalRegion column = rt->get_logical_subregion_by_color(column_logical_partition, domain_color);
                const auto column_partition = rt->get_logical_partition_by_color(column, tile_partition);

                for (Legion::PointInDomainIterator<1> range_iter{range_color_space}; range_iter(); ++range_iter) {

                    const Legion::DomainPoint range_color = *range_iter;
                    const Legion::LogicalRegion tile = rt->get_logical_subregion_by_color(column_partition, range_color);
                    const Legion::Domain tile_domain = rt->get_index_space_domain(tile.get_index_space());

                    for (Legion::PointInDomainIterator<1> kernel_iter{kernel_color_space}; kernel_iter(); ++kernel_iter) {

                        const Legion::DomainPoint kernel_color = *kernel_iter;
                        const Legion::LogicalRegion kernel_piece = rt->get_logical_subregion_by_color(kernel_logical_partition, kernel_color);
                        const Legion::Domain kernel_piece_domain = rt->get_index_space_domain(kernel_piece.get_index_space());
                        const Legion::Domain intersection_domain =  tile_domain.intersection(kernel_piece_domain);
                        const Legion::PointInDomainIterator<KERNEL_DIM> intersection_iter{intersection_domain};
                        if (intersection_iter()) {
                            tile_points.emplace_back(
                                kernel_color[0], domain_color[0], range_color[0]
                            );
                        }
                    }
                }
            }
            tile_index_space = rt->create_index_space(ctx, tile_points);
        }


    }; // class SparseMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SPARSE_MATRIX_HPP
