#ifndef LEGION_SOLVERS_SPARSE_MATRIX_HPP
#define LEGION_SOLVERS_SPARSE_MATRIX_HPP

#include "MaterializedLinearOperator.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T, int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
    class SparseMatrix : public MaterializedLinearOperator<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> {


      public:
        Legion::LogicalRegionT<KERNEL_DIM> matrix_region;
        Legion::IndexPartitionT<DOMAIN_DIM> domain_partition;
        Legion::IndexPartitionT<RANGE_DIM> range_partition;

        Legion::LogicalPartition column_logical_partition;
        Legion::Color tile_partition;
        std::vector<std::pair<Legion::DomainPoint, Legion::DomainPoint>> nonempty_tiles;


      public:
        explicit SparseMatrix(Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
                              Legion::IndexPartitionT<DOMAIN_DIM> domain_partition,
                              Legion::IndexPartitionT<RANGE_DIM> range_partition)
            : matrix_region(matrix_region), domain_partition(domain_partition), range_partition(range_partition) {}


        virtual void compute_nonempty_tiles(Legion::FieldID fid, Legion::Context ctx, Legion::Runtime *rt) {

            std::cout << "Computing nonempty tiles." << std::endl;

            const Legion::IndexPartitionT<KERNEL_DIM> kernel_domain_partition =
                this->kernel_partition_from_domain_partition(domain_partition, ctx, rt);
            const Legion::IndexPartitionT<KERNEL_DIM> kernel_range_partition =
                this->kernel_partition_from_range_partition(range_partition, ctx, rt);
            column_logical_partition = rt->get_logical_partition(matrix_region, kernel_domain_partition);

            std::map<Legion::IndexSpace, Legion::IndexPartition> map{};
            tile_partition =
                rt->create_cross_product_partitions(ctx, kernel_domain_partition, kernel_range_partition, map);

            // TODO: Handle multi-dimensional color spaces
            for (Legion::PointInDomainIterator<1> domain_color_iter{
                     rt->get_index_partition_color_space(domain_partition)};
                 domain_color_iter(); ++domain_color_iter) {

                const Legion::DomainPoint domain_color{*domain_color_iter};
                const Legion::LogicalRegion column =
                    rt->get_logical_subregion_by_color(column_logical_partition, domain_color);
                const auto column_partition = rt->get_logical_partition_by_color(column, tile_partition);

                for (Legion::PointInDomainIterator<1> range_color_iter{
                         rt->get_index_partition_color_space(range_partition)};
                     range_color_iter(); ++range_color_iter) {

                    const Legion::DomainPoint range_color{*range_color_iter};
                    const Legion::LogicalRegion tile =
                        rt->get_logical_subregion_by_color(column_partition, range_color);
                    const auto tile_domain = rt->get_index_space_domain(tile.get_index_space());
                    const Legion::PointInDomainIterator<KERNEL_DIM> tile_iter{tile_domain};

                    if (tile_iter()) {
                        nonempty_tiles.emplace_back(domain_color, range_color);
                        std::cout << "Tile " << range_color << ", " << domain_color << " contains nonzero values."
                                  << std::endl;
                    }
                }
            }
            std::cout << "Computed " << nonempty_tiles.size() << " nonempty tiles." << std::endl;
        }


    }; // class SparseMatrix


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SPARSE_MATRIX_HPP
