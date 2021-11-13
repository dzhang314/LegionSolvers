#ifndef LEGION_SOLVERS_MATERIALIZED_LINEAR_OPERATOR_HPP
#define LEGION_SOLVERS_MATERIALIZED_LINEAR_OPERATOR_HPP

#include <map>

#include <legion.h>

#include "LinearOperator.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class MaterializedLinearOperator : public LinearOperator<ENTRY_T> {

    public:

        virtual Legion::IndexPartition kernel_partition_from_domain_partition(
            Legion::IndexPartition domain_partition
        ) const = 0;

        virtual Legion::IndexPartition kernel_partition_from_range_partition(
            Legion::IndexPartition range_partition
        ) const = 0;

        virtual Legion::IndexSpaceT<3> compute_nonempty_tiles(
            Legion::IndexPartition domain_partition,
            Legion::IndexPartition range_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) const = 0;

    }; // class MaterializedLinearOperator


    template <typename ENTRY_T,
              int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM,
              typename KERNEL_COORD_T,
              typename DOMAIN_COORD_T, typename RANGE_COORD_T>
    class MaterializedLinearOperatorT : public MaterializedLinearOperator<ENTRY_T> {

    public:

        virtual Legion::LogicalRegionT<KERNEL_DIM, KERNEL_COORD_T>
        get_kernel_region() const = 0;

        virtual Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T>
        get_kernel_partition() const = 0;

        Legion::IndexSpaceT<3> compute_nonempty_tiles(
            Legion::IndexPartitionT<DOMAIN_DIM, DOMAIN_COORD_T> domain_partition,
            Legion::IndexPartitionT<RANGE_DIM, RANGE_COORD_T> range_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) const {
            const Legion::Domain kernel_color_space =
                rt->get_index_partition_color_space(get_kernel_partition());
            const Legion::Domain domain_color_space =
                rt->get_index_partition_color_space(domain_partition);
            const Legion::Domain range_color_space =
                rt->get_index_partition_color_space(range_partition);

            const Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_domain_partition{
                this->kernel_partition_from_domain_partition(domain_partition)
            };
            const Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_range_partition{
                this->kernel_partition_from_range_partition(range_partition)
            };

            const Legion::LogicalPartitionT<KERNEL_DIM, KERNEL_COORD_T> column_logical_partition =
                rt->get_logical_partition(get_kernel_region(), kernel_domain_partition);

            std::map<Legion::IndexSpace, Legion::IndexPartition> map{};
            const Legion::Color tile_partition = rt->create_cross_product_partitions(
                ctx,
                kernel_domain_partition,
                kernel_range_partition,
                map,
                LEGION_DISJOINT_COMPLETE_KIND,
                LEGION_SOLVERS_DEFAULT_TILE_PARTITION_COLOR
            );

            // TODO: Handle multi-dimensional color spaces
            // TODO: DomainPoint isn't templated on coordinate type
            // TODO: Once it is, how do we promote three coordinate types to a common type?
            using iter_t = Legion::PointInDomainIterator<1>;
            std::vector<Legion::Point<3>> tile_points{};
            for (iter_t domain_iter{domain_color_space}; domain_iter(); ++domain_iter) {
                const Legion::DomainPoint domain_color = *domain_iter;
                const Legion::LogicalRegion column =
                    rt->get_logical_subregion_by_color(column_logical_partition, domain_color);
                const auto column_partition =
                    rt->get_logical_partition_by_color(column, tile_partition);
                for (iter_t range_iter{range_color_space}; range_iter(); ++range_iter) {
                    const Legion::DomainPoint range_color = *range_iter;
                    const Legion::LogicalRegion tile =
                        rt->get_logical_subregion_by_color(column_partition, range_color);
                    const Legion::Domain tile_domain =
                        rt->get_index_space_domain(tile.get_index_space());
                    for (iter_t kernel_iter{kernel_color_space}; kernel_iter(); ++kernel_iter) {
                        const Legion::DomainPoint kernel_color = *kernel_iter;
                        const Legion::LogicalRegion kernel_piece = rt->get_logical_subregion_by_color(
                            rt->get_logical_partition(get_kernel_region(), get_kernel_partition()),
                            kernel_color
                        );
                        const Legion::Domain kernel_piece_domain =
                            rt->get_index_space_domain(kernel_piece.get_index_space());
                        const Legion::Domain intersection_domain =
                            tile_domain.intersection(kernel_piece_domain);
                        const Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>
                        intersection_iter{intersection_domain};
                        if (intersection_iter()) {
                            tile_points.emplace_back(kernel_color[0], domain_color[0], range_color[0]);
                        }
                    }
                }
            }
            return rt->create_index_space(ctx, tile_points);
        }

        virtual Legion::IndexSpaceT<3> compute_nonempty_tiles(
            Legion::IndexPartition domain_partition,
            Legion::IndexPartition range_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) const override {
            return compute_nonempty_tiles(
                Legion::IndexPartitionT<DOMAIN_DIM, DOMAIN_COORD_T>{domain_partition},
                Legion::IndexPartitionT<RANGE_DIM, RANGE_COORD_T>{range_partition},
                ctx, rt
            );
        }

    }; // class MaterializedLinearOperatorT


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_MATERIALIZED_LINEAR_OPERATOR_HPP
