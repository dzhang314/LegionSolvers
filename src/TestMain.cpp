#include <map>

#include <legion.h>

#include "COOMatrixTasks.hpp"
#include "DistributedCOOMatrix.hpp"
#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"
#include "TaskRegistration.hpp"

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID,
    FILL_TASK_ID,
};

enum FieldIDs : Legion::FieldID {
    FID_ENTRY,
};

struct Args {
    Legion::FieldID fid_i;
    Legion::FieldID fid_j;
    Legion::FieldID fid_entry;
    Legion::coord_t grid_length;
};

static void fill(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1>
    i_writer{matrix, args.fid_i};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1>
    j_writer{matrix, args.fid_j};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 1>
    entry_writer{matrix, args.fid_entry};

    Legion::PointInDomainIterator<1> iter{matrix};
    for (Legion::coord_t i = 0; i < args.grid_length; ++i) {
        i_writer[*iter] = Legion::Point<1>{i};
        j_writer[*iter] = Legion::Point<1>{i};
        entry_writer[*iter] = static_cast<double>(2.0);
        ++iter;
    }

    for (Legion::coord_t i = 0; i < args.grid_length - 1; ++i) {
        i_writer[*iter] = Legion::Point<1>{i + 1};
        j_writer[*iter] = Legion::Point<1>{i};
        entry_writer[*iter] = static_cast<double>(-1.0);
        ++iter;
        i_writer[*iter] = Legion::Point<1>{i};
        j_writer[*iter] = Legion::Point<1>{i + 1};
        entry_writer[*iter] = static_cast<double>(-1.0);
        ++iter;
    }

}


namespace LegionSolvers {


    template <typename ENTRY_T,
              int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM, int COLOR_DIM,
              typename KERNEL_COORD_T, typename DOMAIN_COORD_T,
              typename RANGE_COORD_T, typename COLOR_COORD_T>
    Legion::IndexSpaceT<3> compute_nonempty_tiles(
        const LegionSolvers::DistributedCOOMatrixT<
            ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM, COLOR_DIM,
            KERNEL_COORD_T, DOMAIN_COORD_T, RANGE_COORD_T, COLOR_COORD_T
        > &matrix,
        Legion::IndexPartitionT<DOMAIN_DIM, DOMAIN_COORD_T> domain_partition,
        Legion::IndexPartitionT<RANGE_DIM, RANGE_COORD_T> range_partition
    ) {

        const Legion::Context &ctx = matrix.ctx;
        Legion::Runtime *const rt = matrix.rt;

        const Legion::Domain kernel_color_space =
            rt->get_index_space_domain(matrix.color_space);
        const Legion::Domain domain_color_space =
            rt->get_index_partition_color_space(domain_partition);
        const Legion::Domain range_color_space =
            rt->get_index_partition_color_space(range_partition);

        const Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_domain_partition =
            matrix.kernel_partition_from_domain_partition(domain_partition);
        const Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_range_partition =
            matrix.kernel_partition_from_range_partition(range_partition);

        const Legion::LogicalPartitionT<KERNEL_DIM, KERNEL_COORD_T> column_logical_partition =
            rt->get_logical_partition(matrix.kernel_region, kernel_domain_partition);

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
                    const Legion::LogicalRegion kernel_piece =
                        rt->get_logical_subregion_by_color(matrix.kernel_logical_partition, kernel_color);
                    const Legion::Domain kernel_piece_domain =
                        rt->get_index_space_domain(kernel_piece.get_index_space());
                    const Legion::Domain intersection_domain =
                        tile_domain.intersection(kernel_piece_domain);
                    const Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>
                    intersection_iter{intersection_domain};
                    if (intersection_iter()) {
                        tile_points.emplace_back(kernel_color[0], domain_color[0], range_color[0]);
                        std::cout << kernel_color[0] << " : " << domain_color[0] << " : " << range_color[0] << std::endl;
                    }
                }
            }
        }
        return rt->create_index_space(ctx, tile_points);
    }


} // namespace LegionSolvers


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    const Legion::IndexSpaceT<1> index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 99});
    const Legion::IndexSpaceT<1> color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 10});

    LegionSolvers::DistributedVectorT<double, 1, 1> u{"u", index_space, color_space, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> v{"v", u.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> w{"w", u.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> x{"x", u.index_partition, ctx, rt};

    u = 0.0;
    // v = 0.0;
    // w.random_fill();
    x.random_fill();

    // v.axpy(2.0, w);
    // u.axpy(-1.0, x);
    // v.axpy(2.0, u);
    // v.xpay(0.5, x);
    // v.axpy(-1.0, w);

    // v.print();

    const Legion::IndexSpaceT<1> matrix_index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 3 * 100 - 2 - 1});
    const Legion::IndexSpaceT<1> matrix_color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 19});
    LegionSolvers::DistributedCOOMatrixT<double, 1, 1, 1, 1> coo_matrix{
        "negative_laplacian_1d", matrix_index_space,
        index_space, index_space, matrix_color_space, ctx, rt
    };

    LegionSolvers::LinearOperator<double> &matrix = coo_matrix;

    {
        // launch fill task
        const Args args{coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry, 100};
        Legion::TaskLauncher launcher{FILL_TASK_ID, Legion::TaskArgument{&args, sizeof(args)}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            coo_matrix.kernel_region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, coo_matrix.kernel_region
        });
        launcher.add_field(0, coo_matrix.fid_i);
        launcher.add_field(0, coo_matrix.fid_j);
        launcher.add_field(0, coo_matrix.fid_entry);
        rt->execute_task(ctx, launcher);
    }

    const auto result = LegionSolvers::compute_nonempty_tiles<
        double, 1, 1, 1, 1,
        Legion::coord_t, Legion::coord_t, Legion::coord_t, Legion::coord_t
    >(coo_matrix, x.index_partition, u.index_partition);

    matrix.matvec(u, x, result);

    x.print();
    u.print();

}



int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", false, false
    );
    LegionSolvers::preregister_cpu_task<fill>(
        FILL_TASK_ID, "fill", false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
