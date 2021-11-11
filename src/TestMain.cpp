#include <legion.h>

#include "COOMatrixTasks.hpp"
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
              int KERNEL_DIM = 1, int DOMAIN_DIM = 1,
              int RANGE_DIM = 1, int COLOR_DIM = 1,
              typename KERNEL_COORD_T = Legion::coord_t,
              typename DOMAIN_COORD_T = Legion::coord_t,
              typename RANGE_COORD_T = Legion::coord_t,
              typename COLOR_COORD_T = Legion::coord_t>
    class DistributedCOOMatrixT {

    public:

        Legion::Context ctx;
        Legion::Runtime *rt;
        std::string name;
        Legion::IndexSpaceT<KERNEL_DIM, KERNEL_COORD_T> kernel_space;
        Legion::IndexSpaceT<DOMAIN_DIM, DOMAIN_COORD_T> domain_space;
        Legion::IndexSpaceT<RANGE_DIM, RANGE_COORD_T> range_space;
        Legion::FieldID fid_i;     // Legion::Rect<RANGE_DIM>
        Legion::FieldID fid_j;     // Legion::Rect<DOMAIN_DIM>
        Legion::FieldID fid_entry; // ENTRY_T
        Legion::LogicalRegionT<KERNEL_DIM, KERNEL_COORD_T> kernel_region;
        Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space;
        Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_index_partition;
        Legion::LogicalPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_logical_partition;

        DistributedCOOMatrixT() = delete;
        DistributedCOOMatrixT(const DistributedCOOMatrixT &) = delete;
        DistributedCOOMatrixT(DistributedCOOMatrixT &&) = delete;
        DistributedCOOMatrixT &operator=(const DistributedCOOMatrixT &) = delete;
        DistributedCOOMatrixT &operator=(DistributedCOOMatrixT &&) = delete;

        explicit DistributedCOOMatrixT(
            const std::string &name,
            Legion::IndexSpaceT<KERNEL_DIM, KERNEL_COORD_T> kernel_space,
            Legion::IndexSpaceT<DOMAIN_DIM, DOMAIN_COORD_T> domain_space,
            Legion::IndexSpaceT<RANGE_DIM, RANGE_COORD_T> range_space,
            Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
            Legion::Context ctx, Legion::Runtime *rt
        ) : ctx(ctx),
            rt(rt),
            name(name),
            kernel_space(kernel_space),
            domain_space(domain_space),
            range_space(range_space),
            fid_i(LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_I),
            fid_j(LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_J),
            fid_entry(LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_ENTRY),
            kernel_region(create_region(
                kernel_space,
                {
                    {sizeof(Legion::Point<RANGE_DIM, RANGE_COORD_T>), fid_i},
                    {sizeof(Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>), fid_j},
                    {sizeof(ENTRY_T), fid_entry},
                },
                ctx, rt
            )),
            color_space(color_space),
            kernel_index_partition(
                rt->create_equal_partition(ctx, kernel_space, color_space)
            ),
            kernel_logical_partition(
                rt->get_logical_partition(kernel_region, kernel_index_partition)
            ) {}

    }; // class DistributedCOOMatrixT


} // namespace LegionSolvers


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    const Legion::IndexSpaceT<1> index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 99});
    const Legion::IndexSpaceT<1> color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 0});

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

    {
        // launch fill task
        const Args args{coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry, 100};
        Legion::TaskLauncher launcher{FILL_TASK_ID, Legion::TaskArgument{&args, sizeof(args)}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            Legion::RegionRequirement{coo_matrix.kernel_region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, coo_matrix.kernel_region}
        });
        launcher.add_field(0, coo_matrix.fid_i);
        launcher.add_field(0, coo_matrix.fid_j);
        launcher.add_field(0, coo_matrix.fid_entry);
        rt->execute_task(ctx, launcher);
    }

    const Legion::FieldID fids[3] = {coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry};
    Legion::TaskLauncher launcher{LegionSolvers::COOMatvecTask<double, 1, 1, 1>::task_id, Legion::TaskArgument{&fids, sizeof(fids)}};
    launcher.add_region_requirement(Legion::RegionRequirement{
        u.logical_region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, u.logical_region
    });
    launcher.add_field(0, u.fid);
    launcher.add_region_requirement(Legion::RegionRequirement{
        coo_matrix.kernel_region, LEGION_READ_ONLY, LEGION_EXCLUSIVE, coo_matrix.kernel_region
    });
    launcher.add_field(1, coo_matrix.fid_i);
    launcher.add_field(1, coo_matrix.fid_j);
    launcher.add_field(1, coo_matrix.fid_entry);
    launcher.add_region_requirement(Legion::RegionRequirement{
        x.logical_region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, x.logical_region
    });
    launcher.add_field(2, x.fid);
    rt->execute_task(ctx, launcher);

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
