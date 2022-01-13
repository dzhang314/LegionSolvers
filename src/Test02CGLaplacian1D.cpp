#include <cstddef>

#include <realm/cmdline.h>
#include <legion.h>

#include "BiCGStabSolver.hpp"
#include "CGSolver.hpp"
#include "CGSSolver.hpp"
#include "DistributedCOOMatrix.hpp"
#include "DistributedVector.hpp"
#include "ExampleSystems.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "SquarePlanner.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    FILL_RHS_TASK_ID,
    TOP_LEVEL_TASK_ID
};


using ENTRY_T = double; // vary this
constexpr int VECTOR_DIM = 1;
constexpr int KERNEL_DIM = 1;
using VECTOR_COORD_T = Legion::coord_t; // TODO: can't vary this yet
using KERNEL_COORD_T = Legion::coord_t; // TODO: can't vary this yet
constexpr Legion::TraceID TRACE_ID = 201;


void fill_rhs_task(const Legion::Task *task,
                   const std::vector<Legion::PhysicalRegion> &regions,
                   Legion::Context ctx, Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &region = regions[0];

    assert(task->regions.size() == 1);
    const auto &region_req = task->regions[0];

    assert(region_req.privilege_fields.size() == 1);
    const Legion::FieldID fid = *region_req.privilege_fields.begin();

    assert(task->arglen == sizeof(VECTOR_COORD_T));
    const auto arg_ptr = reinterpret_cast<const VECTOR_COORD_T *>(task->args);
    const VECTOR_COORD_T n = *arg_ptr;

    const LegionSolvers::AffineWriter<ENTRY_T, VECTOR_DIM, VECTOR_COORD_T>
    entry_writer{region, fid};

    for (Legion::PointInDomainIterator<VECTOR_DIM, VECTOR_COORD_T>
         iter{region}; iter(); ++iter) {
        const auto [i] = *iter;
        if (i == 0) {
            entry_writer[*iter] = static_cast<ENTRY_T>(-1);
        } else if (i == n - 1) {
            entry_writer[*iter] = static_cast<ENTRY_T>(+1);
        } else {
            entry_writer[*iter] = static_cast<ENTRY_T>(0);
        }
    }
}


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    constexpr int VECTOR_COLOR_DIM = 1; // vary this
    constexpr int KERNEL_COLOR_DIM = 1; // vary this

    using VECTOR_COLOR_COORD_T = Legion::coord_t; // vary this
    using KERNEL_COLOR_COORD_T = Legion::coord_t; // vary this

    using VectorRect = Legion::Rect<VECTOR_DIM, VECTOR_COORD_T>;
    using KernelRect = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>;
    using VectorColorRect = Legion::Rect<VECTOR_COLOR_DIM,
                                         VECTOR_COLOR_COORD_T>;
    using KernelColorRect = Legion::Rect<KERNEL_COLOR_DIM,
                                         KERNEL_COLOR_COORD_T>;
    using DistributedVector = LegionSolvers::DistributedVectorT<
        ENTRY_T, VECTOR_DIM, VECTOR_COLOR_DIM
    >;

    VECTOR_COORD_T grid_size = 100;
    VECTOR_COLOR_COORD_T num_vector_partitions = 4;
    KERNEL_COLOR_COORD_T num_kernel_partitions = 8;
    std::size_t num_iterations = 10;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();

    bool ok = Realm::CommandLineParser()
        .add_option_int("-n", grid_size)
        .add_option_int("-vp", num_vector_partitions)
        .add_option_int("-kp", num_kernel_partitions)
        .add_option_int("-it", num_iterations)
        .parse_command_line(args.argc, (const char **) args.argv);

    assert(ok);

    const auto vector_index_space = rt->create_index_space(ctx,
        VectorRect{0, grid_size - 1});
    const auto vector_color_space = rt->create_index_space(ctx,
        VectorColorRect{0, num_vector_partitions - 1});

    {

        DistributedVector rhs{"rhs", vector_index_space,
                              vector_color_space, ctx, rt};

        {
            Legion::IndexLauncher launcher{
                FILL_RHS_TASK_ID, vector_color_space,
                Legion::TaskArgument{&grid_size, sizeof(grid_size)},
                Legion::ArgumentMap{}
            };
            launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                rhs.logical_partition, 0,
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, rhs.logical_region
            });
            launcher.add_field(0, rhs.fid);
            rt->execute_index_space(ctx, launcher);
        }

        // rhs.print();

        DistributedVector x{"x", rhs.index_partition, ctx, rt};

        x = 0.0;

        const KERNEL_COORD_T kernel_size = LegionSolvers::laplacian_1d_kernel_size(
            static_cast<KERNEL_COORD_T>(grid_size)
        );

        const auto matrix_index_space = rt->create_index_space(ctx,
            KernelRect{0, kernel_size - 1});

        const auto matrix_color_space = rt->create_index_space(ctx,
            KernelColorRect{0, num_kernel_partitions - 1});

        const auto matrix_field_space = LegionSolvers::create_field_space(
            {
                sizeof(Legion::Point<1, VECTOR_COORD_T>),
                sizeof(Legion::Point<1, VECTOR_COORD_T>),
                sizeof(ENTRY_T)
            },
            {0, 1, 2},
            ctx, rt
        );

        {

            const auto matrix_region = rt->create_logical_region(ctx,
                matrix_index_space, matrix_field_space);

            const auto matrix_partition = rt->create_equal_partition(ctx,
                matrix_index_space, matrix_color_space);

            const auto matrix_logical_partition = rt->get_logical_partition(
                matrix_region, matrix_partition);

            {
                const typename LegionSolvers::FillCOONegativeLaplacian1DTask<ENTRY_T>::Args args{
                    0, 1, 2, grid_size
                };

                Legion::IndexLauncher launcher{
                    LegionSolvers::FillCOONegativeLaplacian1DTask<ENTRY_T>::task_id, matrix_color_space,
                    Legion::TaskArgument{&args, sizeof(args)}, Legion::ArgumentMap{}
                };
                launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
                launcher.add_region_requirement(Legion::RegionRequirement{
                    matrix_logical_partition, 0,
                    LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, matrix_region
                });
                launcher.add_field(0, 0);
                launcher.add_field(0, 1);
                launcher.add_field(0, 2);
                rt->execute_index_space(ctx, launcher);
            }

            LegionSolvers::DistributedCOOMatrixT<
                ENTRY_T, KERNEL_DIM, VECTOR_DIM, VECTOR_DIM, KERNEL_COLOR_DIM,
                KERNEL_COORD_T, VECTOR_COORD_T, VECTOR_COORD_T, KERNEL_COLOR_COORD_T
            > coo_matrix{
                "negative_laplacian_1d_coo", matrix_logical_partition,
                vector_index_space, vector_index_space, 0, 1, 2, ctx, rt
            };

            // coo_matrix.print();

            LegionSolvers::SquarePlanner<ENTRY_T> planner{ctx, rt};
            planner.add_solution_vector(x);
            planner.add_rhs_vector(rhs);
            planner.add_operator(0, 0, coo_matrix);

            LegionSolvers::CGSolver solver{planner};
            solver.setup();
            for (std::size_t i = 0; i < num_iterations; ++i) {
                rt->begin_trace(ctx, TRACE_ID);
                solver.step();
                rt->end_trace(ctx, TRACE_ID);
            }

            Legion::Future dummy = Legion::Future::from_value<int>(rt, 0);
            for (std::size_t i = 0; i <= num_iterations; ++i) {
                dummy = solver.residual_norm_squared[i].print(dummy);
            }

            // x.print();

            rt->destroy_index_partition(ctx, matrix_partition);
            rt->destroy_logical_region(ctx, matrix_region);

        }

        rt->destroy_index_space(ctx, matrix_index_space);
        rt->destroy_index_space(ctx, matrix_color_space);
        rt->destroy_field_space(ctx, matrix_field_space);

    }

    rt->destroy_index_space(ctx, vector_index_space);
    rt->destroy_index_space(ctx, vector_color_space);

}


int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<fill_rhs_task>(
        FILL_RHS_TASK_ID, "fill_rhs", false, true, false
    );
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", true, false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID);
    return Legion::Runtime::start(argc, argv);
}
