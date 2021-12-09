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
// constexpr int KERNEL_DIM = 1;
using VECTOR_COORD_T = Legion::coord_t; // TODO: can't vary this yet
using KERNEL_COORD_T = Legion::coord_t; // TODO: can't vary this yet


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

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, ENTRY_T, VECTOR_DIM>
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
    using VectorColorRect = Legion::Rect<VECTOR_COLOR_DIM,
                                         VECTOR_COLOR_COORD_T>;
    using KernelColorRect = Legion::Rect<KERNEL_COLOR_DIM,
                                         KERNEL_COLOR_COORD_T>;
    using DistributedVector = LegionSolvers::DistributedVectorT<
        ENTRY_T, VECTOR_DIM, VECTOR_COLOR_DIM
    >;

    VECTOR_COORD_T grid_size = 100;
    VECTOR_COLOR_COORD_T num_vector_partitions = 1;
    KERNEL_COLOR_COORD_T num_kernel_partitions = 1;
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

    DistributedVector rhs{"rhs", vector_index_space,
                          vector_color_space, ctx, rt};

    {
        Legion::TaskLauncher launcher{
            FILL_RHS_TASK_ID,
            Legion::TaskArgument{&grid_size, sizeof(grid_size)}
        };
        launcher.add_region_requirement(Legion::RegionRequirement{
            rhs.logical_region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
            rhs.logical_region
        });
        launcher.add_field(0, rhs.fid);
        rt->execute_task(ctx, launcher);
    }

    // rhs.print();

    DistributedVector x{"x", rhs.index_partition, ctx, rt};

    x = 0.0;

    const auto matrix_color_space = rt->create_index_space(ctx,
        KernelColorRect{0, num_kernel_partitions - 1});

    const auto coo_matrix = LegionSolvers::negative_laplacian_1d_coo<
        ENTRY_T, VECTOR_COORD_T, KERNEL_COORD_T,
        KERNEL_COLOR_DIM, KERNEL_COLOR_COORD_T
    >(vector_index_space, matrix_color_space, ctx, rt);

    // coo_matrix.print();

    LegionSolvers::SquarePlanner<ENTRY_T> planner{ctx, rt};
    planner.add_solution_vector(x);
    planner.add_rhs_vector(rhs);
    planner.add_operator(0, 0, coo_matrix);

    LegionSolvers::CGSolver solver{planner};
    solver.setup();
    for (std::size_t i = 0; i < num_iterations; ++i) {
        rt->begin_trace(ctx, 201);
        solver.step();
        rt->end_trace(ctx, 201);
    }

    for (std::size_t i = 0; i <= num_iterations; ++i) {
        std::cout << solver.residual_norm_squared[i].get_value() << std::endl;
    }

    // x.print();

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
