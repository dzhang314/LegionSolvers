#include <cassert>

#include <legion.h>
#include <realm/cmdline.h>

#include "CGSolver.hpp"
#include "CSRMatrix.hpp"
#include "ExampleSystems.hpp"
#include "Initialize.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "PartitionedVector.hpp"
#include "SquarePlanner.hpp"
#include "CudaLibs.hpp"

using ENTRY_T = double;
constexpr int VECTOR_DIM = 1;
constexpr int VECTOR_COLOR_DIM = 1;
using VECTOR_COORD_T = Legion::coord_t;
using VECTOR_COLOR_COORD_T = Legion::coord_t;
using VectorRect = Legion::Rect<VECTOR_DIM, VECTOR_COORD_T>;
using VectorColorRect = Legion::Rect<VECTOR_COLOR_DIM, VECTOR_COLOR_COORD_T>;

enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };


void top_level_task(
    const Legion::Task *,
    const std::vector<Legion::PhysicalRegion> &,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    VECTOR_COORD_T grid_size = 100;
    VECTOR_COLOR_COORD_T num_vector_pieces = 4;
    std::size_t num_iterations = 10;
    bool no_print_results = false;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();
    [[maybe_unused]] bool ok =
        Realm::CommandLineParser()
            .add_option_int("-n", grid_size)
            .add_option_int("-vp", num_vector_pieces)
            .add_option_int("-it", num_iterations)
            .add_option_bool("-np", no_print_results)
            .parse_command_line(args.argc, (const char **) args.argv);
    assert(ok);

    LegionSolvers::loadCUDALibs(ctx, rt);

    const auto vector_color_space =
        rt->create_index_space(ctx, VectorColorRect{0, num_vector_pieces - 1});

    LegionSolvers::CSRMatrix<ENTRY_T> csr_matrix =
        LegionSolvers::csr_negative_laplacian_1d<ENTRY_T>(
            ctx, rt, grid_size, vector_color_space
        );

    const auto vector_index_space =
        csr_matrix.get_auxiliary_regions()[0].get_index_space();

    const auto disjoint_vector_partition =
        rt->create_equal_partition(ctx, vector_index_space, vector_color_space);

    LegionSolvers::PartitionedVector<ENTRY_T> rhs(
        ctx, rt, "rhs", disjoint_vector_partition
    );
    rhs.constant_fill(1.0);

    LegionSolvers::PartitionedVector<ENTRY_T> sol(
        ctx, rt, "sol", disjoint_vector_partition
    );
    sol.zero_fill();

    LegionSolvers::SquarePlanner<ENTRY_T> planner{ctx, rt};
    planner.add_sol_vector(sol);
    planner.add_rhs_vector(rhs);
    planner.add_row_partitioned_matrix(csr_matrix, 0, 0);

    LegionSolvers::CGSolver<ENTRY_T> solver{planner};

    for (std::size_t i = 0; i < num_iterations; ++i) { solver.step(); }

    if (!no_print_results) {
        Legion::Future dummy = Legion::Future::from_value<int>(rt, 0);
        for (std::size_t i = 0; i <= num_iterations; ++i) {
            dummy = solver.residual_norm_squared[i].print(dummy);
        }
    }

#ifndef LEGION_SOLVERS_DISABLE_CLEANUP
    rt->destroy_index_partition(ctx, disjoint_vector_partition);
    rt->destroy_index_space(ctx, vector_color_space);
#endif // LEGION_SOLVERS_DISABLE_CLEANUP
}


int main(int argc, char **argv) {
    using LegionSolvers::TaskFlags;
    LegionSolvers::initialize(false, false);
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID,
        "top_level",
        Legion::Processor::LOC_PROC,
        TaskFlags::INNER | TaskFlags::REPLICABLE
    );
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID
    );
    return Legion::Runtime::start(argc, argv);
}
