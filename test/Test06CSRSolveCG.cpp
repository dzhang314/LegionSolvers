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
    std::size_t prune = 10;
    std::size_t batchsize = 10;
    bool no_print_results = false;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();
    [[maybe_unused]] bool ok =
        Realm::CommandLineParser()
            .add_option_int("-n", grid_size)
            .add_option_int("-vp", num_vector_pieces)
            .add_option_int("-it", num_iterations)
	    .add_option_int("-prune", prune)
	    .add_option_int("-b", batchsize)
            .add_option_bool("-np", no_print_results)
            .parse_command_line(args.argc, (const char **) args.argv);
    assert(ok);

    num_iterations = num_iterations + 2 * prune;

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

    LegionSolvers::CGSolver<ENTRY_T> solver{planner, !no_print_results};


    assert(num_iterations % batchsize == 0);

    Legion::Future ts_start;
    Legion::Future ts_end;

    for (size_t io = 0; io < (num_iterations / batchsize); io++) {
      if (io * batchsize == prune) {
	Legion::Future f = rt->issue_execution_fence(ctx);
	ts_start = rt->get_current_time_in_microseconds(ctx);
      }
      if (io * batchsize == (num_iterations - prune)) {
	Legion::Future f = rt->issue_execution_fence(ctx);
	ts_end = rt->get_current_time_in_microseconds(ctx);
      }
      // TODO (rohany): Does this matter for argument invariance.
      rt->begin_trace(ctx, 15210);
      for (size_t ii = 0; ii < batchsize; ii++) {
        solver.step();
      }
      rt->end_trace(ctx, 15210);
    }

    if (!no_print_results) {
        Legion::Future dummy = Legion::Future::from_value<int>(rt, 0);
        for (std::size_t i = 0; i <= num_iterations; ++i) {
            dummy = solver.residual_norm_squared[i].print(dummy);
        }
    } else {
        solver.last_res_norm.print();
    }

    double start = ts_start.get_result<long long>();
    double end = ts_end.get_result<long long>();
    double seconds = (end - start) * 1e-6;
    double throughput = (num_iterations - (2 * prune)) / seconds;
    LEGION_PRINT_ONCE(rt, ctx, stdout, "Throughput: %f (it/s).\n", throughput);

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
