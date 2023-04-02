#include <cassert>
#include <iostream>

#include <legion.h>
#include <realm/cmdline.h>

#include "CGSolver.hpp"
#include "Initialize.hpp"
#include "LegionUtilities.hpp"
#include "PartitionedVector.hpp"
#include "SquarePlanner.hpp"
#include "StencilGenerator.hpp"


using COORD_T = Legion::coord_t;
using ENTRY_T = double;


enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };


LegionSolvers::CSRMatrix<ENTRY_T> get_laplacian_matrix(
    Legion::Context ctx,
    Legion::Runtime *rt,
    int dim,
    COORD_T nx,
    COORD_T ny,
    COORD_T nz,
    std::size_t num_pieces
) {
    switch (dim) {
    case 1: {
        Legion::Rect<1, COORD_T> bounds = {{0}, {nx - 1}};
        std::vector<std::pair<Legion::Point<1, COORD_T>, ENTRY_T>> offsets;
        offsets.emplace_back(Legion::Point<1, COORD_T>{0}, +2.0);
        offsets.emplace_back(Legion::Point<1, COORD_T>{-1}, -1.0);
        offsets.emplace_back(Legion::Point<1, COORD_T>{+1}, -1.0);
        const auto result = LegionSolvers::create_linearized_csr_stencil_matrix(
            ctx, rt, bounds, offsets, num_pieces
        );
        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);
        return result;
    }
    case 2: {
        Legion::Rect<2, COORD_T> bounds = {{0, 0}, {nx - 1, ny - 1}};
        std::vector<std::pair<Legion::Point<2, COORD_T>, ENTRY_T>> offsets;
        offsets.emplace_back(Legion::Point<2, COORD_T>{0, 0}, +4.0);
        offsets.emplace_back(Legion::Point<2, COORD_T>{-1, 0}, -1.0);
        offsets.emplace_back(Legion::Point<2, COORD_T>{+1, 0}, -1.0);
        offsets.emplace_back(Legion::Point<2, COORD_T>{0, -1}, -1.0);
        offsets.emplace_back(Legion::Point<2, COORD_T>{0, +1}, -1.0);
        const auto result = LegionSolvers::create_linearized_csr_stencil_matrix(
            ctx, rt, bounds, offsets, num_pieces
        );
        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);
        return result;
    }
    case 3: {
        Legion::Rect<3, COORD_T> bounds = {{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
        std::vector<std::pair<Legion::Point<3, COORD_T>, ENTRY_T>> offsets;
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, 0, 0}, +6.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{-1, 0, 0}, -1.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{+1, 0, 0}, -1.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, -1, 0}, -1.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, +1, 0}, -1.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, 0, -1}, -1.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, 0, +1}, -1.0);
        const auto result = LegionSolvers::create_linearized_csr_stencil_matrix(
            ctx, rt, bounds, offsets, num_pieces
        );
        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);
        return result;
    }
    default: {
        std::cout << "INVALID DIM" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    }
}


void top_level_task(
    const Legion::Task *,
    const std::vector<Legion::PhysicalRegion> &,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    int dim = 1;
    COORD_T nx = 16;
    COORD_T ny = 16;
    COORD_T nz = 16;
    std::size_t num_pieces = 4;
    std::size_t num_iterations = 10;
    bool no_print_results = false;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();
    [[maybe_unused]] bool ok =
        Realm::CommandLineParser()
            .add_option_int("-dim", dim)
            .add_option_int("-nx", nx)
            .add_option_int("-ny", ny)
            .add_option_int("-nz", nz)
            .add_option_int("-vp", num_pieces)
            .add_option_int("-it", num_iterations)
            .add_option_bool("-np", no_print_results)
            .parse_command_line(args.argc, (const char **) args.argv);
    assert(ok);

    const LegionSolvers::CSRMatrix<ENTRY_T> matrix =
        get_laplacian_matrix(ctx, rt, dim, nx, ny, nz, num_pieces);

    const auto rowptr_region = matrix.get_auxiliary_regions()[0];
    const auto vector_index_space = rowptr_region.get_index_space();
    const auto color_space = rt->create_index_space(
        ctx, Legion::Rect<1, COORD_T>{0, static_cast<COORD_T>(num_pieces) - 1}
    );
    const auto equal_partition =
        rt->create_equal_partition(ctx, vector_index_space, color_space);

    LegionSolvers::PartitionedVector<ENTRY_T> rhs(
        ctx, rt, "rhs", equal_partition
    );
    rhs.constant_fill(1.0);

    LegionSolvers::PartitionedVector<ENTRY_T> sol(
        ctx, rt, "sol", equal_partition
    );
    sol.zero_fill();

    LegionSolvers::SquarePlanner<ENTRY_T> planner{ctx, rt};
    planner.add_sol_vector(sol);
    planner.add_rhs_vector(rhs);
    planner.add_row_partitioned_matrix(matrix, 0, 0);

    LegionSolvers::CGSolver<ENTRY_T> solver{planner};

    for (std::size_t i = 0; i < num_iterations; ++i) { solver.step(); }

    if (!no_print_results) {
        Legion::Future dummy = Legion::Future::from_value<int>(rt, 0);
        for (std::size_t i = 0; i <= num_iterations; ++i) {
            dummy = solver.residual_norm_squared[i].print(dummy);
        }
    }

#ifndef LEGION_SOLVERS_DISABLE_CLEANUP
    rt->destroy_index_partition(ctx, equal_partition);
    rt->destroy_index_space(ctx, color_space);
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
    return Legion::Runtime::start(argc, argv, false, false);
}
