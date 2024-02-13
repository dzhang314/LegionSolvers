#include <cassert>
#include <iostream>

#include <legion.h>
#include <realm/cmdline.h>

#include "BiCGStabSolver.hpp"
#include "CGSolver.hpp"
#include "GMRESSolver.hpp"
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
    case 4: {
        Legion::Rect<3, COORD_T> bounds = {{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
        std::vector<std::pair<Legion::Point<3, COORD_T>, ENTRY_T>> offsets;
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, 0, 0}, 88.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{-1, 0, 0}, -6.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{+1, 0, 0}, -6.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, -1, 0}, -6.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, +1, 0}, -6.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, 0, -1}, -6.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, 0, +1}, -6.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{-1, -1, 0}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{-1, +1, 0}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{+1, -1, 0}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{+1, +1, 0}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{-1, 0, -1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{-1, 0, +1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{+1, 0, -1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{+1, 0, +1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, -1, -1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, -1, +1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, +1, -1}, -3.0 / 26.0);
        offsets.emplace_back(Legion::Point<3, COORD_T>{0, +1, +1}, -3.0 / 26.0);
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{-1, -1, -1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{-1, -1, +1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{-1, +1, -1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{-1, +1, +1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{+1, -1, -1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{+1, -1, +1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{+1, +1, -1}, -2.0 / 26.0
        );
        offsets.emplace_back(
            Legion::Point<3, COORD_T>{+1, +1, +1}, -2.0 / 26.0
        );
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
    int solver = 1;
    COORD_T nx = 16;
    COORD_T ny = 16;
    COORD_T nz = 16;
    std::size_t num_pieces = 4;
    std::size_t num_iterations = 20;
    std::size_t iters_per_trace = 1;
    std::size_t num_warmup_traces = 5;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();
    [[maybe_unused]] bool ok =
        Realm::CommandLineParser()
            .add_option_int("-dim", dim)
            .add_option_int("-solver", solver)
            .add_option_int("-nx", nx)
            .add_option_int("-ny", ny)
            .add_option_int("-nz", nz)
            .add_option_int("-vp", num_pieces)
            .add_option_int("-it", num_iterations)
            .add_option_int("-pt", iters_per_trace)
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

    LegionSolvers::PartitionedVector<ENTRY_T> rhs1(
        ctx, rt, "rhs1", equal_partition
    );
    rhs1.constant_fill(1.0);

    LegionSolvers::PartitionedVector<ENTRY_T> sol(
        ctx, rt, "sol", equal_partition
    );
    sol.zero_fill();

    LegionSolvers::PartitionedVector<ENTRY_T> sol1(
        ctx, rt, "sol1", equal_partition
    );
    sol1.zero_fill();

    LegionSolvers::SquarePlanner<ENTRY_T> planner{ctx, rt};
    planner.add_sol_vector(sol);
    planner.add_sol_vector(sol1);
    planner.add_rhs_vector(rhs);
    planner.add_rhs_vector(rhs1);
    planner.add_row_partitioned_matrix(matrix, 0, 0);
    planner.add_row_partitioned_matrix(matrix, 1, 1);

    if (solver == 1) {
        LegionSolvers::CGSolver<ENTRY_T> solver{planner};

        const std::size_t num_traces =
            (num_iterations + iters_per_trace - 1) / iters_per_trace;
        const Legion::TraceID trace_id = 51;

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        for (std::size_t i = 0; i < num_warmup_traces; ++i) {
            rt->begin_trace(ctx, trace_id);
            for (std::size_t j = 0; j < iters_per_trace; ++j) { solver.step(); }
            rt->end_trace(ctx, trace_id);
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const Legion::Future begin = rt->get_current_time_in_nanoseconds(ctx);
        if (rt->get_shard_id(ctx, true) == 0) {
            std::cout << "Performed " << num_warmup_traces << " warmup traces."
                      << std::endl;
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        for (std::size_t i = num_warmup_traces; i < num_traces; ++i) {
            rt->begin_trace(ctx, trace_id);
            for (std::size_t j = 0; j < iters_per_trace; ++j) { solver.step(); }
            rt->end_trace(ctx, trace_id);
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const Legion::Future end = rt->get_current_time_in_nanoseconds(ctx);
        if (rt->get_shard_id(ctx, true) == 0) {
            std::cout << "Performed " << (num_traces - num_warmup_traces)
                      << " timed traces." << std::endl;
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const long long begin_time = begin.get_result<long long>();
        const long long end_time = end.get_result<long long>();
        const double ns_per_iter =
            static_cast<double>(end_time - begin_time) /
            (iters_per_trace * (num_traces - num_warmup_traces));

        std::cout << "Achieved " << ns_per_iter / 1.0e6 << "ms per iteration."
                  << std::endl;
    } else if (solver == 2) {
        LegionSolvers::BiCGStabSolver<ENTRY_T> solver{planner};

        const std::size_t num_traces =
            (num_iterations + iters_per_trace - 1) / iters_per_trace;
        const Legion::TraceID trace_id = 51;

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        for (std::size_t i = 0; i < num_warmup_traces; ++i) {
            rt->begin_trace(ctx, trace_id);
            for (std::size_t j = 0; j < iters_per_trace; ++j) { solver.step(); }
            rt->end_trace(ctx, trace_id);
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const Legion::Future begin = rt->get_current_time_in_nanoseconds(ctx);
        if (rt->get_shard_id(ctx, true) == 0) {
            std::cout << "Performed " << num_warmup_traces << " warmup traces."
                      << std::endl;
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        for (std::size_t i = num_warmup_traces; i < num_traces; ++i) {
            rt->begin_trace(ctx, trace_id);
            for (std::size_t j = 0; j < iters_per_trace; ++j) { solver.step(); }
            rt->end_trace(ctx, trace_id);
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const Legion::Future end = rt->get_current_time_in_nanoseconds(ctx);
        if (rt->get_shard_id(ctx, true) == 0) {
            std::cout << "Performed " << (num_traces - num_warmup_traces)
                      << " timed traces." << std::endl;
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const long long begin_time = begin.get_result<long long>();
        const long long end_time = end.get_result<long long>();
        const double ns_per_iter =
            static_cast<double>(end_time - begin_time) /
            (iters_per_trace * (num_traces - num_warmup_traces));

        std::cout << "Achieved " << ns_per_iter / 1.0e6 << "ms per iteration."
                  << std::endl;
    } else if (solver == 3) {
        LegionSolvers::GMRESSolver<ENTRY_T> solver{planner, 10};

        const std::size_t num_traces =
            (num_iterations + iters_per_trace - 1) / iters_per_trace;
        const Legion::TraceID trace_id = 51;

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        for (std::size_t i = 0; i < num_warmup_traces; ++i) {
            rt->begin_trace(ctx, trace_id);
            for (std::size_t j = 0; j < iters_per_trace; ++j) { solver.step(); }
            rt->end_trace(ctx, trace_id);
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const Legion::Future begin = rt->get_current_time_in_nanoseconds(ctx);
        if (rt->get_shard_id(ctx, true) == 0) {
            std::cout << "Performed " << num_warmup_traces << " warmup traces."
                      << std::endl;
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        for (std::size_t i = num_warmup_traces; i < num_traces; ++i) {
            rt->begin_trace(ctx, trace_id);
            for (std::size_t j = 0; j < iters_per_trace; ++j) { solver.step(); }
            rt->end_trace(ctx, trace_id);
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const Legion::Future end = rt->get_current_time_in_nanoseconds(ctx);
        if (rt->get_shard_id(ctx, true) == 0) {
            std::cout << "Performed " << (num_traces - num_warmup_traces)
                      << " timed traces." << std::endl;
        }

        rt->issue_execution_fence(ctx);
        rt->issue_mapping_fence(ctx);

        const long long begin_time = begin.get_result<long long>();
        const long long end_time = end.get_result<long long>();
        const double ns_per_iter =
            static_cast<double>(end_time - begin_time) /
            (iters_per_trace * (num_traces - num_warmup_traces));

        std::cout << "Achieved " << ns_per_iter / 1.0e6 << "ms per iteration."
                  << std::endl;
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
