#include <legion.h>

#include "COOMatrix.hpp"
#include "ExampleSystems.hpp"
#include "Initialize.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"


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
    constexpr VECTOR_COORD_T grid_size = 20;
    constexpr VECTOR_COLOR_COORD_T num_range_pieces = 4;

    const auto color_space =
        rt->create_index_space(ctx, VectorColorRect{0, num_range_pieces - 1});

    LegionSolvers::COOMatrix<ENTRY_T> coo_matrix =
        LegionSolvers::coo_negative_laplacian_1d<ENTRY_T>(
            ctx, rt, grid_size, color_space
        );

    rt->issue_execution_fence(ctx);

    const auto domain_space =
        rt->create_index_space(ctx, VectorRect{0, grid_size - 1});

    const auto range_space =
        rt->create_index_space(ctx, VectorRect{0, grid_size - 1});

    const auto range_partition =
        rt->create_equal_partition(ctx, range_space, color_space);

    LegionSolvers::print_index_partition(
        ctx, rt, "range_partition", range_partition
    );

    const auto matrix_partition =
        coo_matrix.create_kernel_partition_from_range_partition(range_partition
        );

    LegionSolvers::print_index_partition(
        ctx, rt, "matrix_partition", matrix_partition
    );

    const auto domain_partition =
        coo_matrix.create_domain_partition_from_kernel_partition(
            domain_space, matrix_partition
        );

    LegionSolvers::print_index_partition(
        ctx, rt, "domain_partition", domain_partition
    );

#ifndef LEGION_SOLVERS_DISABLE_CLEANUP
    rt->destroy_index_partition(ctx, domain_partition);
    rt->destroy_index_partition(ctx, matrix_partition);
    rt->destroy_index_partition(ctx, range_partition);
    rt->destroy_index_space(ctx, color_space);
    rt->destroy_index_space(ctx, range_space);
    rt->destroy_index_space(ctx, domain_space);
#endif // LEGION_SOLVERS_DISABLE_CLEANUP
}


int main(int argc, char **argv) {
    using LegionSolvers::TaskFlags;
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID,
        "top_level",
        Legion::Processor::LOC_PROC,
        TaskFlags::INNER | TaskFlags::REPLICABLE
    );
    LegionSolvers::initialize(false, false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID
    );
    return Legion::Runtime::start(argc, argv);
}
