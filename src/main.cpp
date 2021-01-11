#include <iostream>
#include <random>

#include <legion.h>

#include "COOMatrix.hpp"
#include "ConjugateGradientSolver.hpp"
#include "ExampleSystems.hpp"
#include "Planner.hpp"
#include "TaskRegistration.hpp"

using LegionSolvers::ConjugateGradientSolver;
using LegionSolvers::COOMatrix;
using LegionSolvers::Planner;

constexpr Legion::coord_t NUM_INPUT_PARTITIONS = 4;
constexpr Legion::coord_t NUM_OUTPUT_PARTITIONS = 4;

constexpr Legion::coord_t GRID_HEIGHT = 5;
constexpr Legion::coord_t GRID_WIDTH = 5;

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID = 10,
    FILL_2D_PLANE_TASK_ID = 19,
};

enum FieldIDs : Legion::FieldID {
    FID_I = 101,
    FID_J = 102,
    FID_ENTRY = 103,
};

void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx,
                    Legion::Runtime *rt) {

    const Legion::LogicalRegionT<1> negative_laplacian =
        LegionSolvers::coo_negative_laplacian_2d<double>(FID_I, FID_J, FID_ENTRY, GRID_HEIGHT, GRID_WIDTH, ctx, rt);

    using LegionSolvers::create_region;

    const Legion::IndexSpaceT<2> index_space =
        rt->create_index_space(ctx, Legion::Rect<2>{{0, 0}, {GRID_HEIGHT - 1, GRID_WIDTH - 1}});

    const auto input_vector = create_region(index_space, {{sizeof(double), FID_ENTRY}}, ctx, rt);
    const auto output_vector = create_region(index_space, {{sizeof(double), FID_ENTRY}}, ctx, rt);

    // Partition input and output vectors.
    const auto input_color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_INPUT_PARTITIONS - 1});
    const auto output_color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_OUTPUT_PARTITIONS - 1});
    const auto input_partition = rt->create_equal_partition(ctx, index_space, input_color_space);
    const auto output_partition = rt->create_equal_partition(ctx, index_space, output_color_space);

    { // Fill input vector entries.
        Legion::TaskLauncher launcher{FILL_2D_PLANE_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{input_vector, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, input_vector});
        launcher.add_field(0, FID_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    // Construct map of nonzero tiles.
    COOMatrix<double, 1, 2, 2> matrix_obj{negative_laplacian, FID_I, FID_J, FID_ENTRY, input_partition,
                                          output_partition,   ctx,   rt};

    // Construct map of nonzero tiles.
    COOMatrix<double, 1, 2, 2> matrix_obj_2{negative_laplacian, FID_I, FID_J, FID_ENTRY, output_partition,
                                            output_partition,   ctx,   rt};

    // Launch matrix-vector multiplication tasks.
    matrix_obj.matvec(output_vector, FID_ENTRY, input_vector, FID_ENTRY, ctx, rt);
    // LegionSolvers::zero_fill<double>(output_vector, FID_ENTRY, output_partition, ctx, rt);

    // Construct map of nonzero tiles.
    COOMatrix<double, 1, 2, 2> matrix_obj_3{negative_laplacian, FID_I,           FID_J, FID_ENTRY,
                                            input_partition,    input_partition, ctx,   rt};

    Planner<double> planner{};
    planner.add_rhs(output_vector, FID_ENTRY, output_partition);

    planner.add_coo_matrix<1, 2, 2>(0, 0, negative_laplacian, FID_I, FID_J, FID_ENTRY, ctx, rt);

    ConjugateGradientSolver<double> solver{planner, ctx, rt};
    solver.set_max_iterations(2);
    solver.solve(ctx, rt);
}

void fill_2d_plane_task(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx,
                        Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &vector = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 2> entry_writer{vector, FID_ENTRY};

    for (Legion::PointInDomainIterator<2> iter{vector}; iter(); ++iter) {
        const auto [i, j] = *iter;
        entry_writer[*iter] = i + j;
    }
}

int main(int argc, char **argv) {
    using namespace LegionSolvers;
    preregister_solver_tasks(false);

    preregister_cpu_task<top_level_task>(TOP_LEVEL_TASK_ID, "top_level");
    preregister_cpu_task<fill_2d_plane_task>(FILL_2D_PLANE_TASK_ID, "fill_2d_plane");

    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
