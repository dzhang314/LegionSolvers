#include <legion.h>
#include <realm/cmdline.h>

#include "COOMatrix.hpp"
#include "ConjugateGradientSolver.hpp"
#include "ExampleSystems.hpp"
#include "Planner.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID,
    FILL_2D_PLANE_TASK_ID,
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

    Legion::coord_t NUM_KERNEL_PARTITIONS = 16;
    Legion::coord_t NUM_INPUT_PARTITIONS = 4;
    Legion::coord_t NUM_OUTPUT_PARTITIONS = 4;
    Legion::coord_t GRID_HEIGHT = 1000;
    Legion::coord_t GRID_WIDTH = 1000;
    int MAX_ITERATIONS = 10;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();

    bool ok = Realm::CommandLineParser()
        .add_option_int("-kp", NUM_KERNEL_PARTITIONS)
        .add_option_int("-ip", NUM_INPUT_PARTITIONS)
        .add_option_int("-op", NUM_OUTPUT_PARTITIONS)
        .add_option_int("-h", GRID_HEIGHT)
        .add_option_int("-w", GRID_WIDTH)
        .add_option_int("-it", MAX_ITERATIONS)
        .parse_command_line(args.argc, (const char **) args.argv);

    assert(ok);

    // Create index space and two vector regions (input and output).
    const Legion::IndexSpaceT<2> index_space =
        rt->create_index_space(ctx, Legion::Rect<2>{{0, 0}, {GRID_HEIGHT - 1, GRID_WIDTH - 1}});
    const Legion::LogicalRegionT<2> input_vector =
        LegionSolvers::create_region(index_space, {{sizeof(double), FID_ENTRY}}, ctx, rt);
    const Legion::LogicalRegionT<2> output_vector =
        LegionSolvers::create_region(index_space, {{sizeof(double), FID_ENTRY}}, ctx, rt);

    // Partition input and output vectors.
    const Legion::IndexSpaceT<1> input_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_INPUT_PARTITIONS - 1});
    const Legion::IndexSpaceT<1> output_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_OUTPUT_PARTITIONS - 1});
    const Legion::IndexPartitionT<2> input_partition = rt->create_equal_partition(ctx, index_space, input_color_space);
    const Legion::IndexPartitionT<2> output_partition =
        rt->create_equal_partition(ctx, index_space, output_color_space);

    { // Fill input vector entries.
        Legion::TaskLauncher launcher{FILL_2D_PLANE_TASK_ID, Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(
            Legion::RegionRequirement{input_vector, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, input_vector});
        launcher.add_field(0, FID_ENTRY);
        rt->execute_task(ctx, launcher);
    }

    // Create 2D Laplacian matrix.
    const Legion::LogicalRegionT<1> negative_laplacian =
        LegionSolvers::coo_negative_laplacian_2d<double>(
            FID_I, FID_J, FID_ENTRY, GRID_HEIGHT, GRID_WIDTH, ctx, rt);
    const Legion::IndexSpaceT<1> kernel_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_KERNEL_PARTITIONS - 1});
    const Legion::IndexPartitionT<1> kernel_partition{
        rt->create_equal_partition(
            ctx, negative_laplacian.get_index_space(), kernel_color_space)};

    // Call COOMatrix constructor, which computes map of nonzero tiles.
    LegionSolvers::COOMatrix<double, 1, 2, 2> matrix_obj{
        negative_laplacian, FID_I, FID_J, FID_ENTRY,
        kernel_partition, input_partition, output_partition,
        ctx, rt
    };

    // Construct a linear system by computing output_vector = negative_laplacian * input_vector...
    matrix_obj.matvec(output_vector, FID_ENTRY, input_vector, FID_ENTRY, ctx, rt);

    // ...then, discard the input_vector, and ask for the solution of negative_laplacian * x == output_vector.
    LegionSolvers::Planner<double> planner{};
    planner.add_rhs_vector(output_vector, FID_ENTRY, output_partition);
    planner.add_coo_matrix<1, 2, 2>(0, 0, negative_laplacian, kernel_partition, FID_I, FID_J, FID_ENTRY, ctx, rt);

    const Legion::LogicalRegionT<2> solution_vector =
        LegionSolvers::create_region(index_space, {{sizeof(double), FID_ENTRY}}, ctx, rt);
    LegionSolvers::zero_fill<double>(solution_vector, FID_ENTRY, output_partition, ctx, rt);
    planner.add_solution_vector(solution_vector, FID_ENTRY, output_partition);

    LegionSolvers::ConjugateGradientSolver<double> solver{planner, ctx, rt};
    solver.set_max_iterations(MAX_ITERATIONS);
    solver.solve(ctx, rt, true);
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
        entry_writer[*iter] = static_cast<double>(i + j);
    }
}


int main(int argc, char **argv) {
    LegionSolvers::preregister_solver_tasks();

    LegionSolvers::preregister_cpu_task<top_level_task>(TOP_LEVEL_TASK_ID, "top_level", false, false);
    LegionSolvers::preregister_cpu_task<fill_2d_plane_task>(FILL_2D_PLANE_TASK_ID, "fill_2d_plane", true, false);

    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
