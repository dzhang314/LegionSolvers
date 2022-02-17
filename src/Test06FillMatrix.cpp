#include <cassert>
#include <iostream>
#include <vector>

#include <realm/cmdline.h>
#include <legion.h>

#include "CGSolver.hpp"
#include "COOMatrix.hpp"
#include "DenseDistributedVector.hpp"
#include "ExampleSystems.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "SquarePlanner.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID
};


using ENTRY_T = double;
constexpr Legion::FieldID FID_I = 0;
constexpr Legion::FieldID FID_J = 1;
constexpr Legion::FieldID FID_ENTRY = 2;


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    Legion::coord_t grid_size = 100;
    Legion::coord_t num_pieces = 4;

    const auto color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, num_pieces - 1});
    const Legion::coord_t kernel_size = LegionSolvers::laplacian_1d_kernel_size(grid_size);
    const auto matrix_index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, kernel_size - 1});
    const auto matrix_field_space = LegionSolvers::create_field_space(ctx, rt,
        {
            sizeof(Legion::Point<1>),
            sizeof(Legion::Point<1>),
            sizeof(ENTRY_T)
        },
        {FID_I, FID_J, FID_ENTRY}
    );
    const auto matrix_region = rt->create_logical_region(ctx, matrix_index_space, matrix_field_space);
    const auto matrix_partition = rt->create_equal_partition(ctx, matrix_index_space, color_space);

    {
        const typename
        LegionSolvers::FillCOONegativeLaplacian1DTask<ENTRY_T>::Args args{
            FID_I, FID_J, FID_ENTRY, grid_size};
        Legion::IndexTaskLauncher launcher{
            LegionSolvers::FillCOONegativeLaplacian1DTask<ENTRY_T>::task_id,
            color_space,
            Legion::TaskArgument{&args, sizeof(args)},
            Legion::ArgumentMap{}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            rt->get_logical_partition(matrix_region, matrix_partition),
            0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, matrix_region});
        launcher.add_field(0, FID_I);
        launcher.add_field(0, FID_J);
        launcher.add_field(0, FID_ENTRY);
        rt->execute_index_space(ctx, launcher);
    }

    rt->destroy_index_partition(ctx, matrix_partition);
    rt->destroy_logical_region(ctx, matrix_region);
    rt->destroy_field_space(ctx, matrix_field_space);
    rt->destroy_index_space(ctx, matrix_index_space);
    rt->destroy_index_space(ctx, color_space);

}


int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", true, true, false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
