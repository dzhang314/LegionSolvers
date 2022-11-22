#include <legion.h>

#include "COOMatrix.hpp"
#include "ExampleSystems.hpp"
#include "Initialize.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"


using ENTRY_T = double; // vary this
constexpr int VECTOR_DIM = 1;
constexpr int KERNEL_DIM = 1;
using VECTOR_COORD_T = Legion::coord_t; // TODO: can't vary this yet
using KERNEL_COORD_T = Legion::coord_t; // TODO: can't vary this yet
using VectorRect = Legion::Rect<VECTOR_DIM, VECTOR_COORD_T>;
using KernelRect = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>;
constexpr Legion::FieldID FID_I = 0;
constexpr Legion::FieldID FID_J = 1;
constexpr Legion::FieldID FID_ENTRY = 2;

enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };


void top_level_task(
    const Legion::Task *,
    const std::vector<Legion::PhysicalRegion> &,
    Legion::Context ctx,
    Legion::Runtime *rt
) {

    constexpr int VECTOR_COLOR_DIM = 1;           // TODO: vary this!
    using VECTOR_COLOR_COORD_T = Legion::coord_t; // TODO: vary this!
    using VectorColorRect =
        Legion::Rect<VECTOR_COLOR_DIM, VECTOR_COLOR_COORD_T>;

    constexpr VECTOR_COORD_T grid_size = 20;
    constexpr VECTOR_COLOR_COORD_T num_range_pieces = 4;

    const auto domain_space =
        rt->create_index_space(ctx, VectorRect{0, grid_size - 1});

    const auto range_space =
        rt->create_index_space(ctx, VectorRect{0, grid_size - 1});

    const auto range_color_space =
        rt->create_index_space(ctx, VectorColorRect{0, num_range_pieces - 1});

    const KERNEL_COORD_T kernel_size = LegionSolvers::laplacian_1d_kernel_size(
        static_cast<KERNEL_COORD_T>(grid_size)
    );

    const auto matrix_index_space =
        rt->create_index_space(ctx, KernelRect{0, kernel_size - 1});

    const auto matrix_field_space = LegionSolvers::create_field_space(
        ctx,
        rt,
        {sizeof(Legion::Point<VECTOR_DIM, VECTOR_COORD_T>),
         sizeof(Legion::Point<VECTOR_DIM, VECTOR_COORD_T>),
         sizeof(ENTRY_T)},
        {FID_I, FID_J, FID_ENTRY}
    );

    {
        const auto range_partition =
            rt->create_equal_partition(ctx, range_space, range_color_space);

        LegionSolvers::print_index_partition(
            ctx, rt, "range_partition", range_partition
        );

        const auto matrix_region = rt->create_logical_region(
            ctx, matrix_index_space, matrix_field_space
        );

        {
            const typename LegionSolvers::FillCOONegativeLaplacianTask<
                ENTRY_T,
                1,
                1,
                1,
                Legion::coord_t,
                Legion::coord_t,
                Legion::coord_t>::Args args{FID_I, FID_J, FID_ENTRY, grid_size};
            Legion::TaskLauncher launcher{
                LegionSolvers::FillCOONegativeLaplacianTask<
                    ENTRY_T,
                    1,
                    1,
                    1,
                    Legion::coord_t,
                    Legion::coord_t,
                    Legion::coord_t>::task_id,
                Legion::TaskArgument{&args, sizeof(args)}};
            launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                matrix_region,
                LEGION_WRITE_DISCARD,
                LEGION_EXCLUSIVE,
                matrix_region});
            launcher.add_field(0, FID_I);
            launcher.add_field(0, FID_J);
            launcher.add_field(0, FID_ENTRY);
            rt->execute_task(ctx, launcher);
        }

        LegionSolvers::COOMatrix<ENTRY_T> coo_matrix(
            ctx, rt, matrix_region, FID_I, FID_J, FID_ENTRY
        );

        const auto matrix_partition =
            coo_matrix.kernel_partition_from_range_partition(range_partition);

        LegionSolvers::print_index_partition(
            ctx, rt, "matrix_partition", matrix_partition
        );

        const auto domain_partition =
            coo_matrix.domain_partition_from_kernel_partition(
                domain_space, matrix_partition
            );

        LegionSolvers::print_index_partition(
            ctx, rt, "domain_partition", domain_partition
        );

        rt->destroy_index_partition(ctx, domain_partition);
        rt->destroy_index_partition(ctx, matrix_partition);
        rt->destroy_index_partition(ctx, range_partition);
        rt->destroy_logical_region(ctx, matrix_region);
    }

    rt->destroy_index_space(ctx, domain_space);
    rt->destroy_index_space(ctx, range_space);
    rt->destroy_index_space(ctx, range_color_space);
    rt->destroy_index_space(ctx, matrix_index_space);
    rt->destroy_field_space(ctx, matrix_field_space);
}


int main(int argc, char **argv) {
    using LegionSolvers::TaskFlags;
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID,
        "top_level",
        Legion::Processor::LOC_PROC,
        TaskFlags::INNER | TaskFlags::REPLICABLE
    );
    LegionSolvers::initialize(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID
    );
    return Legion::Runtime::start(argc, argv);
}
