#include <legion.h>

#include "DistributedCOOMatrix.hpp"
#include "DistributedVector.hpp"
#include "ExampleSystems.hpp"
#include "LegionUtilities.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID
};


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    using ENTRY_T = double; // vary this
    constexpr int VECTOR_DIM = 1; // constant
    constexpr int KERNEL_DIM = 1; // constant
    constexpr int VECTOR_COLOR_DIM = 1; // vary this
    constexpr int KERNEL_COLOR_DIM = 1; // vary this
    using VECTOR_COORD_T = Legion::coord_t;
    using VECTOR_COLOR_COORD_T = Legion::coord_t;

    using DistributedVector = LegionSolvers::DistributedVectorT<
        ENTRY_T, VECTOR_DIM, VECTOR_COLOR_DIM
    >;

    VECTOR_COORD_T GRID_SIZE = 16;
    VECTOR_COLOR_COORD_T NUM_VECTOR_PARTITIONS = 1;
    Legion::coord_t NUM_KERNEL_PARTITIONS = 1;

    const Legion::IndexSpaceT<VECTOR_DIM, VECTOR_COORD_T> index_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, GRID_SIZE - 1});
    const Legion::IndexSpaceT<VECTOR_COLOR_DIM, VECTOR_COLOR_COORD_T> color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_VECTOR_PARTITIONS - 1});

    DistributedVector sol{"sol", index_space, color_space, ctx, rt};
    DistributedVector rhs{"rhs", sol.index_partition, ctx, rt};
    DistributedVector x{"x", sol.index_partition, ctx, rt};

    const Legion::IndexSpaceT<1> matrix_index_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, 3 * GRID_SIZE - 3});
    const Legion::IndexSpaceT<1> matrix_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_KERNEL_PARTITIONS - 1});

    LegionSolvers::DistributedCOOMatrixT<
        ENTRY_T, KERNEL_DIM, VECTOR_DIM, VECTOR_DIM, KERNEL_COLOR_DIM
    > coo_matrix{
        "negative_laplacian_1d", matrix_index_space,
        index_space, index_space, matrix_color_space, ctx, rt
    };

    {
        const LegionSolvers::FillCOONegativeLaplacian1DTask<double>::Args
        args{coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry, GRID_SIZE};
        Legion::TaskLauncher launcher{
            LegionSolvers::FillCOONegativeLaplacian1DTask<double>::task_id,
            Legion::TaskArgument{&args, sizeof(args)}
        };
        launcher.add_region_requirement(Legion::RegionRequirement{
            coo_matrix.kernel_region,
            LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
            coo_matrix.kernel_region
        });
        launcher.add_field(0, coo_matrix.fid_i);
        launcher.add_field(0, coo_matrix.fid_j);
        launcher.add_field(0, coo_matrix.fid_entry);
        rt->execute_task(ctx, launcher);
    }

}


int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
