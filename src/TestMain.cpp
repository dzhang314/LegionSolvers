int main() {}

// #include "CGSolver.hpp"
// #include "COOMatrixTasks.hpp"
// #include "DistributedCOOMatrix.hpp"
// #include "DistributedVector.hpp"
// #include "ExampleSystems.hpp"
// #include "LegionUtilities.hpp"
// #include "LibraryOptions.hpp"
// #include "SquarePlanner.hpp"
// #include "TaskRegistration.hpp"


// enum TaskIDs : Legion::TaskID {
//     TOP_LEVEL_TASK_ID,
//     FILL_2D_PLANE_TASK_ID,
// };


// void fill_2d_plane_task(const Legion::Task *task,
//                         const std::vector<Legion::PhysicalRegion> &regions,
//                         Legion::Context ctx, Legion::Runtime *rt) {
//     assert(regions.size() == 1);
//     const auto &vector = regions[0];

//     const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 2> entry_writer{
//         vector, 101
//     };

//     for (Legion::PointInDomainIterator<2> iter{vector}; iter(); ++iter) {
//         const auto [i, j] = *iter;
//         entry_writer[*iter] = static_cast<double>(i + j);
//     }
// }


// void top_level_task(const Legion::Task *,
//                     const std::vector<Legion::PhysicalRegion> &,
//                     Legion::Context ctx, Legion::Runtime *rt) {

//     // Legion::coord_t NUM_KERNEL_PARTITIONS = 16;
//     // Legion::coord_t NUM_VECTOR_PARTITIONS = 4;
//     // Legion::coord_t GRID_HEIGHT = 1000;
//     // Legion::coord_t GRID_WIDTH = 1000;
//     // int MAX_ITERATIONS = 10;


//     // const Legion::IndexSpaceT<2> index_space =
//     //     rt->create_index_space(ctx, Legion::Rect<2>{{0, 0}, {GRID_HEIGHT - 1, GRID_WIDTH - 1}});
//     // const Legion::IndexSpaceT<1> color_space =
//     //     rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_VECTOR_PARTITIONS - 1});

//     using ENTRY_T = double;

//     Legion::coord_t GRID_SIZE = 100;
//     Legion::coord_t NUM_VECTOR_PARTITIONS = 1;
//     Legion::coord_t NUM_KERNEL_PARTITIONS = 1;

//     const Legion::IndexSpaceT<1> index_space =
//         rt->create_index_space(ctx, Legion::Rect<1>{0, GRID_SIZE - 1});
//     const Legion::IndexSpaceT<1> color_space =
//         rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_VECTOR_PARTITIONS - 1});

//     LegionSolvers::DistributedVectorT<double, 1, 1> sol{"sol", index_space, color_space, ctx, rt};
//     LegionSolvers::DistributedVectorT<double, 1, 1> rhs{"rhs", sol.index_partition, ctx, rt};
//     LegionSolvers::DistributedVectorT<double, 1, 1> x{"x", sol.index_partition, ctx, rt};

//     // {
//     //     Legion::TaskLauncher launcher{
//     //         FILL_2D_PLANE_TASK_ID,
//     //         Legion::TaskArgument{nullptr, 0}
//     //     };
//     //     launcher.add_region_requirement(Legion::RegionRequirement{
//     //         sol.logical_region, LEGION_WRITE_DISCARD,
//     //         LEGION_EXCLUSIVE, sol.logical_region
//     //     });
//     //     launcher.add_field(0, sol.fid);
//     //     rt->execute_task(ctx, launcher);
//     // }

//     const Legion::IndexSpaceT<1> matrix_index_space =
//         rt->create_index_space(ctx, Legion::Rect<1>{0, 3 * GRID_SIZE - 3});
//     const Legion::IndexSpaceT<1> matrix_color_space =
//         rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_KERNEL_PARTITIONS - 1});

//     LegionSolvers::DistributedCOOMatrixT<double, 1, 1, 1, 1> coo_matrix{
//         "negative_laplacian_1d", matrix_index_space,
//         index_space, index_space, matrix_color_space, ctx, rt
//     };

//     // const LegionSolvers::MaterializedLinearOperator<double> &matrix = coo_matrix;
//     // const auto tile_map = matrix.compute_nonempty_tiles(
//     //     sol.index_partition, sol.index_partition, ctx, rt
//     // );

// }
