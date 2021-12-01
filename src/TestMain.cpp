int main() {}

// #include <realm/cmdline.h>

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

//     // const Legion::InputArgs &args = Legion::Runtime::get_input_args();

//     // bool ok = Realm::CommandLineParser()
//     //     .add_option_int("-kp", NUM_KERNEL_PARTITIONS)
//     //     .add_option_int("-vp", NUM_VECTOR_PARTITIONS)
//     //     .add_option_int("-h", GRID_HEIGHT)
//     //     .add_option_int("-w", GRID_WIDTH)
//     //     .add_option_int("-it", MAX_ITERATIONS)
//     //     .parse_command_line(args.argc, (const char **) args.argv);

//     // assert(ok);

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

//     {
//         const LegionSolvers::FillCOONegativeLaplacian1DTask<double>::Args
//         args{coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry, GRID_SIZE};
//         Legion::TaskLauncher launcher{
//             LegionSolvers::FillCOONegativeLaplacian1DTask<double>::task_id,
//             Legion::TaskArgument{&args, sizeof(args)}
//         };
//         launcher.add_region_requirement(Legion::RegionRequirement{
//             coo_matrix.kernel_region,
//             LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
//             coo_matrix.kernel_region
//         });
//         launcher.add_field(0, coo_matrix.fid_i);
//         launcher.add_field(0, coo_matrix.fid_j);
//         launcher.add_field(0, coo_matrix.fid_entry);
//         rt->execute_task(ctx, launcher);
//     }

//     // const LegionSolvers::MaterializedLinearOperator<double> &matrix = coo_matrix;
//     // const auto tile_map = matrix.compute_nonempty_tiles(
//     //     sol.index_partition, sol.index_partition, ctx, rt
//     // );

//     // x = 0.0;
//     // rhs = 0.0;
//     // sol.random_fill();
//     // matrix.matvec(rhs, sol, tile_map);
//     // // rhs.print();

//     // LegionSolvers::SquarePlanner<double> planner{ctx, rt};
//     // planner.add_solution_vector(x);
//     // planner.add_rhs_vector(rhs);
//     // planner.add_operator(0, 0, coo_matrix);

//     // LegionSolvers::CGSolver solver{planner};
//     // solver.setup();
//     // for (int i = 0; i < 100; ++i) {
//     //     solver.step();
//     //     std::cout << solver.residual_norm_squared.back().get_value() << std::endl;
//     // }

// }
