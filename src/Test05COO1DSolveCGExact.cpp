#include <cassert>
#include <iostream>
#include <vector>

#include <realm/cmdline.h>
#include <legion.h>

#include "COOMatrix.hpp"
#include "COOMatrixTasks.hpp"
#include "DenseDistributedVector.hpp"
#include "ExampleSystems.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID
};


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


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    constexpr int VECTOR_COLOR_DIM = 1;           // TODO: vary this!
    using VECTOR_COLOR_COORD_T = Legion::coord_t; // TODO: vary this!
    using VectorColorRect = Legion::Rect<VECTOR_COLOR_DIM,
                                         VECTOR_COLOR_COORD_T>;

    VECTOR_COORD_T grid_size = 100;
    VECTOR_COLOR_COORD_T num_vector_pieces = 4;
    std::size_t num_iterations = 10;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();
    bool ok = Realm::CommandLineParser()
        .add_option_int("-n", grid_size)
        .add_option_int("-vp", num_vector_pieces)
        .add_option_int("-it", num_iterations)
        .parse_command_line(args.argc, (const char **) args.argv);
    assert(ok);

    const auto vector_index_space = rt->create_index_space(ctx,
        VectorRect{0, grid_size - 1});

    const auto vector_color_space = rt->create_index_space(ctx,
        VectorColorRect{0, num_vector_pieces - 1});

    const KERNEL_COORD_T kernel_size = LegionSolvers::laplacian_1d_kernel_size(
        static_cast<KERNEL_COORD_T>(grid_size));

    const auto matrix_index_space = rt->create_index_space(ctx,
        KernelRect{0, kernel_size - 1});

    const auto matrix_field_space = LegionSolvers::create_field_space(ctx, rt,
        {
            sizeof(Legion::Point<VECTOR_DIM, VECTOR_COORD_T>),
            sizeof(Legion::Point<VECTOR_DIM, VECTOR_COORD_T>),
            sizeof(ENTRY_T)
        },
        {FID_I, FID_J, FID_ENTRY}
    );

    {
        const Legion::IndexPartitionT<VECTOR_DIM, VECTOR_COORD_T>
        disjoint_vector_partition = rt->create_equal_partition(ctx,
            vector_index_space, vector_color_space);

        const auto matrix_region = rt->create_logical_region(ctx,
            matrix_index_space, matrix_field_space);

        const auto temp_matrix_partition = rt->create_equal_partition(ctx,
            matrix_index_space, vector_color_space);

        {
            const typename
            LegionSolvers::FillCOONegativeLaplacian1DTask<ENTRY_T>::Args args{
                FID_I, FID_J, FID_ENTRY, grid_size};
            Legion::IndexTaskLauncher launcher{
                LegionSolvers::FillCOONegativeLaplacian1DTask<ENTRY_T>::task_id,
                vector_color_space,
                Legion::TaskArgument{&args, sizeof(args)},
                Legion::ArgumentMap{}};
            launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                rt->get_logical_partition(matrix_region, temp_matrix_partition),
                0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, matrix_region});
            launcher.add_field(0, FID_I);
            launcher.add_field(0, FID_J);
            launcher.add_field(0, FID_ENTRY);
            rt->execute_index_space(ctx, launcher);
        }

        LegionSolvers::COOMatrix<ENTRY_T> coo_matrix{
            ctx, rt, matrix_region, FID_I, FID_J, FID_ENTRY};

        const auto matrix_partition =
            coo_matrix.kernel_partition_from_range_partition(
                disjoint_vector_partition);

        const auto matrix_logical_partition = rt->get_logical_partition(
            matrix_region, matrix_partition);

        const auto ghost_vector_partition =
            coo_matrix.domain_partition_from_kernel_partition(
                vector_index_space, matrix_partition);

        {
            LegionSolvers::DenseDistributedVector<ENTRY_T> rhs{ctx, rt, "rhs", disjoint_vector_partition};
            LegionSolvers::DenseDistributedVector<ENTRY_T> sol{ctx, rt, "sol", disjoint_vector_partition};

            rhs.constant_fill(1.0);
            sol.zero_fill();

            LegionSolvers::DenseDistributedVector<ENTRY_T> P{ctx, rt, "P", disjoint_vector_partition};
            LegionSolvers::DenseDistributedVector<ENTRY_T> Q{ctx, rt, "Q", disjoint_vector_partition};
            LegionSolvers::DenseDistributedVector<ENTRY_T> R{ctx, rt, "R", disjoint_vector_partition};

            P = rhs;
            R = rhs;

            std::vector<LegionSolvers::Scalar<ENTRY_T>> residual_norm_squared;

            residual_norm_squared.push_back(R.dot(R));

            const auto P_ghost = rt->get_logical_partition(
                P.get_logical_region(), ghost_vector_partition);

            for (std::size_t i = 0; i < num_iterations; ++i) {
                rt->begin_trace(ctx, 51);
                Q.zero_fill();
                {
                    const Legion::FieldID fids[3] = {FID_I, FID_J, FID_ENTRY};
                    Legion::IndexLauncher launcher{
                        LegionSolvers::COOMatvecTask<ENTRY_T, 1, 1, 1>::task_id,
                        vector_color_space,
                        Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])},
                        Legion::ArgumentMap{}
                    };
                    launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;

                    launcher.add_region_requirement(Legion::RegionRequirement{
                        Q.get_logical_partition(), 0,
                        LegionSolvers::LEGION_REDOP_SUM<ENTRY_T>,
                        LEGION_SIMULTANEOUS, Q.get_logical_region()
                    });
                    launcher.add_field(0, Q.get_fid());

                    launcher.add_region_requirement(Legion::RegionRequirement{
                        matrix_logical_partition, 0,
                        LEGION_READ_ONLY, LEGION_EXCLUSIVE, matrix_region
                    });
                    launcher.add_field(1, FID_I);
                    launcher.add_field(1, FID_J);
                    launcher.add_field(1, FID_ENTRY);

                    launcher.add_region_requirement(Legion::RegionRequirement{
                        P_ghost, 0,
                        LEGION_READ_ONLY, LEGION_EXCLUSIVE, P.get_logical_region()
                    });
                    launcher.add_field(2, P.get_fid());

                    rt->execute_index_space(ctx, launcher);
                }
                LegionSolvers::Scalar<ENTRY_T> p_norm = P.dot(Q);
                LegionSolvers::Scalar<ENTRY_T> alpha = residual_norm_squared.back() / p_norm;
                sol.axpy(alpha, P);
                R.axpy(-alpha, Q);
                LegionSolvers::Scalar<ENTRY_T> r_norm2_new = R.dot(R);
                LegionSolvers::Scalar<ENTRY_T> beta = r_norm2_new / residual_norm_squared.back();
                residual_norm_squared.push_back(r_norm2_new);
                P.xpay(beta, R);
                rt->end_trace(ctx, 51);
            }

            Legion::Future dummy = Legion::Future::from_value<int>(rt, 0);
            for (std::size_t i = 0; i <= num_iterations; ++i) {
                dummy = residual_norm_squared[i].print(dummy);
            }

        }

        rt->destroy_index_partition(ctx, ghost_vector_partition);
        rt->destroy_index_partition(ctx, matrix_partition);
        rt->destroy_index_partition(ctx, disjoint_vector_partition);
        rt->destroy_index_partition(ctx, temp_matrix_partition);
        rt->destroy_logical_region(ctx, matrix_region);
    }

    rt->destroy_index_space(ctx, vector_index_space);
    rt->destroy_index_space(ctx, vector_color_space);
    rt->destroy_index_space(ctx, matrix_index_space);
    rt->destroy_field_space(ctx, matrix_field_space);

}


int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", true, true, false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID);
    return Legion::Runtime::start(argc, argv);
}
