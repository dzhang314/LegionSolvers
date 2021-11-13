#include <legion.h>
#include <realm/cmdline.h>

#include "COOMatrixTasks.hpp"
#include "DistributedCOOMatrix.hpp"
#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID,
    FILL_1D_LAPLACIAN_TASK_ID,
    FILL_2D_LAPLACIAN_TASK_ID,
    FILL_2D_PLANE_TASK_ID,
    DIVIDE_TASK_ID,
    NEGATE_TASK_ID,
    MULTIPLY_TASK_ID,
};


struct Args1D {
    Legion::FieldID fid_i;
    Legion::FieldID fid_j;
    Legion::FieldID fid_entry;
    Legion::coord_t grid_length;
};


static void fill_1d_laplacian_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    using T = double;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->arglen == sizeof(Args1D));
    const Args1D args = *reinterpret_cast<const Args1D *>(task->args);

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1>
    i_writer{matrix, args.fid_i};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1>
    j_writer{matrix, args.fid_j};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1>
    entry_writer{matrix, args.fid_entry};

    Legion::PointInDomainIterator<1> iter{matrix};
    for (Legion::coord_t i = 0; i < args.grid_length; ++i) {
        i_writer[*iter] = Legion::Point<1>{i};
        j_writer[*iter] = Legion::Point<1>{i};
        entry_writer[*iter] = static_cast<T>(2.0);
        ++iter;
    }

    for (Legion::coord_t i = 0; i < args.grid_length - 1; ++i) {
        i_writer[*iter] = Legion::Point<1>{i + 1};
        j_writer[*iter] = Legion::Point<1>{i};
        entry_writer[*iter] = static_cast<T>(-1.0);
        ++iter;
        i_writer[*iter] = Legion::Point<1>{i};
        j_writer[*iter] = Legion::Point<1>{i + 1};
        entry_writer[*iter] = static_cast<T>(-1.0);
        ++iter;
    }
}


struct Args2D {
    Legion::FieldID fid_i;
    Legion::FieldID fid_j;
    Legion::FieldID fid_entry;
    Legion::coord_t grid_height;
    Legion::coord_t grid_width;
};


static void fill_2d_laplacian_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    using T = double;
    std::cout << "Constructing 2D Laplacian matrix..." << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->arglen == sizeof(Args2D));
    const Args2D args = *reinterpret_cast<const Args2D *>(task->args);

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1> i_writer{matrix, args.fid_i};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1> j_writer{matrix, args.fid_j};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1> entry_writer{matrix, args.fid_entry};

    Legion::PointInDomainIterator<1> iter{matrix};
    for (Legion::coord_t i = 0; i < args.grid_height; ++i) {
        for (Legion::coord_t j = 0; j < args.grid_width; ++j) {
            i_writer[*iter] = Legion::Point<2>{i, j};
            j_writer[*iter] = Legion::Point<2>{i, j};
            entry_writer[*iter] = static_cast<T>(4.0);
            ++iter;
            if (i > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i - 1, j};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (j > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j - 1};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (i + 1 < args.grid_height) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i + 1, j};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (j + 1 < args.grid_width) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j + 1};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
        }
    }
    std::cout << "Finished constructing 2D Laplacian." << std::endl;
}


void fill_2d_plane_task(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx, Legion::Runtime *rt) {
    assert(regions.size() == 1);
    const auto &vector = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 2> entry_writer{
        vector, 101
    };

    for (Legion::PointInDomainIterator<2> iter{vector}; iter(); ++iter) {
        const auto [i, j] = *iter;
        entry_writer[*iter] = static_cast<double>(i + j);
    }
}


namespace LegionSolvers {


    template <typename T>
    class Planner {

    public:

        Legion::Context ctx;
        Legion::Runtime *rt;
        std::vector<DistributedVector<T> &> solution_vectors;
        std::vector<DistributedVector<T> &> rhs_vectors;
        std::vector<std::tuple<
            std::size_t, std::size_t, const LinearOperator<T> &
        >> operators;
        std::vector<Legion::IndexSpaceT<3>> tile_maps;

        explicit Planner(Legion::Context ctx, Legion::Runtime *rt) :
            ctx(ctx),
            rt(rt),
            solution_vectors(),
            rhs_vectors(),
            operators() {}

        std::size_t add_solution_vector(DistributedVector<T> &sol) {
            solution_vectors.emplace_back(sol);
            return solution_vectors.size() - 1;
        }

        std::size_t add_rhs_vector(DistributedVector<T> &rhs) {
            rhs_vectors.emplace_pack(rhs);
            return rhs_vectors.size() - 1;
        }

        void add_operator(
            std::size_t sol_index, std::size_t rhs_index,
            const MaterializedLinearOperator<T> &matrix
        ) {
            operators.emplace_back(sol_index, rhs_index, matrix);
            tile_maps.emplace_back(matrix.compute_nonempty_tiles(
                solution_vectors[sol_index].index_partition,
                rhs_vectors[rhs_index].index_partition,
                ctx, rt
            ));
        }

    }; // class Planner


} // namespace LegionSolvers


double divide_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context, Legion::Runtime *
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<double>() / b.get_result<double>();
}


double multiply_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context, Legion::Runtime *
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<double>() * b.get_result<double>();
}


double negate_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context, Legion::Runtime *
) {
    assert(task->futures.size() == 1);
    Legion::Future a = task->futures[0];
    return -a.get_result<double>();
}


Legion::Future divide(Legion::Future a, Legion::Future b,
                      Legion::Context ctx, Legion::Runtime *rt) {
    Legion::TaskLauncher launcher{DIVIDE_TASK_ID, Legion::TaskArgument{nullptr, 0}};
    launcher.add_future(a);
    launcher.add_future(b);
    return rt->execute_task(ctx, launcher);
}


Legion::Future multiply(Legion::Future a, Legion::Future b,
                      Legion::Context ctx, Legion::Runtime *rt) {
    Legion::TaskLauncher launcher{MULTIPLY_TASK_ID, Legion::TaskArgument{nullptr, 0}};
    launcher.add_future(a);
    launcher.add_future(b);
    return rt->execute_task(ctx, launcher);
}


Legion::Future negate(Legion::Future a,
                      Legion::Context ctx, Legion::Runtime *rt) {
    Legion::TaskLauncher launcher{NEGATE_TASK_ID, Legion::TaskArgument{nullptr, 0}};
    launcher.add_future(a);
    return rt->execute_task(ctx, launcher);
}


constexpr Legion::coord_t laplacian_2d_kernel_size(Legion::coord_t height, Legion::coord_t width) {
    return 8 + (height - 2) * 2 * 3 + (width - 2) * 2 * 3 + (height - 2) * (width - 2) * 4 + width * height;
}


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    // for (Legion::coord_t num_colors = 1; num_colors <= 20; ++num_colors) {

    //     const Legion::IndexSpaceT<1> index_space =
    //         rt->create_index_space(ctx, Legion::Rect<1>{0, 999});
    //     const Legion::IndexSpaceT<1> color_space =
    //         rt->create_index_space(ctx, Legion::Rect<1>{0, num_colors - 1});

    //     LegionSolvers::DistributedVectorT<double, 1, 1> u{"u", index_space, color_space, ctx, rt};
    //     LegionSolvers::DistributedVectorT<double, 1, 1> v{"v", u.index_partition, ctx, rt};
    //     LegionSolvers::DistributedVectorT<double, 1, 1> w{"w", u.index_partition, ctx, rt};
    //     LegionSolvers::DistributedVectorT<double, 1, 1> x{"x", u.index_partition, ctx, rt};

    //     u = 0.0;
    //     v = 0.0;
    //     w.random_fill();
    //     x.random_fill();

    //     v.axpy(2.0, w);
    //     u.axpy(-1.0, x);
    //     v.axpy(2.0, u);
    //     v.xpay(0.5, x);
    //     v.axpy(-1.0, w);

    //     // v.print();
    //     assert(v.dot(v).get<double>() < 1.0e-20);

    // }

    Legion::coord_t NUM_KERNEL_PARTITIONS = 16;
    Legion::coord_t NUM_VECTOR_PARTITIONS = 4;
    Legion::coord_t GRID_HEIGHT = 1000;
    Legion::coord_t GRID_WIDTH = 1000;
    int MAX_ITERATIONS = 10;

    const Legion::InputArgs &args = Legion::Runtime::get_input_args();

    bool ok = Realm::CommandLineParser()
        .add_option_int("-kp", NUM_KERNEL_PARTITIONS)
        .add_option_int("-vp", NUM_VECTOR_PARTITIONS)
        .add_option_int("-h", GRID_HEIGHT)
        .add_option_int("-w", GRID_WIDTH)
        .add_option_int("-it", MAX_ITERATIONS)
        .parse_command_line(args.argc, (const char **) args.argv);

    assert(ok);

    // const Legion::IndexSpaceT<2> index_space =
    //     rt->create_index_space(ctx, Legion::Rect<2>{{0, 0}, {GRID_HEIGHT - 1, GRID_WIDTH - 1}});
    // const Legion::IndexSpaceT<1> color_space =
    //     rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_VECTOR_PARTITIONS - 1});

    const Legion::IndexSpaceT<1> index_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, GRID_HEIGHT - 1});
    const Legion::IndexSpaceT<1> color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_VECTOR_PARTITIONS - 1});

    LegionSolvers::DistributedVectorT<double, 1, 1> sol{"sol", index_space, color_space, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> rhs{"rhs", sol.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> x{"x", sol.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> p{"x", sol.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> q{"x", sol.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, 1> r{"x", sol.index_partition, ctx, rt};

    // {
    //     Legion::TaskLauncher launcher{
    //         FILL_2D_PLANE_TASK_ID,
    //         Legion::TaskArgument{nullptr, 0}
    //     };
    //     launcher.add_region_requirement(Legion::RegionRequirement{
    //         sol.logical_region, LEGION_WRITE_DISCARD,
    //         LEGION_EXCLUSIVE, sol.logical_region
    //     });
    //     launcher.add_field(0, sol.fid);
    //     rt->execute_task(ctx, launcher);
    // }

    sol.random_fill();

    const Legion::IndexSpaceT<1> matrix_index_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, 3 * GRID_HEIGHT - 3});
    const Legion::IndexSpaceT<1> matrix_color_space =
        rt->create_index_space(ctx, Legion::Rect<1>{0, NUM_KERNEL_PARTITIONS - 1});

    LegionSolvers::DistributedCOOMatrixT<double, 1, 1, 1, 1> coo_matrix{
        "negative_laplacian_1d", matrix_index_space,
        index_space, index_space, matrix_color_space, ctx, rt
    };

    {
        const Args1D args{coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry, GRID_HEIGHT};
        Legion::TaskLauncher launcher{FILL_1D_LAPLACIAN_TASK_ID, Legion::TaskArgument{&args, sizeof(args)}};
        launcher.add_region_requirement(Legion::RegionRequirement{
            coo_matrix.kernel_region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, coo_matrix.kernel_region
        });
        launcher.add_field(0, coo_matrix.fid_i);
        launcher.add_field(0, coo_matrix.fid_j);
        launcher.add_field(0, coo_matrix.fid_entry);
        rt->execute_task(ctx, launcher);
    }

    const LegionSolvers::MaterializedLinearOperator<double> &matrix = coo_matrix;
    const auto tile_map = matrix.compute_nonempty_tiles(
        sol.index_partition, sol.index_partition, ctx, rt
    );

    rhs = 0.0;
    matrix.matvec(rhs, sol, tile_map);
    rhs.print();

    x = 0.0;
    p = rhs;
    r = rhs;
    Legion::Future r2 = r.dot(r);

    std::cout << r2.get_result<double>() << std::endl;

    for (int i = 0; i < 10; ++i) {
        q = 0.0;
        matrix.matvec(q, p, tile_map);
        Legion::Future p_norm = p.dot(q);
        Legion::Future alpha = divide(r2, p_norm, ctx, rt);
        x.axpy(alpha, p);
        r.axpy(negate(alpha, ctx, rt), q);
        Legion::Future r2_new = r.dot(r);
        Legion::Future beta = divide(r2_new, r2, ctx, rt);
        std::cout << r2.get_result<double>() << std::endl;
        r2 = r2_new;
        p.xpay(beta, r);
    }

}


int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", false, false
    );
    LegionSolvers::preregister_cpu_task<fill_1d_laplacian_task>(
        FILL_1D_LAPLACIAN_TASK_ID, "fill_1d_laplacian", false, false
    );
    LegionSolvers::preregister_cpu_task<fill_2d_laplacian_task>(
        FILL_2D_LAPLACIAN_TASK_ID, "fill_2d_laplacian", false, false
    );
    LegionSolvers::preregister_cpu_task<fill_2d_plane_task>(
        FILL_2D_PLANE_TASK_ID, "fill_2d_plane", false, false
    );
    LegionSolvers::preregister_cpu_task<double, multiply_task>(
        MULTIPLY_TASK_ID, "multiply", false, false
    );
    LegionSolvers::preregister_cpu_task<double, divide_task>(
        DIVIDE_TASK_ID, "divide", false, false
    );
    LegionSolvers::preregister_cpu_task<double, negate_task>(
        NEGATE_TASK_ID, "negate", false, false
    );
    LegionSolvers::preregister_solver_tasks(false);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
