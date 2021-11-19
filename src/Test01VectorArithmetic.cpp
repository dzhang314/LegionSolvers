#include <legion.h>

#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"
#include "TaskRegistration.hpp"


enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {

    using ENTRY_T = double;
    constexpr int VECTOR_DIM = 1;
    constexpr int VECTOR_COLOR_DIM = 1;
    using VECTOR_COORD_T = Legion::coord_t;
    using VECTOR_COLOR_COORD_T = Legion::coord_t;

    using IndexSpace = Legion::IndexSpaceT<VECTOR_DIM, VECTOR_COORD_T>;
    using ColorSpace = Legion::IndexSpaceT<
        VECTOR_COLOR_DIM, VECTOR_COLOR_COORD_T
    >;
    using Rect = Legion::Rect<VECTOR_DIM, VECTOR_COORD_T>;
    using ColorRect = Legion::Rect<VECTOR_COLOR_DIM, VECTOR_COLOR_COORD_T>;
    using DistributedVector = LegionSolvers::DistributedVectorT<
        ENTRY_T, VECTOR_DIM, VECTOR_COLOR_DIM,
        VECTOR_COORD_T, VECTOR_COLOR_COORD_T
    >;

    for (VECTOR_COORD_T vec_size = 500; vec_size <= 1000; vec_size += 100) {
        for (VECTOR_COLOR_COORD_T num_colors = 1; num_colors <= 20; ++num_colors) {

            const IndexSpace index_space =
                rt->create_index_space(ctx, Rect{0, vec_size - 1});
            const ColorSpace color_space =
                rt->create_index_space(ctx, ColorRect{0, num_colors - 1});

            DistributedVector u{"u", index_space, color_space, ctx, rt};
            DistributedVector v{"v", u.index_partition, ctx, rt};
            DistributedVector w{"w", u.index_partition, ctx, rt};

            u.random_fill();
            v.random_fill();
            w = u;

            w.axpy(1.0, v);
            v.xpay(-1.0, u);
            u.axpy(-0.5, v);
            u.axpy(-0.5, w);

            // u.print();
            u.dot(u).assert_small();

        }
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
