#include <legion.h>

#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs : Legion::FieldID {
    FID_ENTRY,
};

void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {
    const Legion::IndexSpaceT<1> index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 99});
    const Legion::IndexSpaceT<1> color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 10});

    LegionSolvers::DistributedVectorT<double, 1, Legion::coord_t, 1, Legion::coord_t> v{"v", index_space, color_space, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, Legion::coord_t, 1, Legion::coord_t> w{"w", index_space, color_space, ctx, rt};
    v.zero_fill();
    w.random_fill();
    Legion::Future alpha = Legion::Future::from_value<double>(rt, 1.0);
    v.axpy(alpha, w);
    v.print();
    w.print();
}



int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", false, false
    );
    LegionSolvers::AxpyTask<float, 1>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<float, 2>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<float, 3>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<double, 1>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<double, 2>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<double, 3>::preregister_kokkos(true);
    LegionSolvers::RandomFillTask<float, 1>::preregister_cpu(true);
    LegionSolvers::RandomFillTask<float, 2>::preregister_cpu(true);
    LegionSolvers::RandomFillTask<float, 3>::preregister_cpu(true);
    LegionSolvers::RandomFillTask<double, 1>::preregister_cpu(true);
    LegionSolvers::RandomFillTask<double, 2>::preregister_cpu(true);
    LegionSolvers::RandomFillTask<double, 3>::preregister_cpu(true);
    LegionSolvers::PrintVectorTask<float, 1>::preregister_cpu(true);
    LegionSolvers::PrintVectorTask<float, 2>::preregister_cpu(true);
    LegionSolvers::PrintVectorTask<float, 3>::preregister_cpu(true);
    LegionSolvers::PrintVectorTask<double, 1>::preregister_cpu(true);
    LegionSolvers::PrintVectorTask<double, 2>::preregister_cpu(true);
    LegionSolvers::PrintVectorTask<double, 3>::preregister_cpu(true);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
