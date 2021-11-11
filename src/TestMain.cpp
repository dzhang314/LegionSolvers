#include <legion.h>

#include "COOMatrixTasks.hpp"
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

    LegionSolvers::DistributedVectorT<double, 1, Legion::coord_t, 1, Legion::coord_t> u{"u", index_space, color_space, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, Legion::coord_t, 1, Legion::coord_t> v{"v", u.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, Legion::coord_t, 1, Legion::coord_t> w{"w", u.index_partition, ctx, rt};
    LegionSolvers::DistributedVectorT<double, 1, Legion::coord_t, 1, Legion::coord_t> x{"x", u.index_partition, ctx, rt};

    u = 0.0;
    v = 0.0;
    w.random_fill();
    x.random_fill();

    v.axpy(2.0, w);
    u.axpy(-1.0, x);
    v.axpy(2.0, u);
    v.xpay(0.5, x);
    v.axpy(-1.0, w);

    v.print();

}



int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", false, false
    );
    LegionSolvers::ScalTask<float, 1>::preregister_kokkos(true);
    LegionSolvers::ScalTask<float, 2>::preregister_kokkos(true);
    LegionSolvers::ScalTask<float, 3>::preregister_kokkos(true);
    LegionSolvers::ScalTask<double, 1>::preregister_kokkos(true);
    LegionSolvers::ScalTask<double, 2>::preregister_kokkos(true);
    LegionSolvers::ScalTask<double, 3>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<float, 1>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<float, 2>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<float, 3>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<double, 1>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<double, 2>::preregister_kokkos(true);
    LegionSolvers::AxpyTask<double, 3>::preregister_kokkos(true);
    LegionSolvers::XpayTask<float, 1>::preregister_kokkos(true);
    LegionSolvers::XpayTask<float, 2>::preregister_kokkos(true);
    LegionSolvers::XpayTask<float, 3>::preregister_kokkos(true);
    LegionSolvers::XpayTask<double, 1>::preregister_kokkos(true);
    LegionSolvers::XpayTask<double, 2>::preregister_kokkos(true);
    LegionSolvers::XpayTask<double, 3>::preregister_kokkos(true);
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
    LegionSolvers::COOMatvecTask<float, 1, 1, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 1, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 1, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 2, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 2, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 2, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 3, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 3, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 1, 3, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 1, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 1, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 1, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 2, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 2, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 2, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 3, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 3, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 2, 3, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 1, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 1, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 1, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 2, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 2, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 2, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 3, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 3, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<float, 3, 3, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 1, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 1, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 1, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 2, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 2, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 2, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 3, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 3, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 1, 3, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 1, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 1, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 1, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 2, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 2, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 2, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 3, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 3, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 2, 3, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 1, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 1, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 1, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 2, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 2, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 2, 3>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 3, 1>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 3, 2>::preregister_kokkos(true);
    LegionSolvers::COOMatvecTask<double, 3, 3, 3>::preregister_kokkos(true);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
