#include "LinearAlgebraTasks.hpp"

#include <tuple>


template <typename T, int N>
template <typename KokkosExecutionSpace>
void LegionSolvers::AxpyTask<T, N>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    AxpyTask::announce_kokkos(
        task->index_point, typeid(KokkosExecutionSpace), ctx, rt
    );

    assert(regions.size() == 2);
    const auto &y = regions[0];
    const auto &x = regions[1];

    assert(task->regions.size() == 2);
    const auto &y_req = task->regions[0];
    const auto &x_req = task->regions[1];

    assert(y_req.privilege_fields.size() == 1);
    const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    assert(task->futures.size() == 1);
    const T alpha = task->futures[0].get_result<T>();

    Legion::FieldAccessor<
        LEGION_READ_WRITE, T, N, Legion::coord_t,
        Realm::AffineAccessor<T, N, Legion::coord_t>
    > y_writer{y, y_fid};

    Legion::FieldAccessor<
        LEGION_READ_ONLY, T, N, Legion::coord_t,
        Realm::AffineAccessor<T, N, Legion::coord_t>
    > x_reader{x, x_fid};

    const Legion::Domain y_domain = rt->get_index_space_domain(
        ctx, y_req.region.get_index_space()
    );

    const Legion::Domain x_domain = rt->get_index_space_domain(
        ctx, x_req.region.get_index_space()
    );

    assert(y_domain == x_domain);

    for (Legion::RectInDomainIterator<N> it{y_domain}; it(); ++it) {
        const Legion::Rect<N> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, N>::create(
                rect, ctx, rt
            ),
            KokkosAxpyFunctor<KokkosExecutionSpace, T, N>{
                y_writer.accessor, alpha, x_reader.accessor
            }
        );
    }
}


[[maybe_unused]]
static constexpr auto instantiations = std::make_tuple(
    LegionSolvers::AxpyTask<float, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::AxpyTask<float, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::AxpyTask<float, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::AxpyTask<float, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::AxpyTask<float, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::AxpyTask<float, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::AxpyTask<float, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::AxpyTask<float, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::AxpyTask<float, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::AxpyTask<double, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::AxpyTask<double, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::AxpyTask<double, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::AxpyTask<double, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::AxpyTask<double, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::AxpyTask<double, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body,
    LegionSolvers::AxpyTask<double, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body,
    LegionSolvers::AxpyTask<double, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body,
    LegionSolvers::AxpyTask<double, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body
);
