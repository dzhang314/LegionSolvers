#include "LinearAlgebraTasks.hpp"

#include <tuple>
#include <typeinfo>


template <typename T, int N>
template <typename KokkosExecutionSpace>
void LegionSolvers::ScalTask<T, N>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // ScalTask::announce_kokkos(
    //     task->index_point, typeid(KokkosExecutionSpace), ctx, rt
    // );

    assert(regions.size() == 1);
    const auto &x = regions[0];

    assert(task->regions.size() == 1);
    const auto &x_req = task->regions[0];

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    assert(task->futures.size() == 1);
    const T alpha = task->futures[0].get_result<T>();

    Legion::FieldAccessor<
        LEGION_READ_WRITE, T, N, Legion::coord_t,
        Realm::AffineAccessor<T, N, Legion::coord_t>
    > x_writer{x, x_fid};

    const Legion::Domain x_domain = rt->get_index_space_domain(
        ctx, x_req.region.get_index_space()
    );

    for (Legion::RectInDomainIterator<N> it{x_domain}; it(); ++it) {
        const Legion::Rect<N> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, N>::create(
                rect, ctx, rt
            ),
            KokkosScalFunctor<KokkosExecutionSpace, T, N>{
                x_writer.accessor, alpha
            }
        );
    }
}


template <typename T, int N>
template <typename KokkosExecutionSpace>
void LegionSolvers::AxpyTask<T, N>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // AxpyTask::announce_kokkos(
    //     task->index_point, typeid(KokkosExecutionSpace), ctx, rt
    // );

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


template <typename T, int N>
template <typename KokkosExecutionSpace>
void LegionSolvers::XpayTask<T, N>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // XpayTask::announce_kokkos(
    //     task->index_point, typeid(KokkosExecutionSpace), ctx, rt
    // );

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
            KokkosXpayFunctor<KokkosExecutionSpace, T, N>{
                y_writer.accessor, alpha, x_reader.accessor
            }
        );
    }
}

template <typename T, int N>
template <typename KokkosExecutionSpace>
T LegionSolvers::DotTask<T, N>::
KokkosTaskTemplate<KokkosExecutionSpace>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // DotTask::announce_kokkos(
    //     task->index_point, typeid(KokkosExecutionSpace), ctx, rt
    // );

    assert(regions.size() == 2);
    const auto &v = regions[0];
    const auto &w = regions[1];

    assert(task->regions.size() == 2);
    const auto &v_req = task->regions[0];
    const auto &w_req = task->regions[1];

    assert(v_req.privilege_fields.size() == 1);
    const Legion::FieldID v_fid = *v_req.privilege_fields.begin();

    assert(w_req.privilege_fields.size() == 1);
    const Legion::FieldID w_fid = *w_req.privilege_fields.begin();

    Legion::FieldAccessor<
        LEGION_READ_ONLY, T, N, Legion::coord_t,
        Realm::AffineAccessor<T, N, Legion::coord_t>
    > v_reader{v, v_fid}, w_reader{w, w_fid};

    const Legion::Domain v_domain = rt->get_index_space_domain(
        ctx, v_req.region.get_index_space()
    );

    const Legion::Domain w_domain = rt->get_index_space_domain(
        ctx, w_req.region.get_index_space()
    );

    assert(v_domain == w_domain);

    T result = static_cast<T>(0);
    for (Legion::RectInDomainIterator<N> it{v_domain}; it(); ++it) {
        const Legion::Rect<N> rect = *it;
        T temp = static_cast<T>(0);
        Kokkos::parallel_reduce(
            KokkosRangeFactory<KokkosExecutionSpace, N>::create(
                rect, ctx, rt
            ),
            KokkosDotFunctor<KokkosExecutionSpace, T, N>{
                v_reader.accessor, w_reader.accessor
            },
            temp
        );
        result += temp;
    }
    return result;
}


template void LegionSolvers::ScalTask<float, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DotTask<float, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3>::KokkosTaskTemplate<Kokkos::Cuda>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);