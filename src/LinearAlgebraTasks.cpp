#include "LinearAlgebraTasks.hpp"

#include <cassert>
#include <typeinfo>
#include <vector>


template <typename ENTRY_T>
inline ENTRY_T get_alpha(const std::vector<Legion::Future> &futures) {
    if (futures.size() == 0) {
        return static_cast<ENTRY_T>(1);
    } else if (futures.size() == 1) {
        return futures[0].get_result<ENTRY_T>();
    } else if (futures.size() == 2) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        return f0 / f1;
    } else if (futures.size() == 3) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        const ENTRY_T f2 = futures[2].get_result<ENTRY_T>();
        return f0 * f1 / f2;
    } else if (futures.size() == 4) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        const ENTRY_T f2 = futures[2].get_result<ENTRY_T>();
        const ENTRY_T f3 = futures[3].get_result<ENTRY_T>();
        return f0 * f1 / (f2 * f3);
    } else {
        assert(false);
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
template <typename KokkosExecutionSpace>
void LegionSolvers::ScalTask<ENTRY_T, DIM, COORD_T>::
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

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineWriter<ENTRY_T, DIM, COORD_T> x_writer{x, x_fid};

    const Legion::Domain x_domain = rt->get_index_space_domain(
        ctx, x_req.region.get_index_space()
    );

    for (Legion::RectInDomainIterator<DIM, COORD_T> it{x_domain}; it(); ++it) {
        const Legion::Rect<DIM, COORD_T> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, DIM, COORD_T>::create(ctx, rt,
                rect
            ),
            KokkosScalFunctor<KokkosExecutionSpace, ENTRY_T, DIM, COORD_T>{
                x_writer.accessor, alpha
            }
        );
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
template <typename KokkosExecutionSpace>
void LegionSolvers::AxpyTask<ENTRY_T, DIM, COORD_T>::
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

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineWriter<ENTRY_T, DIM, COORD_T> y_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain = rt->get_index_space_domain(
        ctx, y_req.region.get_index_space()
    );

    const Legion::Domain x_domain = rt->get_index_space_domain(
        ctx, x_req.region.get_index_space()
    );

    assert(y_domain == x_domain);

    for (Legion::RectInDomainIterator<DIM, COORD_T> it{y_domain}; it(); ++it) {
        const Legion::Rect<DIM, COORD_T> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, DIM, COORD_T>::create(ctx, rt,
                rect
            ),
            KokkosAxpyFunctor<KokkosExecutionSpace, ENTRY_T, DIM, COORD_T>{
                y_writer.accessor, alpha, x_reader.accessor
            }
        );
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
template <typename KokkosExecutionSpace>
void LegionSolvers::XpayTask<ENTRY_T, DIM, COORD_T>::
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

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineWriter<ENTRY_T, DIM, COORD_T> y_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain = rt->get_index_space_domain(
        ctx, y_req.region.get_index_space()
    );

    const Legion::Domain x_domain = rt->get_index_space_domain(
        ctx, x_req.region.get_index_space()
    );

    assert(y_domain == x_domain);

    for (Legion::RectInDomainIterator<DIM, COORD_T> it{y_domain}; it(); ++it) {
        const Legion::Rect<DIM, COORD_T> rect = *it;
        Kokkos::parallel_for(
            KokkosRangeFactory<KokkosExecutionSpace, DIM, COORD_T>::create(ctx, rt,
                rect
            ),
            KokkosXpayFunctor<KokkosExecutionSpace, ENTRY_T, DIM, COORD_T>{
                y_writer.accessor, alpha, x_reader.accessor
            }
        );
    }
}

template <typename ENTRY_T, int DIM, typename COORD_T>
template <typename KokkosExecutionSpace>
ENTRY_T LegionSolvers::DotTask<ENTRY_T, DIM, COORD_T>::
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

    AffineReader<ENTRY_T, DIM, COORD_T> v_reader{v, v_fid}, w_reader{w, w_fid};

    const Legion::Domain v_domain = rt->get_index_space_domain(
        ctx, v_req.region.get_index_space()
    );

    const Legion::Domain w_domain = rt->get_index_space_domain(
        ctx, w_req.region.get_index_space()
    );

    assert(v_domain == w_domain);

    ENTRY_T result = static_cast<ENTRY_T>(0);
    for (Legion::RectInDomainIterator<DIM, COORD_T> it{v_domain}; it(); ++it) {
        const Legion::Rect<DIM, COORD_T> rect = *it;
        ENTRY_T temp = static_cast<ENTRY_T>(0);
        Kokkos::parallel_reduce(
            KokkosRangeFactory<KokkosExecutionSpace, DIM, COORD_T>::create(ctx, rt,
                rect
            ),
            KokkosDotFunctor<KokkosExecutionSpace, ENTRY_T, DIM, COORD_T>{
                v_reader.accessor, w_reader.accessor
            },
            temp
        );
        result += temp;
    }
    return result;
}


template void LegionSolvers::ScalTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::ScalTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::AxpyTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AxpyTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::XpayTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::XpayTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template float  LegionSolvers::DotTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float  LegionSolvers::DotTask<float , 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 1, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 2, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, int      >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, unsigned >::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Serial>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::OpenMP>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DotTask<double, 3, long long>::KokkosTaskTemplate<Kokkos::Cuda  >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
