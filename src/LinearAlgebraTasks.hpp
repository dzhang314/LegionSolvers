#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP

#include <string>

#include <legion.h>

#include "KokkosUtilities.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T, int DIM>
    struct AxpyTask : TaskTD<AXPY_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "axpy"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

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

            const Legion::FieldAccessor<LEGION_READ_WRITE, T, DIM> y_writer{y, y_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> x_reader{x, x_fid};

            for (Legion::PointInDomainIterator<DIM> iter{y}; iter(); ++iter) {
                y_writer[*iter] = alpha * x_reader[*iter] + y_writer[*iter];
            }
        }

    }; // struct AxpyTask


    template <typename T, int DIM>
    struct XpayTask : TaskTD<XPAY_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "xpay"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

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

            const Legion::FieldAccessor<LEGION_READ_WRITE, T, DIM> y_writer{y, y_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> x_reader{x, x_fid};
            for (Legion::PointInDomainIterator<DIM> iter{y}; iter(); ++iter) {
                y_writer[*iter] = x_reader[*iter] + alpha * y_writer[*iter];
            }
        }

    }; // struct XpayTask


    template <typename KokkosExecutionSpace, typename T, int N>
    struct KokkosDotProductFunctor {

        using value_type = T;

        KokkosOffsetView<KokkosExecutionSpace, T, N> v_view;
        KokkosOffsetView<KokkosExecutionSpace, T, N> w_view;

        explicit KokkosDotProductFunctor(
            KokkosOffsetView<KokkosExecutionSpace, T, N> v,
            KokkosOffsetView<KokkosExecutionSpace, T, N> w
        ) : v_view{v}, w_view{w} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a, T &acc) const {
            acc += v_view(a) * w_view(a);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, T &acc) const {
            acc += v_view(a, b) * w_view(a, b);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, T &acc
        ) const {
            acc += v_view(a, b, c) * w_view(a, b, c);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, T &acc
        ) const {
            acc += v_view(a, b, c, d) * w_view(a, b, c, d);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, T &acc
        ) const {
            acc += v_view(a, b, c, d, e) * w_view(a, b, c, d, e);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f) * w_view(a, b, c, d, e, f);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, int g, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f, g) * w_view(a, b, c, d, e, f, g);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d,
            int e, int f, int g, int h, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f, g, h) *
                   w_view(a, b, c, d, e, f, g, h);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e,
            int f, int g, int h, int i, T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f, g, h, i) *
                   w_view(a, b, c, d, e, f, g, h, i);
        }

    }; // struct KokkosDotProductFunctor


    template <typename KokkosExecutionSpace, typename T, int N>
    struct DotProductTask : TaskTD<DOT_PRODUCT_TASK_BLOCK_ID, T, N> {

        static std::string task_name() { return "dot_product"; }

        static T task_body(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *rt) {

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

            KokkosOffsetView<KokkosExecutionSpace, T, N>
            v_view{v_reader.accessor}, w_view{w_reader.accessor};

            const Legion::Domain v_domain = rt->get_index_space_domain(
                ctx, v_req.region.get_index_space()
            );

            const Legion::Domain w_domain = rt->get_index_space_domain(
                ctx, w_req.region.get_index_space()
            );

            const Legion::DomainT<N> domain = v_domain.intersection(w_domain);

            T result = static_cast<T>(0);
            for (Legion::RectInDomainIterator<N> iter{domain}; iter(); ++iter) {
                const Legion::Rect<N> rect = *iter;
                T temp = static_cast<T>(0);
                Kokkos::parallel_reduce(
                    KokkosRangePolicyFactory<KokkosExecutionSpace, N>::create(
                        rect, ctx, rt
                    ),
                    KokkosDotProductFunctor<KokkosExecutionSpace, T, N>{
                        v_view, w_view
                    },
                    temp
                );
                result += temp;
            }
            return result;
        }

    }; // struct DotProductTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
