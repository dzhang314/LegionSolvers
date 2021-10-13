#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP

#include <cmath> // for std::fma

#include <Kokkos_Core.hpp>
#include <legion.h>

#include "KokkosUtilities.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename ExecutionSpace, typename T, int N>
    struct KokkosScalFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, T, N> x_view;
        const T alpha;

        explicit KokkosScalFunctor(
            Realm::AffineAccessor<T, N, Legion::coord_t> x_accessor,
            const T &a
        ) : x_view{x_accessor}, alpha{a} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a) const {
            x_view(a) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b) const {
            x_view(a, b) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, int c) const {
            x_view(a, b, c) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d
        ) const {
            x_view(a, b, c, d) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e
        ) const {
            x_view(a, b, c, d, e) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f
        ) const {
            x_view(a, b, c, d, e, f) *= alpha;
        }

    }; // struct KokkosScalFunctor


    template <typename T, int N>
    struct ScalTask : TaskTD<SCAL_TASK_BLOCK_ID, ScalTask, T, N> {

        static constexpr const char *task_base_name = "scal";

        static constexpr bool is_leaf = true;

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct ScalTask


    template <typename ExecutionSpace, typename T, int N>
    struct KokkosAxpyFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, T, N> y_view;
        const T alpha;
        const KokkosConstOffsetView<ExecutionSpace, T, N> x_view;

        explicit KokkosAxpyFunctor(
            Realm::AffineAccessor<T, N, Legion::coord_t> y_accessor,
            const T &a,
            Realm::AffineAccessor<T, N, Legion::coord_t> x_accessor
        ) : y_view{y_accessor}, alpha{a}, x_view{x_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a) const {
            using std::fma;
            y_view(a) = fma(
                alpha,
                x_view(a),
                y_view(a)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b) const {
            using std::fma;
            y_view(a, b) = fma(
                alpha,
                x_view(a, b),
                y_view(a, b)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, int c) const {
            using std::fma;
            y_view(a, b, c) = fma(
                alpha,
                x_view(a, b, c),
                y_view(a, b, c)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d
        ) const {
            using std::fma;
            y_view(a, b, c, d) = fma(
                alpha,
                x_view(a, b, c, d),
                y_view(a, b, c, d)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e
        ) const {
            using std::fma;
            y_view(a, b, c, d, e) = fma(
                alpha,
                x_view(a, b, c, d, e),
                y_view(a, b, c, d, e)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f
        ) const {
            using std::fma;
            y_view(a, b, c, d, e, f) = fma(
                alpha,
                x_view(a, b, c, d, e, f),
                y_view(a, b, c, d, e, f)
            );
        }

    }; // struct KokkosAxpyFunctor


    template <typename T, int N>
    struct AxpyTask : TaskTD<AXPY_TASK_BLOCK_ID, AxpyTask, T, N> {

        static constexpr const char *task_base_name = "axpy";

        static constexpr bool is_leaf = true;

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct AxpyTask


    template <typename ExecutionSpace, typename T, int N>
    struct KokkosXpayFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, T, N> y_view;
        const T alpha;
        const KokkosConstOffsetView<ExecutionSpace, T, N> x_view;

        explicit KokkosXpayFunctor(
            Realm::AffineAccessor<T, N, Legion::coord_t> y_accessor,
            const T &a,
            Realm::AffineAccessor<T, N, Legion::coord_t> x_accessor
        ) : y_view{y_accessor}, alpha{a}, x_view{x_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a) const {
            using std::fma;
            y_view(a) = fma(
                alpha,
                y_view(a),
                x_view(a)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b) const {
            using std::fma;
            y_view(a, b) = fma(
                alpha,
                y_view(a, b),
                x_view(a, b)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, int c) const {
            using std::fma;
            y_view(a, b, c) = fma(
                alpha,
                y_view(a, b, c),
                x_view(a, b, c)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d
        ) const {
            using std::fma;
            y_view(a, b, c, d) = fma(
                alpha,
                y_view(a, b, c, d),
                x_view(a, b, c, d)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e
        ) const {
            using std::fma;
            y_view(a, b, c, d, e) = fma(
                alpha,
                y_view(a, b, c, d, e),
                x_view(a, b, c, d, e)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f
        ) const {
            using std::fma;
            y_view(a, b, c, d, e, f) = fma(
                alpha,
                y_view(a, b, c, d, e, f),
                x_view(a, b, c, d, e, f)
            );
        }

    }; // struct KokkosXpayFunctor


    template <typename T, int N>
    struct XpayTask : TaskTD<XPAY_TASK_BLOCK_ID, XpayTask, T, N> {

        static constexpr const char *task_base_name = "xpay";

        static constexpr bool is_leaf = true;

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct XpayTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
