#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP

#include <cmath> // for std::fma

#include <Kokkos_Core.hpp>
#include <legion.h>

#include "KokkosUtilities.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename ExecutionSpace,
              typename ENTRY_T, int DIM, typename COORD_T>
    struct KokkosScalFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, ENTRY_T, DIM> x_view;
        const ENTRY_T alpha;

        explicit KokkosScalFunctor(
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> x_accessor,
            const ENTRY_T &a
        ) : x_view{x_accessor}, alpha{a} {}

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a) const {
            x_view(a) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a, COORD_T b) const {
            x_view(a, b) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c
        ) const {
            x_view(a, b, c) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d
        ) const {
            x_view(a, b, c, d) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e
        ) const {
            x_view(a, b, c, d, e) *= alpha;
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e, COORD_T f
        ) const {
            x_view(a, b, c, d, e, f) *= alpha;
        }

    }; // struct KokkosScalFunctor


    template <typename ENTRY_T, int DIM, typename COORD_T>
    struct ScalTask : public TaskTDI<SCAL_TASK_BLOCK_ID, ScalTask,
                                     ENTRY_T, DIM, COORD_T> {

        static constexpr const char *task_base_name = "scal";

        static constexpr bool is_replicable = true;

        static constexpr bool is_inner = false;

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


    template <typename ExecutionSpace,
              typename ENTRY_T, int DIM, typename COORD_T>
    struct KokkosAxpyFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, ENTRY_T, DIM> y_view;
        const ENTRY_T alpha;
        const KokkosConstOffsetView<ExecutionSpace, ENTRY_T, DIM> x_view;

        explicit KokkosAxpyFunctor(
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> y_accessor,
            const ENTRY_T &a,
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> x_accessor
        ) : y_view{y_accessor}, alpha{a}, x_view{x_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a) const {
            using std::fma;
            y_view(a) = fma(
                alpha,
                x_view(a),
                y_view(a)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a, COORD_T b) const {
            using std::fma;
            y_view(a, b) = fma(
                alpha,
                x_view(a, b),
                y_view(a, b)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c
        ) const {
            using std::fma;
            y_view(a, b, c) = fma(
                alpha,
                x_view(a, b, c),
                y_view(a, b, c)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d
        ) const {
            using std::fma;
            y_view(a, b, c, d) = fma(
                alpha,
                x_view(a, b, c, d),
                y_view(a, b, c, d)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e
        ) const {
            using std::fma;
            y_view(a, b, c, d, e) = fma(
                alpha,
                x_view(a, b, c, d, e),
                y_view(a, b, c, d, e)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e, COORD_T f
        ) const {
            using std::fma;
            y_view(a, b, c, d, e, f) = fma(
                alpha,
                x_view(a, b, c, d, e, f),
                y_view(a, b, c, d, e, f)
            );
        }

    }; // struct KokkosAxpyFunctor


    template <typename ENTRY_T, int DIM, typename COORD_T>
    struct AxpyTask : public TaskTDI<AXPY_TASK_BLOCK_ID, AxpyTask,
                                     ENTRY_T, DIM, COORD_T> {

        static constexpr const char *task_base_name = "axpy";

        static constexpr bool is_replicable = true;

        static constexpr bool is_inner = false;

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


    template <typename ExecutionSpace,
              typename ENTRY_T, int DIM, typename COORD_T>
    struct KokkosXpayFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, ENTRY_T, DIM> y_view;
        const ENTRY_T alpha;
        const KokkosConstOffsetView<ExecutionSpace, ENTRY_T, DIM> x_view;

        explicit KokkosXpayFunctor(
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> y_accessor,
            const ENTRY_T &a,
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> x_accessor
        ) : y_view{y_accessor}, alpha{a}, x_view{x_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a) const {
            using std::fma;
            y_view(a) = fma(
                alpha,
                y_view(a),
                x_view(a)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a, COORD_T b) const {
            using std::fma;
            y_view(a, b) = fma(
                alpha,
                y_view(a, b),
                x_view(a, b)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c
        ) const {
            using std::fma;
            y_view(a, b, c) = fma(
                alpha,
                y_view(a, b, c),
                x_view(a, b, c)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d
        ) const {
            using std::fma;
            y_view(a, b, c, d) = fma(
                alpha,
                y_view(a, b, c, d),
                x_view(a, b, c, d)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e
        ) const {
            using std::fma;
            y_view(a, b, c, d, e) = fma(
                alpha,
                y_view(a, b, c, d, e),
                x_view(a, b, c, d, e)
            );
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e, COORD_T f
        ) const {
            using std::fma;
            y_view(a, b, c, d, e, f) = fma(
                alpha,
                y_view(a, b, c, d, e, f),
                x_view(a, b, c, d, e, f)
            );
        }

    }; // struct KokkosXpayFunctor


    template <typename ENTRY_T, int DIM, typename COORD_T>
    struct XpayTask : public TaskTDI<XPAY_TASK_BLOCK_ID, XpayTask,
                                     ENTRY_T, DIM, COORD_T> {

        static constexpr const char *task_base_name = "xpay";

        static constexpr bool is_replicable = true;

        static constexpr bool is_inner = false;

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


    template <typename ExecutionSpace,
              typename ENTRY_T, int DIM, typename COORD_T>
    struct KokkosDotFunctor {

        using value_type = ENTRY_T;

        const KokkosConstOffsetView<ExecutionSpace, ENTRY_T, DIM> v_view;
        const KokkosConstOffsetView<ExecutionSpace, ENTRY_T, DIM> w_view;

        explicit KokkosDotFunctor(
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> v_accessor,
            Realm::AffineAccessor<ENTRY_T, DIM, COORD_T> w_accessor
        ) : v_view{v_accessor}, w_view{w_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(COORD_T a, ENTRY_T &acc) const {
            acc += v_view(a) * w_view(a);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, ENTRY_T &acc
        ) const {
            acc += v_view(a, b) * w_view(a, b);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, ENTRY_T &acc
        ) const {
            acc += v_view(a, b, c) * w_view(a, b, c);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, ENTRY_T &acc
        ) const {
            acc += v_view(a, b, c, d) * w_view(a, b, c, d);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e, ENTRY_T &acc
        ) const {
            acc += v_view(a, b, c, d, e) * w_view(a, b, c, d, e);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            COORD_T a, COORD_T b, COORD_T c, COORD_T d, COORD_T e, COORD_T f,
            ENTRY_T &acc
        ) const {
            acc += v_view(a, b, c, d, e, f) * w_view(a, b, c, d, e, f);
        }

    }; // struct KokkosDotFunctor


    template <typename ENTRY_T, int DIM, typename COORD_T>
    struct DotTask : public TaskTDI<DOT_TASK_BLOCK_ID, DotTask,
                                    ENTRY_T, DIM, COORD_T> {

        static constexpr const char *task_base_name = "dot_product";

        static constexpr bool is_replicable = true;

        static constexpr bool is_inner = false;

        static constexpr bool is_leaf = true;

        using return_type = ENTRY_T;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static ENTRY_T task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct DotTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
